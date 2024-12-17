from fastapi import FastAPI, UploadFile, File, HTTPException
from PyPDF2 import PdfReader
import re

from fastapi.openapi.utils import get_openapi
from sentence_transformers import SentenceTransformer
import nltk
from transformers import pipeline
import numpy as np

app = FastAPI()

nltk.download('punkt_tab')

pipe_pegasus = pipeline(
    "text2text-generation",
    model="arthd24/ext_abs_pegasus_all",  # Pegasus model
    framework="tf"  # TensorFlow
)

# Initialize the SentenceTransformer model
model_sformer = SentenceTransformer("all-MiniLM-L6-v2")


# Function to extract sections
def extract_sections(text: str):
    section_patterns = {
        "Introduction": r"(?:Introduction|INTRODUCTION)\s*\n",
        "Materials and Methods": r"(?:Materials and Methods|METHODS|Methodology|Methods|Method|METHOD)\s*\n",
        "Results": r"(?:Results|RESULTS)\s*\n",
        "Discussion": r"(?:Discussion|DISCUSSION)\s*\n",
        "Conclusions": r"(?:Conclusions?|CONCLUSIONS?)\s*\n",
    }
    sections = {}
    section_positions = []

    for section_name, pattern in section_patterns.items():
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in matches:
            section_positions.append((match.start(), section_name))

    section_positions.sort()

    for i in range(len(section_positions)):
        start_pos = section_positions[i][0]
        section_name = section_positions[i][1]

        if i == len(section_positions) - 1:
            content = text[start_pos:].strip()
        else:
            end_pos = section_positions[i + 1][0]
            content = text[start_pos:end_pos].strip()

        content = re.sub(section_patterns[section_name], "", content, flags=re.IGNORECASE)
        sections[section_name] = content.strip()

    return sections


# Function to compute sentence embeddings and LexRank scores
def degree_centrality_scores(similarity_matrix):
    markov_matrix = similarity_matrix / similarity_matrix.sum(axis=1, keepdims=True)
    scores = stationary_distribution(markov_matrix)
    return scores


def stationary_distribution(transition_matrix):
    n = len(transition_matrix)
    eigenvector = np.ones(n)
    transition = transition_matrix.transpose()

    for _ in range(10000):
        next_vector = np.dot(transition, eigenvector)
        if np.allclose(next_vector, eigenvector):
            return next_vector
        eigenvector = next_vector

    return eigenvector


def extractive_summary(text, model_extra, top_n=20):
    sentences = nltk.sent_tokenize(text)

    if len(sentences) < 2:
        return text

    embeddings = model_extra.encode(sentences, show_progress_bar=False)

    similarity_matrix = np.dot(embeddings, embeddings.T)

    centrality_scores = degree_centrality_scores(similarity_matrix)

    ranked_indices = np.argsort(-centrality_scores)[:top_n]
    top_sentences = [sentences[i] for i in ranked_indices]

    return ". ".join(top_sentences)


# Add preprocessing function for text cleaning
def preprocess_text(text):
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r'([^.]*\b(figure|table)\b[^.]*\.)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\([^\)]+\d{4}[^\)]+\)', '', text)
    text = re.sub(r'\bn\.d\.\b', '', text)
    text = re.sub(r'\[\s*\d+\s*\]', '', text)
    return text.strip()


# Common logic for PDF processing
async def process_file(file: UploadFile, pipe_model):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        pdf_content = await file.read()

        temp_file_path = "temp_uploaded.pdf"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(pdf_content)

        reader = PdfReader(temp_file_path)
        text = "".join([page.extract_text() for page in reader.pages])

        sections = extract_sections(text)

        for section in sections:
            sections[section] = preprocess_text(sections[section])

        combined_text = " ".join(
            [content for section, content in sections.items() if section.lower() != "references"]
        )

        combined_text = preprocess_text(combined_text)

        ext_summary = preprocess_text(extractive_summary(combined_text, model_sformer, top_n=10))

        try:
            generated_text = pipe_model(
                ext_summary,
                min_length=200,
                max_length=250,
                num_beams=4,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
            )[0].get('generated_text', 'No summary generated.')
        except Exception as e:
            generated_text = f"Error generating summary: {str(e)}"

        return {
            "sections": sections,
            "extractive_summary": ext_summary,
            "generated_text": generated_text
        }

    finally:
        import os
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/summarization-pegasus/", tags=["Summarization Pegasus"])
async def summarization_pegasus(file: UploadFile = File(...)):
    return await process_file(file, pipe_pegasus)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="summarization API",
        version="1.0.0",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
