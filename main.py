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

# Initialize the T5 summarization pipeline
# Initialize the pipeline with `from_tf=True` to use TensorFlow weights
pipe = pipeline(
    "text2text-generation",
    model="arthd24/ext_abs_t5small",
    framework="tf"  # If you're using TensorFlow
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


def extractive_summary(text, model_extra, top_n=25):
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)

    if len(sentences) < 2:
        return text  # Return original text if not enough sentences for ranking

    # Generate sentence embeddings
    embeddings = model_extra.encode(sentences, show_progress_bar=False)
    print(f"Embeddings shape: {embeddings.shape}")  # Debugging statement

    # Compute similarity matrix
    similarity_matrix = np.dot(embeddings, embeddings.T)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")  # Debugging statement

    # Calculate centrality scores
    centrality_scores = degree_centrality_scores(similarity_matrix)
    print(f"Centrality scores: {centrality_scores}")  # Debugging statement

    # Get the most central sentences
    ranked_indices = np.argsort(-centrality_scores)[:top_n]
    top_sentences = [sentences[i] for i in ranked_indices]

    # Return the summary
    return ". ".join(top_sentences)


# Add preprocessing function for text cleaning
def preprocess_text(text):
    """
    Perform basic text preprocessing:
    - Remove multiple spaces
    - Remove URLs
    - Convert to lowercase (optional)
    - Keep punctuation
    - Remove extra dots (..)
    - Remove sentences containing 'figure' or 'table'
    - Remove citations like (Author, Year) or n.d.
    """
    # Remove URLs (links)
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Optionally, convert to lowercase (if needed)
    text = text.lower()

    # Remove extra dots (.. becomes .)
    text = re.sub(r'\.\.+', '.', text)

    # Remove sentences containing the words 'figure' or 'table' (case-insensitive)
    text = re.sub(r'([^.]*\b(figure|table)\b[^.]*\.)', '', text, flags=re.IGNORECASE)

    # Remove citations (e.g., (Author, Year) or n.d.)
    text = re.sub(r'\([^\)]+\d{4}[^\)]+\)', '', text)  # Removes (Author, Year) type citations
    text = re.sub(r'\bn\.d\.\b', '', text)  # Removes 'n.d.' citations

    # Remove cite format [number]
    text = re.sub(r'\[\s*\d+\s*\]', '', text)

    # Strip any leading/trailing spaces
    text = text.strip()

    return text


@app.post("/extractive-summary/", tags=["Summarization T5 Small"])
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Read the uploaded PDF file
        pdf_content = await file.read()

        # Save the file temporarily for PyPDF2 to read
        temp_file_path = "temp_uploaded.pdf"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(pdf_content)

        # Read the text from the PDF
        reader = PdfReader(temp_file_path)
        text = "".join([page.extract_text() for page in reader.pages])

        # Extract sections
        sections = extract_sections(text)

        # Preprocess each section's text
        for section in sections:
            sections[section] = preprocess_text(sections[section])

        # Combine sections except References
        combined_text = " ".join(
            [content for section, content in sections.items() if section.lower() != "references"]
        )

        # Preprocess combined text before summarization
        combined_text = preprocess_text(combined_text)

        # Generate extractive summary
        ext_summary = preprocess_text(extractive_summary(combined_text, model_sformer, top_n=10))

        # Use extractive summary as input to the abstractive model
        try:
            # Use the extractive summary as input to the abstractive model
            generated_text = pipe(
                ext_summary,  # Use the extractive summary here
                max_length=200,
                num_beams=4,
                length_penalty=1.0
            )[0].get('generated_text', 'No summary generated.')
        except Exception as e:
            generated_text = f"Error generating summary: {str(e)}"

        return {
            "sections": sections,
            "extractive_summary": ext_summary,
            "generated_text": generated_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    finally:
        # Clean up the temporary file
        import os
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


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
