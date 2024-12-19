from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from bs4 import BeautifulSoup
import json
import time
import re
import csv
import os

PATH = r"C:\Users\miase\Downloads\Deep-Learning-Project2\Drivers\chromedriver.exe"

options = Options()
options.add_argument("--start-maximized")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")

chrome_service = ChromeService(executable_path=PATH)
driver = webdriver.Chrome(service=chrome_service, options=options)

input_csv_file = r'C:\Users\miase\Downloads\Deep-Learning-Project2\Split 2020\split_20204.csv'

base_filename = os.path.splitext(os.path.basename(input_csv_file))[0]
output_json_file = os.path.join('C:\\Users\\miase\\Downloads\\Deep-Learning-Project2\\Output 2020 Json', f"{base_filename}.json")

with open(input_csv_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    results = []

    for row in reader:
        url = row['Link']  
        title = row['Title']  
        driver.get(url)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "section.body.main-article-body"))
            )
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            abstract_h2 = soup.select_one("section.abstract h2").get_text(strip=True) if soup.select_one("section.abstract h2") else ""
            abstract_p = soup.select_one("section.abstract p").get_text(strip=True) if soup.select_one("section.abstract p") else ""
            combined_abstract = f"{abstract_h2} {abstract_p}".strip() if abstract_h2 or abstract_p else "Tidak ditemukan"
            keywords_section = soup.select_one("section.kwd-group").get_text(strip=True) if soup.select_one("section.kwd-group") else "Tidak ditemukan"
            
            sections = []
            section_ids = set()  
            h2_elements = [] 
            for sec in soup.select("section[id^='sec']"):
                if re.match(r'^sec\d+-', sec['id']):  
                    sec_content = sec.get_text(strip=True)
                    if sec['id'] not in section_ids:  
                        sections.append(sec_content)
                        section_ids.add(sec['id'])  
                        h2_text = sec.find("h2").get_text(strip=True) if sec.find("h2") else ""
                        h2_elements.append(h2_text)  
            combined_sections = " ".join(sections) if sections else "Tidak ditemukan"
            section_count = len(sections)
            
            result = {
                'Link': url,
                'Title': title,
                'Abstract': combined_abstract,
                'Keywords': keywords_section,
                'Sections': combined_sections,
                'SectionCount': section_count,
                'SectionIDs': list(section_ids),  
                'H2Elements': h2_elements  
            }
            results.append(result) 
        
            print(f"Berhasil mengambil data dari URL: {url}")

        except Exception as e:
            print(f"Gagal mengambil data dari {url}: {e}")
            results.append({
                'Link': url,
                'Title': title,
                'Abstract': "Gagal mengambil data",
                'Keywords': "Gagal mengambil data",
                'Sections': "",
                'SectionCount': 0,
                'SectionIDs': [],
                'H2Elements': []
            })

        time.sleep(2)  

with open(output_json_file, mode='w', encoding='utf-8') as jsonfile:
    json.dump(results, jsonfile, ensure_ascii=False, indent=4)

driver.quit()
