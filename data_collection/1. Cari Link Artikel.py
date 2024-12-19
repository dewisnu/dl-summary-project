from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from bs4 import BeautifulSoup
import csv
import time

url = 'https://www.ncbi.nlm.nih.gov/pmc/?term=%22Behav%20Sci%20(Basel)%22%5Bjour%5D'
options = Options()
options.add_argument("--start-maximized")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")

PATH = r"C:\Users\Lenovo\Documents\Deep-Learning-Project\Drivers\chromedriver.exe"
chrome_service = ChromeService(executable_path=PATH)
driver = webdriver.Chrome(service=chrome_service, options=options)
driver.maximize_window()

driver.get(url)
data = []
max_pages = 170  
page_count = 0   

try:
    while page_count < max_pages:
       
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "rprt")))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        articles = soup.find_all(class_='rprt')
        
        for article in articles:
            # Ambil judul dan link
            title_tag = article.find(class_='title')
            title = title_tag.get_text(strip=True) if title_tag else 'N/A'
            link = "https://www.ncbi.nlm.nih.gov" + title_tag.find('a')['href'] if title_tag and title_tag.find('a') else 'N/A'
            
            # Ambil deskripsi (penulis)
            desc_tag = article.find(class_='desc')
            description = desc_tag.get_text(strip=True) if desc_tag else 'N/A'
            
            # Ambil detail tambahan
            details_tag = article.find(class_='details')
            details = details_tag.get_text(strip=True) if details_tag else 'N/A'
            
            data.append((title, link, description, details))
            print(f"Title: {title}, Link: {link}, Authors: {description}, Details: {details}")

        page_count += 1

        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "next"))
            )
            next_button.click()
            time.sleep(2)  
        except:
            print("Halaman terakhir telah dicapai atau tombol 'Next' tidak ditemukan.")
            break

    with open('articles_data.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Link', 'Authors', 'Details'])
        for entry in data:
            writer.writerow(entry)

finally:
    driver.quit()
