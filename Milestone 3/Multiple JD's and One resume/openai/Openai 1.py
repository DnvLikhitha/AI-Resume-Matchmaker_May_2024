import fitz  # PyMuPDF for PDF extraction
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import openai
import numpy as np
import os
from collections import defaultdict

# Ensure nltk resources are available
nltk.download('punkt')
nltk.download('stopwords')

# OpenAI API key setup
openai.api_key = 'sk-proj-jgxQFYKoAYUmMTKdS9LCT3BlbkFJ0LZfcIgh1d2jNil9FOA7'  # Replace with your actual OpenAI API key

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No such file: '{pdf_path}'")
    
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def clean_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenization and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]
    
    # Join tokens back into cleaned text
    cleaned_text = ' '.join(clean_tokens)
    
    # Additional cleaning steps
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with a single space
    return cleaned_text

def generate_embeddings(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    embedding = response['data'][0]['embedding']
    return embedding

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def match_resume_with_jds(resume_path, jd_pdf_paths):
    try:
        resume_text = extract_text_from_pdf(resume_path)
        cleaned_resume_text = clean_text(resume_text)
        resume_embedding = generate_embeddings(cleaned_resume_text)
        
        jd_scores = defaultdict(float)
        
        for jd_path in jd_pdf_paths:
            jd_text = extract_text_from_pdf(jd_path)
            cleaned_jd_text = clean_text(jd_text)
            
            jd_title = os.path.basename(jd_path)
            jd_embedding = generate_embeddings(cleaned_jd_text)
            
            # Calculate cosine similarity
            matching_score = cosine_similarity(resume_embedding, jd_embedding)
            jd_scores[jd_title] = matching_score * 100  # Convert to percentage
        
        # Sort JDs by matching score in descending order and select top 5
        sorted_jd_scores = sorted(jd_scores.items(), key=lambda item: item[1], reverse=True)[:5]
        
        for jd_title, score in sorted_jd_scores:
            print(f"JD Title: {jd_title}, Matching Score: {score:.2f}%")
        
        # Best match
        best_match_title, best_match_score = sorted_jd_scores[0]
        print(f"Best match: {best_match_title} with a score of {best_match_score:.2f}%")
    
    except FileNotFoundError as e:
        print(f"File not found error: {e}")

def main():
    # Example paths to JD PDFs and resume PDF
    jd_pdf_paths = [
        r"C:\Users\chdnv\Desktop\Job Descriptions CyberSecurity\jd-web.pdf", 
        r"C:\Users\chdnv\Desktop\Job Descriptions CyberSecurity\jd-uiux.pdf", 
        r"C:\Users\chdnv\Desktop\Job Descriptions CyberSecurity\Jd-manager.pdf",
        r"c:\Users\chdnv\Desktop\Job Descriptions CyberSecurity\Cyber Security specialist JD1.pdf",
        r"C:\Users\chdnv\Desktop\Job Descriptions CyberSecurity\SoftwareEngineer.pdf",
        r"C:\Users\chdnv\Desktop\Job Descriptions CyberSecurity\systems_administrator.pdf",
        r"C:\Users\chdnv\Desktop\Job Descriptions CyberSecurity\jd-CyberSecurityAnalyst11.pdf",
        r"C:\Users\chdnv\Desktop\Job Descriptions CyberSecurity\Web-Developer-Job-Description-for-Java-Free-PDF-Template.pdf"
    ]
    
    resume_pdf_path = r"C:\Users\chdnv\Desktop\RESUMES\res-web.pdf"
    
    # Match the resume with the JDs
    match_resume_with_jds(resume_pdf_path, jd_pdf_paths)

if __name__ == "__main__":
    main()
