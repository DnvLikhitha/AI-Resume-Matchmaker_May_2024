import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import openai
import numpy as np
import os
import base64

# Ensure nltk resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# OpenAI API key setup
openai.api_key = 'sk-proj-jgxQFYKoAYUmMTKdS9LCT3BlbkFJ0LZfcIgh1d2jNil9FOA7'  # Replace with your actual OpenAI API key

def extract_text_from_pdf(pdf_file):
    try:
        document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenization, lemmatization and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]
    
    # Join tokens back into cleaned text
    cleaned_text = ' '.join(clean_tokens)
    
    # Additional cleaning steps
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with a single space
    return cleaned_text

def generate_embeddings(text):
    # Split text into chunks if necessary (OpenAI model limits)
    max_tokens = 8192
    tokens = text.split()
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    
    embeddings = []
    for chunk in chunks:
        chunk_text = ' '.join(chunk)
        response = openai.Embedding.create(model="text-embedding-ada-002", input=chunk_text)
        embedding = response['data'][0]['embedding']
        embeddings.append(embedding)
    
    # Average embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding / np.linalg.norm(avg_embedding)  # Normalize embedding vector

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)

def map_similarity_to_score(similarity):
    # Map similarity score directly from 0 to 100%
    return similarity * 100

def match_resume_with_jds(resume_file, jd_files):
    try:
        resume_text = extract_text_from_pdf(resume_file)
        cleaned_resume_text = clean_text(resume_text)
        resume_embedding = generate_embeddings(cleaned_resume_text)
        
        jd_scores = []

        for jd_file in jd_files:
            jd_text = extract_text_from_pdf(jd_file)
            if jd_text == "":
                continue
            cleaned_jd_text = clean_text(jd_text)
            jd_title = os.path.basename(jd_file.name)
            jd_embedding = generate_embeddings(cleaned_jd_text)

            # Calculate cosine similarity
            matching_score = cosine_similarity(resume_embedding, jd_embedding)
            
            jd_scores.append((jd_title, matching_score))

        # Sort JDs by matching score (descending)
        jd_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return only top 5 matches
        return jd_scores[:5]
    except Exception as e:
        st.error(f"Error matching resume with JDs: {e}")
        return []

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode()
        return base64_str
    except Exception as e:
        st.error(f"Error reading image file: {e}")
        return ""

def main():
    st.title("AI-Driven Resume and Job Description Alignment")
    
    # Apply custom CSS for background image
    image_path = "C:/Users/chdnv/Pictures/Screenshots/Screenshot 2024-07-05 175949.png"
    base64_image = image_to_base64(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{base64_image}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Upload Resume
    resume_file = st.file_uploader("Upload the Resume (PDF)", type="pdf")
    
    if resume_file:
        if st.button("Match"):
            # Example paths to JD PDFs
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
            
            jd_files = [open(jd_path, 'rb') for jd_path in jd_pdf_paths]
            
            jd_scores = match_resume_with_jds(resume_file, jd_files)
            
            st.subheader("Top 5 Matching JDs")
            for idx, (jd_title, score) in enumerate(jd_scores):
                mapped_score = map_similarity_to_score(score)
                st.write(f"{idx + 1}. Matching Score with {jd_title}: {mapped_score:.2f}%")
            
            # Display best match
            if jd_scores:
                best_match_title = jd_scores[0][0]
                st.subheader(f"Best Match: {best_match_title}")
            else:
                st.error("No match found!")

if __name__ == "__main__":
    main()
