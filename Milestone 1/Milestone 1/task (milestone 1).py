#!/usr/bin/env python
# coding: utf-8

# In[4]:


import fitz  # PyMuPDF
import re
import nltk
import spacy
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Spacy model outside the function for efficiency
nlp = spacy.load('en_core_web_sm')

def read_pdf(file_path):
    """Reads a PDF file and returns its text content."""
    pdf_text = ""
    try:
        document = fitz.open(file_path)
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            pdf_text += page.get_text()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None  # Return None if there's an error
    return pdf_text

def clean_text(text):
    """Cleans text by removing emails, URLs, non-alphabetic characters, and extra spaces."""
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def tokenize_text(text):
    """Tokenizes text into words."""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Removes stopwords from a list of tokens."""
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

def stem_tokens(tokens):
    """Stems a list of tokens."""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def lemmatize_tokens(tokens):
    """Lemmatizes a list of tokens."""
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

def normalize_text(text, use_stemming=False):
    """Normalizes text by cleaning, tokenizing, removing stopwords, and stemming or lemmatizing."""
    text = text.lower()  # Convert to lowercase
    text = clean_text(text)  # Clean text
    tokens = tokenize_text(text)  # Tokenize text
    tokens = remove_stopwords(tokens)  # Remove stop words
    if use_stemming:
        tokens = stem_tokens(tokens)  # Stem tokens
    else:
        tokens = lemmatize_tokens(tokens)  # Lemmatize tokens
    return ' '.join(tokens)

# Example usage
resume_file_path = 'Resume-Dnv-Likhitha.pdf'
job_description_file_path = 'sample-job-description.pdf'

resume_text = read_pdf(resume_file_path)
job_description_text = read_pdf(job_description_file_path)

# Check if PDFs were read successfully
if resume_text is not None and job_description_text is not None:
    normalized_resume_text = normalize_text(resume_text)
    normalized_job_description_text = normalize_text(job_description_text)

    # Write normalized text to files
    with open('resume.txt', 'w') as file:
        file.write(normalized_resume_text)
    with open('jobdescription.txt', 'w') as file:
        file.write(normalized_job_description_text)

    print("Done! Check out the files.")
    # Open the files with default system program
    os.startfile("resume.txt")
    os.startfile("jobdescription.txt")
else:
    print("Error reading PDFs. Check file paths and try again.")


# In[ ]:




