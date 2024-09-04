import streamlit as st
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to calculate match score using cosine similarity
def calculate_match_score(job_description, resume):
    # Combine job description and resume into a list
    documents = [job_description, resume]
    
    # Create a TfidfVectorizer to convert the text into TF-IDF vectors
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    # Calculate cosine similarity between the two vectors
    cosine_sim = cosine_similarity(vectors)
    
    # Return the similarity score between the job description and resume as percentage
    match_score = cosine_sim[0][1] * 100  # Multiply by 100 to get percentage
    return match_score

# Function to load an image and convert it to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image

# Path to the local image file
image_path = r"C:/Users/chdnv/Pictures/Screenshots/Screenshot 2024-07-05 175949.png"
base64_image = get_base64_image(image_path)

# Inject custom CSS to set a background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_image}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.title("AI-Driven Resume and Job Description Alignment")

# Upload Job Description PDF
st.header("Upload Job Description")
job_description_file = st.file_uploader("Choose a Job Description PDF file", type="pdf")

# Upload Resume PDF
st.header("Upload Resume")
resume_file = st.file_uploader("Choose a Resume PDF file", type="pdf")

# Match Button
if st.button("Match"):
    if job_description_file is not None and resume_file is not None:
        # Extract text from the uploaded PDF files
        job_description = extract_text_from_pdf(job_description_file)
        resume = extract_text_from_pdf(resume_file)
        
        # Call the matching function
        match_score = calculate_match_score(job_description, resume)

        # Display the match score as percentage
        st.write(f"Match Score: {match_score:.2f}%")
    else:
        st.write("Please upload both Job Description and Resume PDF files.")
