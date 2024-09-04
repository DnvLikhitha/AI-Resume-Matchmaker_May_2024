import openai
import re
import fitz  # PyMuPDF
import os

# Replace with your actual OpenAI API key
openai.api_key = 'sk-proj-jgxQFYKoAYUmMTKdS9LCT3BlbkFJ0LZfcIgh1d2jNil9FOA7'

def check_resume_jd_match(resume, job_description):
    # Constructing the prompt
    prompt = f"""
    I have the following resume:

    {resume}

    And the following job description:

    {job_description}

    Please analyze the resume and the job description and determine how well they match. Provide a score out of 10 where 10 means an excellent match and 0 means no match at all. Additionally, provide a brief explanation for the score.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150  # You can adjust the number of tokens as needed
    )

    return response.choices[0].message['content'].strip()

def extract_score(response_text):
    # Extract the score from the response text
    match = re.search(r'\b(\d{1,2})\b', response_text)
    if match:
        return int(match.group(1))
    return None

def predict_resume_to_jd(resume, job_description):
    response_text = check_resume_jd_match(resume, job_description)
    score = extract_score(response_text)
    # Convert the score to a percentage out of 100
    score_percentage = score * 10 if score is not None else 0
    prediction = 1 if score_percentage >= 50 else 0
    return prediction, score_percentage, response_text

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a given PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: The extracted text.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    """
    Cleans the text by removing extra whitespace, special characters, and making it lowercase.
    
    Args:
        text (str): The text to be cleaned.
        
    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_pdf_path(prompt):
    while True:
        file_path = input(prompt).strip().strip('"')
        if os.path.splitext(file_path)[1].lower() == '.pdf':
            return file_path
        else:
            print("Please enter a valid PDF file path.")

def main():
    # Take input from user for resume and job description PDFs
    resume_pdf_path = get_pdf_path("Enter the path to the resume PDF file: ")
    jd_pdf_path = get_pdf_path("Enter the path to the job description PDF file: ")

    # Extract text from the PDF files
    try:
        resume_text = extract_text_from_pdf(resume_pdf_path)
        jd_text = extract_text_from_pdf(jd_pdf_path)
    except Exception as e:
        print(f"An error occurred while extracting text from the PDFs: {e}")
        return

    # Clean the extracted text
    resume_text_cleaned = clean_text(resume_text)
    jd_text_cleaned = clean_text(jd_text)

    # Check the match
    try:
        prediction, score_percentage, response_text = predict_resume_to_jd(resume_text_cleaned, jd_text_cleaned)
        print(f"Prediction: {prediction}")
        print(f"Matching Score: {score_percentage}%")
        print(f"Explanation: {response_text}")
    except Exception as e:
        print(f"An error occurred while checking the match: {e}")

if __name__ == "__main__":
    main()
