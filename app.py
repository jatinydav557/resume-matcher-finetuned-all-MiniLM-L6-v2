import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

# Load fine-tuned model
model = SentenceTransformer("./finetuned_resume_job_similarity_model")

st.title("🔍 Resume Matcher")

# Upload resume
uploaded_file = st.file_uploader("📄 Upload your resume (PDF only):", type=["pdf"])

# Job description input
job_input = st.text_area("Enter a Job Description:")

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Resume processing
if uploaded_file is not None:
    uploaded_text = extract_text_from_pdf(uploaded_file)

    if st.button("Calculate Similarity"):
        if not job_input.strip():
            st.warning("Please enter a job description.")
        else:
            # Encode and calculate similarity
            embeddings = model.encode([uploaded_text, job_input], convert_to_tensor=True)
            similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()
            st.success(f"Similarity Score: {similarity_score:.4f}")
else:
    st.info("Please upload a PDF resume to begin.")
