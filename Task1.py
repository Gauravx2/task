import os
import PyPDF2
import google.generativeai as genai
import pandas as pd  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Configuration
GEMINI_API_KEY = "AIzaSyA8mOqEmhw5gdbn55POfVSMSRUjSEg0qvQ"  # Be careful with exposed API keys
EXCEL_FILE = "Paraform_Jobs.xlsx"  # Local Excel file
RESUME_DIR = "resumes"  # Folder containing resumes

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with improved error handling"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text.strip()
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return ""

def get_job_data():
    """Read job data from local Excel file"""
    try:
        jobs_df = pd.read_excel(EXCEL_FILE)
        jobs = jobs_df.to_dict('records')
        
        structured_jobs = []
        for job in jobs:
            structured_jobs.append({
                'id': str(job.get('Link', '')).split('/')[-1],
                'text': (
                    f"Title: {job.get('Role', '')} at {job.get('Company', '')}\n"
                    f"Requirements: {job.get('Requirements', '')}\n"
                    f"Tech Stack: {job.get('Tech Stack', '')}\n"
                    f"Experience Needed: {job.get('YOE', '')} years\n"
                    f"Description: {job.get('One liner', '')}\n"
                    f"Industry: {job.get('Industry', '')}"
                ),
                'raw_data': job
            })
        return structured_jobs
    except Exception as e:
        print(f"Error reading job data: {str(e)}")
        return []

def get_embedding(text, model="models/embedding-001"):
    """Generate embeddings using Gemini with proper chunking"""
    if not text:
        return None
        
    try:
        # Gemini has a 2048 token limit - chunk conservatively
        chunk_size = 1500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        embeddings = []
        for chunk in chunks:
            result = genai.embed_content(
                model=model,
                content=chunk,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        
        return np.mean(embeddings, axis=0) if len(embeddings) > 1 else embeddings[0]
    except Exception as e:
        print(f"Embedding generation failed: {str(e)}")
        return None

def match_resumes_to_jobs():
    """Main matching function"""
    print("Starting resume-job matching process...")
    
    jobs = get_job_data()
    if not jobs:
        print("No job data found. Please check the Excel file.")
        return {}

    print(f"Loaded {len(jobs)} job postings")
    
    # Generate embeddings for all jobs
    job_embeddings = {}
    for job in jobs:
        embedding = get_embedding(job['text'])
        if embedding is not None:
            job_embeddings[job['id']] = embedding
    
    if not job_embeddings:
        print("Failed to generate job embeddings.")
        return {}

    # Process resumes
    results = {}
    for resume_file in os.listdir(RESUME_DIR):
        if resume_file.lower().endswith('.pdf'):
            resume_path = os.path.join(RESUME_DIR, resume_file)
            resume_text = extract_text_from_pdf(resume_path)
            
            if not resume_text:
                continue
                
            resume_embedding = get_embedding(resume_text)
            if resume_embedding is None:
                continue
                
            # Compare with all jobs
            scores = []
            for job in jobs:
                if job['id'] in job_embeddings:
                    similarity = cosine_similarity([resume_embedding], [job_embeddings[job['id']]])[0][0]
                    score = (similarity + 1) / 2  # Normalize to 0-1 scale
                    scores.append((job['id'], score))
            
            # Get top 2 matches
            top_matches = sorted(scores, key=lambda x: x[1], reverse=True)[:2]
            results[resume_file] = top_matches
    
    return results

def main():
    matches = match_resumes_to_jobs()
    jobs = {job['id']: job for job in get_job_data()}
    
    print("\n=== RESULTS ===")
    for resume, matches in matches.items():
        print(f"\nResume: {resume}")
        for job_id, score in matches:
            job = jobs[job_id]
            print(f"\nMatch: {job['raw_data']['Company']} - {job['raw_data']['Role']}")
            print(f"Score: {score:.2%}")
            print(f"Tech: {job['raw_data'].get('Tech Stack', 'N/A')}")
            print(f"Salary: {job['raw_data'].get('Salary', 'N/A')}")

if __name__ == "__main__":
    main()
