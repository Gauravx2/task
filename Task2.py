import pandas as pd
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
GEMINI_API_KEY = "AIzaSyA8mOqEmhw5gdbn55POfVSMSRUjSEg0qvQ"
JOBS_FILE = "Paraform_Jobs.xlsx"
STUDENTS_CSV = "JuiceboxExport_1743820890826.csv"

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

def create_student_text(row):
    """Create semantic text for embedding from student data"""
    components = [
        row['Current Title'],
        row['Current Org Name'],
        row['Education'],
        row['Location'].split(',')[0]  
    ]
    return " ".join([str(c) for c in components if pd.notna(c)])

def get_job_embedding(job_text):
    """Generate embedding for job description"""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=job_text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return None

def get_student_embeddings(students_df):
    """Generate embeddings for all students"""
    embeddings = {}
    for idx, row in students_df.iterrows():
        student_text = create_student_text(row)
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=student_text,
                task_type="retrieval_query"
            )
            embeddings[idx] = result['embedding']
        except Exception as e:
            print(f"Error embedding student {row['First name']}: {str(e)}")
    return embeddings

def calculate_cultural_fit(student_row, job_data):
    """Calculate bonus points for cultural fit"""
    bonus = 0
    # Location match bonus
    if job_data['Locations'] and any(loc in student_row['Location'] for loc in job_data['Locations'].split(',')):
        bonus += 0.05
    
    # Education tier bonus 
    elite_schools = ['Stanford', 'MIT', 'Harvard']
    if any(school in student_row['Education'] for school in elite_schools):
        bonus += 0.03
    
    # Startup experience bonus
    if 'startup' in student_row['Current Org Name'].lower():
        bonus += 0.02
    
    return bonus

def match_students_to_job(job_title, top_n=10):
    """Main matching function with enhanced features"""
    # Load data
    jobs_df = pd.read_excel(JOBS_FILE)
    students_df = pd.read_csv(STUDENTS_CSV)
    
    # Get job data
    job = jobs_df[jobs_df['Role'].str.lower() == job_title.lower()].iloc[0]
    job_text = f"{job['Role']} {job['Requirements']} {job['Tech Stack']} {job['YOE']}"
    
    # Get embeddings
    job_embedding = get_job_embedding(job_text)
    student_embeddings = get_student_embeddings(students_df)
    
    # Calculate matches
    matches = []
    for student_id, student_embedding in student_embeddings.items():
        # Base similarity
        similarity = cosine_similarity([job_embedding], [student_embedding])[0][0]
        normalized_score = (similarity + 1) / 2
        
        # Cultural fit bonus
        normalized_score += calculate_cultural_fit(students_df.iloc[student_id], job)
        
        # Title match boost
        if job_title.lower() in students_df.iloc[student_id]['Current Title'].lower():
            normalized_score += 0.10
        
        matches.append((student_id, min(normalized_score, 1.0)))  # Cap at 1.0

    # Sort and get top matches
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # Prepare results
    results = []
    for match in matches[:top_n]:
        student = students_df.iloc[match[0]]
        results.append({
            'Name': f"{student['First name']} {student['Last name']}",
            'Current Title': student['Current Title'],
            'Organization': student['Current Org Name'],
            'Location': student['Location'],
            'Education': student['Education'],
            'LinkedIn': student['LinkedIn'],
            'Email': student['Personal Email'] if pd.notna(student['Personal Email']) else student['Work Email'],
            'Match Score': f"{match[1]:.1%}",
            'Contact Priority': "Risky" if 'Risky' in str(student['Work Email Verification']) else "Verified"
        })
    
    return pd.DataFrame(results)

def generate_outreach(student_data, job_data):
    """Generate personalized outreach message with error handling"""
    try:
        # Safely get student details with defaults
        first_name = student_data.get('First name', 'there')
        current_title = student_data.get('Current Title', 'your current position')
        current_org = student_data.get('Current Org Name', 'your organization')
        
        # Get job details
        company = job_data.get('Company', 'the company')
        role = job_data.get('Role', 'this position')
        tech_stack = job_data.get('Tech Stack', 'relevant technologies')
        
        # Build message
        message = (
            f"Hi {first_name},\n\n"
            f"I came across your profile and noticed your experience as {current_title} at {current_org} "
            f"could be a great fit for {company}'s {role} position. "
            f"Your background with {tech_stack} seems particularly relevant.\n\n"
            f"Would you be open to a quick chat about this opportunity?\n\n"
            f"Best regards,\n"
            f"[Your Name]\n"
            f"[Your Position]\n"
            f"[Your Contact Info]"
        )
        return message
    except Exception as e:
        print(f"Error generating outreach: {str(e)}")
        return "Could not generate message - missing data"

def main():
    # Show available jobs
    jobs_df = pd.read_excel(JOBS_FILE)
    print("\nAvailable Positions:")
    print(jobs_df[['Role']].to_string(index=False))
    
    # Get user input
    job_title = 'Software Engineer, Product'
    
    # Run matching
    results = match_students_to_job(job_title)
    
    # Display results
    print(f"\nTop Candidates for {job_title}:")
    print(results[['Name', 'Match Score', 'Current Title', 'Organization', 'Contact Priority']].to_string(index=False))
    
    # Generate sample outreach
    if not results.empty:
        job_data = jobs_df[jobs_df['Role'].str.lower() == job_title.lower()].iloc[0]
        sample = results.iloc[0]
        print("\nSample Outreach Message:")
        print(generate_outreach(sample, job_data))

if __name__ == "__main__":
    main()