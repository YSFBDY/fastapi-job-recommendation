import os
import re
import nltk
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.encoders import jsonable_encoder
from typing import List, Optional
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the API application
app = FastAPI()

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

# Preprocessing function to clean and normalize text
def preprocess_text(text: str) -> str:
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Define Pydantic models for input data
class Job(BaseModel):
    job_id: int
    title: str
    about: Optional[str] = ""
    skills: Optional[List[str]] = []
    country: Optional[str] = ""

class JobMatchRequest(BaseModel):
    seeker_headline: str
    seeker_skills: List[str]
    seeker_country: str
    jobs: List[Job]

# Function to calculate a combined score for job-seeker matching
def calculate_score(seeker: JobMatchRequest, job: Job, seeker_embedding: List[float], job_embedding: List[float]) -> float:
    textual_similarity = cosine_similarity([seeker_embedding], [job_embedding])[0][0]
    country_match = 1 if seeker.seeker_country.lower() == job.country.lower() else 0
    total_score = (0.9 * textual_similarity) + (0.1 * country_match)
    return total_score

# Health check endpoint
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)  # Add this line
async def root():
    return {"message": "Service is running"}

# API endpoint for recommending jobs
@app.post("/recommend_jobs")
def recommend_jobs(request: JobMatchRequest):
    try:
        seeker_text = preprocess_text(request.seeker_headline + " " + " ".join(request.seeker_skills))
        seeker_embedding = model.encode(seeker_text).tolist()
        job_scores = []
        
        for job in request.jobs:
            job_text = preprocess_text(job.title + " " + (job.about or "") + " " + " ".join(job.skills or []))
            job_embedding = model.encode(job_text).tolist()
            score = calculate_score(request, job, seeker_embedding, job_embedding)
            if score >= 0.25:
                job_scores.append({"job_id": job.job_id, "title": job.title, "score": score})
        
        job_scores.sort(key=lambda x: x["score"], reverse=True)
        return jsonable_encoder(job_scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server (only if executed directly)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Default to 10000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)