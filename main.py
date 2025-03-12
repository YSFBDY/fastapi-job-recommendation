import os
import re
import nltk
import uvicorn
import gc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.encoders import jsonable_encoder
from typing import List, Optional
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from functools import lru_cache

# Ensure NLTK resources are available (download manually before deployment)
nltk_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Lazy load model to reduce memory usage
@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Initialize the API application
app = FastAPI()

# Preprocessing function to clean and normalize text
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in nltk_stopwords]
    return ' '.join(words)

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
    return (0.9 * textual_similarity) + (0.1 * country_match)

# Health check endpoint
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
async def root():
    return {"message": "Service is running"}

# API endpoint for recommending jobs
@app.post("/recommend_jobs")
def recommend_jobs(request: JobMatchRequest):
    try:
        seeker_text = preprocess_text(request.seeker_headline + " " + " ".join(request.seeker_skills))
        seeker_embedding = get_model().encode(seeker_text).tolist()
        job_scores = []
        
        for job in request.jobs:
            job_text = preprocess_text(job.title + " " + (job.about or "") + " " + " ".join(job.skills or []))
            job_embedding = get_model().encode(job_text).tolist()
            score = calculate_score(request, job, seeker_embedding, job_embedding)
            if score >= 0.25:
                job_scores.append({"job_id": job.job_id, "title": job.title, "score": score})
        
        job_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Free up memory after processing
        gc.collect()
        
        return jsonable_encoder(job_scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server (only if executed directly)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
