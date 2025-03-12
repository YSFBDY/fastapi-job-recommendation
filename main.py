from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.encoders import jsonable_encoder
from typing import List, Optional
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize the API application and load the pre-trained model
app = FastAPI()
model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

# Preprocessing function to clean and normalize text
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Define Pydantic models for input data
class Job(BaseModel):
    job_id: int
    title: str
    about: Optional[str] = ""
    skills: Optional[List[str]] = []
    country: Optional[str] = ""  # Updated field

class JobMatchRequest(BaseModel):
    seeker_headline: str
    seeker_skills: List[str]
    seeker_country: str  # Updated field
    jobs: List[Job]

# Function to calculate a combined score for job-seeker matching
def calculate_score(seeker, job, seeker_embedding, job_embedding):
    # Textual similarity using cosine similarity
    textual_similarity = cosine_similarity([seeker_embedding], [job_embedding])[0][0]

    # Country match (binary score)
    country_match = 1 if seeker.seeker_country == job.country else 0

    # Combine scores with weights
    total_score = (
        0.9 * textual_similarity +  # Weight for textual similarity
        0.1 * country_match         # Weight for country match
    )
    return total_score

# API endpoint for recommending jobs
@app.post("/recommend_jobs")
def recommend_jobs(request: JobMatchRequest):
    # Preprocess and encode seeker profile
    seeker_text = preprocess_text(request.seeker_headline + " " + " ".join(request.seeker_skills))
    seeker_embedding = model.encode(seeker_text).tolist()

    job_scores = []
    for job in request.jobs:
        # Preprocess and encode job description
        job_text = preprocess_text(job.title + " " + (job.about or "") + " " + " ".join(job.skills or []))
        job_embedding = model.encode(job_text).tolist()

        # Calculate combined score
        score = calculate_score(
            seeker=request,
            job=job,
            seeker_embedding=seeker_embedding,
            job_embedding=job_embedding
        )

        # Filter low matches
        if score >= 0.25:
            job_scores.append({"job_id": job.job_id, "title": job.title, "score": score})

    # Sort by highest match score
    job_scores.sort(key=lambda x: x["score"], reverse=True)

    # Return serialized results
    return jsonable_encoder(job_scores)
