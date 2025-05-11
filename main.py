# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.encoders import jsonable_encoder
from typing import List, Optional
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# For running server with ngrok
import nest_asyncio
from pyngrok import ngrok
import uvicorn

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the FastAPI app
app = FastAPI()

# Load the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Request Models
class Job(BaseModel):
    job_id: int
    title: str
    about: str
    skills: List[str]
    country: str

class JobMatchRequest(BaseModel):
    seeker_headline: str
    seeker_skills: Optional[List[str]] = None
    seeker_country: str
    jobs: List[Job]

# Matching logic
def calculate_score(seeker, job, seeker_embedding, job_embedding):
    textual_similarity = cosine_similarity([seeker_embedding], [job_embedding])[0][0]
    country_match = 1 if seeker.seeker_country.strip().lower() == job.country.strip().lower() else 0
    total_score = 0.9 * textual_similarity + 0.1 * country_match
    return total_score

# API route
@app.post("/recommend_jobs")
def recommend_jobs(request: JobMatchRequest):
    # Preprocess seeker text
    seeker_text = preprocess_text(request.seeker_headline)

    # Include seeker_skills if available
    if request.seeker_skills:
        seeker_text += " " + " ".join(request.seeker_skills)

    seeker_embedding = model.encode(seeker_text).tolist()

    job_scores = []
    for job in request.jobs:
        job_text = preprocess_text(job.title + " " + job.about + " " + " ".join(job.skills))
        job_embedding = model.encode(job_text).tolist()

        score = calculate_score(
            seeker=request,
            job=job,
            seeker_embedding=seeker_embedding,
            job_embedding=job_embedding
        )

        if score >= 0:
            job_scores.append({
                "job_id": job.job_id,
                "title": job.title,
                "score": score
            })

    job_scores.sort(key=lambda x: x["score"], reverse=True)
    return jsonable_encoder(job_scores)

# Main entry
if __name__ == "__main__":
    # Allow nested asyncio loop (needed for pyngrok + uvicorn)
    nest_asyncio.apply()

    # Connect ngrok to your local server
    public_url = ngrok.connect(8000)
    print("🔗 Public URL:", public_url)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
