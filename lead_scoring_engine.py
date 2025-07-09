# Lead Scoring Engine: Code + Documentation

## Architecture (Based on Provided Key Components)

### 🧩 Key Components:
# 1. **Data Ingestion**: Batch or real-time ingestion of lead data.
# 2. **pgvector Feature Store**: Store vector embeddings of lead-related text fields (e.g. job title, notes).
# 3. **Gradient Boosted Model** (LightGBM/XGBoost) for structured features.
# 4. **LLM Re-Ranker**: Rank high-probability leads using LLM-based heuristics.
# 5. **Serving**: FastAPI + Redis for real-time inference.
# 6. **Monitoring**: Daily retraining pipeline with drift detection.



## 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import pgvector.psycopg
import psycopg2
import openai
import os

## 2. Load and Ingest Data
# Simulating data load
data = pd.read_csv("leads.csv")
target = "converted"
X = data.drop(columns=[target])
y = data[target]

# Fill missing values
X = X.fillna({
    'industry': 'Unknown',
    'company_size': 'Unknown',
    'budget': X['budget'].median(),
    'page_visits': 0,
    'time_on_site': 0,
    'lead_source': 'Other',
})

## 3. Feature Engineering with Embeddings
# For simplicity, mock embeddings (Replace with actual OpenAI/SBERT etc)
def get_embedding(text):
    return np.random.rand(768)

X['job_title_emb'] = X['job_title'].apply(get_embedding)

# Connect to pgvector and store embeddings
conn = psycopg2.connect("dbname=leads user=postgres password=secret")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS lead_embeddings (
        lead_id TEXT PRIMARY KEY,
        embedding VECTOR(768)
    );
""")
for i, row in X.iterrows():
    cursor.execute("INSERT INTO lead_embeddings (lead_id, embedding) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                   (row['lead_id'], row['job_title_emb'].tolist()))
conn.commit()

## 4. Modeling with LightGBM
cat_cols = ['industry', 'company_size', 'lead_source']
num_cols = ['budget', 'page_visits', 'time_on_site']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = MinMaxScaler()
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("F1 Score:", f1_score(y_test, y_pred))

## 5. Save Model
joblib.dump(clf, "lead_model_gbm.pkl")

## 6. API with FastAPI and Redis + LLM Re-Ranker
# Run with: uvicorn main:app --reload
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import redis

app = FastAPI()
model = joblib.load("lead_model_gbm.pkl")
cache = redis.Redis(host='localhost', port=6379, db=0)
openai.api_key = os.getenv("OPENAI_API_KEY")

class LeadInput(BaseModel):
    lead_id: str
    industry: str
    company_size: str
    lead_source: str
    budget: float
    page_visits: int
    time_on_site: float
    job_title: str

@app.post("/score")
def score_lead(lead: LeadInput):
    if cache.get(lead.lead_id):
        return {"score": float(cache.get(lead.lead_id))}

    df = pd.DataFrame([lead.dict()])
    df['job_title_emb'] = get_embedding(lead.job_title)
    score = model.predict_proba(df)[0, 1] * 100

    # LLM Re-Ranker
    prompt = f"""
    A lead has a job title '{lead.job_title}', industry '{lead.industry}', and a lead score of {round(score)}.
    Should this lead be considered high-priority for sales follow-up? Justify in one line.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    rationale = response.choices[0].message['content']

    cache.set(lead.lead_id, score)
    return {"lead_id": lead.lead_id, "score": round(score, 2), "llm_rationale": rationale.strip()}





