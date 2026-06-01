import os
import pickle
import pandas as pd
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------- LOAD CSV FILES ----------------

files = [
    "data/placement_2023.csv",
    "data/placement_2024.csv",
    "data/placement_2025.csv"
]

all_docs = []

for file in files:
    df = pd.read_csv(file)

    for _, row in df.iterrows():

        text = " | ".join(
            [f"{col}: {row[col]}" for col in df.columns]
        )

        all_docs.append({
            "text": text,
            "year": str(row.get("Academic Year", "")),
            "department": str(row.get("Department", "")),
            "company": str(row.get("Company", ""))
        })


print(f"Loaded {len(all_docs)} records")


# ---------------- TF-IDF EMBEDDING ----------------

vectorizer = TfidfVectorizer()

texts = [doc["text"] for doc in all_docs]

vectors = vectorizer.fit_transform(texts)

embeddings = vectors.toarray().astype("float32")


# ---------------- CREATE FAISS INDEX ----------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index created")


# ---------------- SAVE ----------------

os.makedirs("index", exist_ok=True)

faiss.write_index(index, "index/faiss.index")

with open("index/docs.pkl", "wb") as f:
    pickle.dump(all_docs, f)

with open("index/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Index saved successfully!")