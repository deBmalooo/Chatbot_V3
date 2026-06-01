import os
import pickle
import re
import csv
import json
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv
from google import genai
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


# ================= GEMINI SETUP =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

api_key = os.getenv("GEMINI_API_KEY")


if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# Prefer concrete model names over moving aliases such as
# "gemini-flash-latest", which can route to temporarily overloaded models.
CHAT_MODELS = [
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.5-flash",
]


# ================= SAFE GEMINI WRAPPER =================

def safe_generate(prompt: str) -> str:
    last_error = None

    for model in CHAT_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            if hasattr(response, "text"):
                return response.text

            return str(response)

        except Exception as e:
            last_error = e
            print(f"REAL GEMINI ERROR ({model}):", str(e))

    # show REAL error instead of fake hardcoded message
    return f"Gemini Error: {str(last_error)}"

# ================= TF-IDF EMBEDDING =================

vectorizer = TfidfVectorizer()


def embed_texts(texts):
    vectors = vectorizer.fit_transform(texts)
    return vectors.toarray().astype("float32")


# ================= LOAD INDEX =================

def load_index(index_dir=None):
    if index_dir is None:
        index_dir = os.path.join(BASE_DIR, "index")

    index = faiss.read_index(
        os.path.join(index_dir, "faiss.index")
    )

    with open(
        os.path.join(index_dir, "docs.pkl"),
        "rb"
    ) as f:
        docs = pickle.load(f)

    return index, docs


# ================= RETRIEVAL =================

def retrieve_similar(query, faiss_index, docs, k=5):
    try:
        # Use same text corpus
        corpus = [doc["text"] for doc in docs]

        # Fit vectorizer on corpus
        vectorizer.fit(corpus)

        # Transform query
        q_vec = vectorizer.transform([query])

        q_vec = q_vec.toarray().astype("float32")

        # Search FAISS
        distances, indices = faiss_index.search(q_vec, k)

        print("Distances:", distances)
        print("Indices:", indices)

        results = []

        for rank, idx in enumerate(indices[0]):

            # Ignore invalid FAISS index
            if idx == -1:
                continue

            doc = docs[int(idx)].copy()

            doc["score"] = float(distances[0][rank])

            results.append(doc)

        return results

    except Exception as e:
        print("Retrieval Error:", str(e))
        return []
# ================= RAG PROMPT =================

def build_rag_prompt(
    context_docs: List[Dict],
    question: str
) -> str:

    context = "\n\n---\n\n".join(
        [
            f"Document {i+1}:\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        ]
    )

    return f"""
You are a helpful assistant for the College Placement Cell.

Use ONLY the context below.

If answer is not found, say:
"Data not found in records."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# ================= RAG =================


def answer_with_rag(question: str, faiss_index, docs, k: int = 20):

    question_lower = question.lower()

    try:

        # ---------------- YEAR ----------------
        year = None
        for y in ["2023", "2024", "2025"]:
            if y in question_lower:
                year = y

        # ---------------- DEPARTMENT ----------------
        department = None

        branches = [
            "cse", "eee", "ece",
            "it", "aeie", "me",
            "ce", "csbs"
        ]

        for branch in branches:
            if branch in question_lower:
                department = branch.upper()

        # ============================
        # TOTAL COMPANIES IN YEAR
        # ============================

        if (
            "how many companies" in question_lower
            or "total company" in question_lower
            or "total companies" in question_lower
        ):

            matched_companies = set()

            for d in docs:

                yr = str(d.get("year", ""))

                if year and yr != year:
                    continue

                company = str(d.get("company", "")).strip()

                if company:
                    matched_companies.add(company)

            company_list = sorted(list(matched_companies))

            return {
                "answer":
                f"There are {len(company_list)} companies in {year}:\n\n"
                + "\n".join(
                    [f"{i+1}. {c}" for i, c in enumerate(company_list)]
                ),

                "sources": []
            }

        # ============================
        # COUNT STUDENTS
        # ============================

        if (
            "number of students" in question_lower
            or "count" in question_lower
        ):

            matched_students = []

            for d in docs:

                dept = str(
                    d.get("department", "")
                ).upper()

                yr = str(
                    d.get("year", "")
                )

                if year and yr != year:
                    continue

                if department and dept != department:
                    continue

                matched_students.append(d)

            return {
                "answer":
                f"There are {len(matched_students)} students "
                f"from {department} who got placement in {year}.",

                "sources": matched_students[:10]
            }

        # ============================
        # HIGHEST HIRING COMPANY
        # ============================

        if (
            "highest placement" in question_lower
            or "highest hiring" in question_lower
            or "most students" in question_lower
        ):

            company_count = {}

            for d in docs:

                yr = str(d.get("year", ""))

                if year and yr != year:
                    continue

                company = d.get("company", "")

                if not company:
                    continue

                company_count[company] = (
                    company_count.get(company, 0)
                    + 1
                )

            highest = max(
                company_count,
                key=company_count.get
            )

            count = company_count[highest]

            return {
                "answer":
                f"{highest} hired the highest "
                f"number of students in {year} "
                f"({count} students).",

                "sources": []
            }

        # ============================
        # NORMAL RAG
        # ============================

        retrieved_docs = retrieve_similar(
            question,
            faiss_index,
            docs,
            k=k
        )

        prompt = build_rag_prompt(
            retrieved_docs,
            question
        )

        answer_text = safe_generate(prompt)

        return {
            "answer": answer_text,
            "sources": retrieved_docs
        }

    except Exception as e:

        print("REAL RAG ERROR:", str(e))

        return {
            "answer": f"RAG Error: {str(e)}",
            "sources": []
        }
# ================= FEW SHOT =================

def answer_with_few_shot(question: str):

    few_shot_prompt = f"""
You are a general AI assistant.

You DO NOT have access to placement database.

If asked for exact college placement records,
reply:
"I do not have access to actual records."

Question:
{question}

Answer:
"""

    try:
        answer_text = safe_generate(
            few_shot_prompt
        )

        return {
            "answer": answer_text,
            "sources": []
        }

    except Exception as e:
        print("Few Shot Error:", str(e))

        return {
            "answer":
            "⚠️ Internal server error during Few-Shot processing.",
            "sources": []
        }


# ================= ZERO SHOT =================

def answer_with_zero_shot(question: str):

    zero_shot_prompt = f"""
You are a helpful AI assistant.

Answer using general knowledge.

If unsure, say:
"I am not sure."

Question:
{question}

Answer:
"""

    try:
        answer_text = safe_generate(
            zero_shot_prompt
        )

        return {
            "answer": answer_text,
            "sources": []
        }

    except Exception as e:
        print("Zero Shot Error:", str(e))

        return {
            "answer":
            "⚠️ Internal server error during Zero-Shot processing.",
            "sources": []
        }
    
def _clean_tokens(text: str) -> set:
    stopwords = {
        "and", "the", "for", "with", "from", "that", "this", "have", "has",
        "are", "was", "were", "you", "your", "student", "students", "company",
        "department", "academic", "year", "name", "placed", "placement",
        "resume", "project", "college", "school", "email", "phone",
    }

    return {
        token
        for token in re.findall(r"[a-zA-Z0-9+#.]+", str(text).lower())
        if len(token) > 2 and token not in stopwords
    }


def _extract_list(value) -> list:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return []
    return [item.strip() for item in re.split(r"[,|;/\s]+", str(value)) if item.strip()]


def extract_resume_profile(resume_text: str) -> dict:
    """
    Extract structured resume signals with Gemini, then fall back to token parsing.
    """
    fallback_tokens = sorted(_clean_tokens(resume_text))
    known_departments = {"CSE", "IT", "ECE", "EEE", "EE", "ME", "CE", "AEIE", "CSBS"}
    inferred_departments = [
        dept for dept in known_departments
        if re.search(rf"\b{re.escape(dept)}\b", resume_text, re.I)
    ]
    fallback_profile = {
        "technical_skills": fallback_tokens[:40],
        "domains": [],
        "projects": [],
        "departments": sorted(inferred_departments),
    }

    prompt = f"""
Extract a concise JSON object from this resume text.

Return only valid JSON with these keys:
technical_skills: array of programming languages, tools, frameworks, databases, CS concepts
domains: array of domains such as AI, web development, data analysis, networking, embedded, manufacturing
projects: array of short project/domain phrases
departments: array of likely academic branches such as CSE, IT, ECE, EEE, ME, CE, AEIE, CSBS

RESUME:
{resume_text[:12000]}
"""

    response = safe_generate(prompt)

    try:
        json_text = response.strip()
        if "```" in json_text:
            json_text = re.sub(r"^```(?:json)?|```$", "", json_text, flags=re.MULTILINE).strip()
        profile = json.loads(json_text)
    except Exception:
        return fallback_profile

    return {
        "technical_skills": _extract_list(profile.get("technical_skills")) or fallback_profile["technical_skills"],
        "domains": _extract_list(profile.get("domains")),
        "projects": _extract_list(profile.get("projects")),
        "departments": [item.upper() for item in _extract_list(profile.get("departments"))] or fallback_profile["departments"],
    }


def load_company_profiles(docs: list) -> list:
    profile_path = os.path.join(BASE_DIR, "data", "company_profiles.csv")
    profiles = []

    if os.path.exists(profile_path):
        with open(profile_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                company = str(row.get("Company", "")).strip()
                if not company:
                    continue
                profiles.append({
                    "company": company,
                    "role": str(row.get("Role", "")).strip(),
                    "required_skills": _extract_list(row.get("Required Skills")),
                    "preferred_skills": _extract_list(row.get("Preferred Skills")),
                    "departments": [item.upper() for item in _extract_list(row.get("Departments"))],
                })

    existing = {profile["company"].upper() for profile in profiles}
    for doc in docs:
        company = str(doc.get("company", "")).strip()
        if company and company.upper() not in existing:
            profiles.append({
                "company": company,
                "role": "Placement opportunity",
                "required_skills": [],
                "preferred_skills": [],
                "departments": [],
            })
            existing.add(company.upper())

    return profiles


def _company_history(company: str, docs: list) -> dict:
    years = set()
    departments = set()
    count = 0

    for doc in docs:
        if str(doc.get("company", "")).strip().upper() != company.upper():
            continue

        count += 1
        year = str(doc.get("year", "")).strip()
        department = str(doc.get("department", "")).strip().upper()
        if year:
            years.add(year)
        if department:
            departments.add(department)

        stream_match = re.search(r"(?:Stream|Department):\s*([^|]+)", str(doc.get("text", "")), re.I)
        if stream_match:
            departments.add(stream_match.group(1).strip().upper())

        year_match = re.search(r"(?:Academic Year|Year):\s*(\d{4})", str(doc.get("text", "")), re.I)
        if year_match:
            years.add(year_match.group(1))

    return {
        "years": sorted(years),
        "departments": sorted(departments),
        "placement_count": count,
    }


def _overlap(source_items: list, target_items: list) -> list:
    source_tokens = _clean_tokens(" ".join(source_items))
    target_tokens = _clean_tokens(" ".join(target_items))
    return sorted(source_tokens.intersection(target_tokens))


def explain_company_match(suggestion: dict) -> str:
    skills = ", ".join(suggestion.get("matched_required", [])[:4] + suggestion.get("matched_preferred", [])[:3])
    missing = ", ".join(suggestion.get("missing_required", [])[:3])

    if skills and missing:
        return (
            f"Good fit for {suggestion['role']} because your resume matches {skills}. "
            f"To improve fit, strengthen {missing}."
        )
    if skills:
        return f"Good fit for {suggestion['role']} because your resume matches {skills}."
    if suggestion.get("department_match"):
        return "Potential fit because your branch aligns with this company's historical hiring pattern."
    return "Potential fit based on this company's past placement activity."

def match_resume_to_companies(resume_text: str, docs: list) -> list:
    """
    Rank companies using extracted resume skills, company skill profiles, and
    historical placement records.
    """
    resume_profile = extract_resume_profile(resume_text)
    resume_skills = (
        resume_profile["technical_skills"]
        + resume_profile["domains"]
        + resume_profile["projects"]
    )
    resume_departments = {item.upper() for item in resume_profile.get("departments", [])}
    profiles = load_company_profiles(docs)

    recommendations = []

    for company_profile in profiles:
        history = _company_history(company_profile["company"], docs)
        profile_departments = set(company_profile.get("departments") or history["departments"])
        matched_required = _overlap(resume_skills, company_profile.get("required_skills", []))
        matched_preferred = _overlap(resume_skills, company_profile.get("preferred_skills", []))
        resume_skill_tokens = _clean_tokens(" ".join(resume_skills))
        missing_required = [
            skill for skill in company_profile.get("required_skills", [])
            if _clean_tokens(skill).isdisjoint(resume_skill_tokens)
        ]
        department_match = bool(
            resume_departments
            and profile_departments
            and resume_departments.intersection(profile_departments)
        )

        history_score = min(history["placement_count"], 25) * 0.4
        match_score = (
            len(matched_required) * 12
            + len(matched_preferred) * 6
            + (10 if department_match else 0)
            + history_score
        )

        if match_score <= 0:
            continue

        suggestion = {
            "company": company_profile["company"],
            "role": company_profile["role"],
            "match_score": round(match_score, 2),
            "matched_keywords": sorted(set(matched_required + matched_preferred))[:8],
            "matched_required": matched_required[:8],
            "matched_preferred": matched_preferred[:8],
            "missing_required": missing_required[:5],
            "years": history["years"],
            "departments": sorted(profile_departments or set(history["departments"]))[:6],
            "placement_count": history["placement_count"],
            "department_match": department_match,
        }
        suggestion["explanation"] = explain_company_match(suggestion)
        recommendations.append(suggestion)

    recommendations.sort(key=lambda item: item["match_score"], reverse=True)

    return recommendations[:5]
