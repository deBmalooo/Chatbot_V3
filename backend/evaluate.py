import re
import time
from collections import Counter
# from rag_utils import load_index, answer_with_rag, answer_with_few_shot
from rag_utils import load_index, answer_with_rag, answer_with_few_shot, answer_with_zero_shot


# ---------------- LOAD INDEX ----------------

faiss_index, docs = load_index("index")


# ---------------- TEXT NORMALIZATION ----------------

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# ---------------- EXACT MATCH ----------------

def exact_match(prediction, ground_truth):
    return normalize_text(prediction) == normalize_text(ground_truth)


# ---------------- F1 SCORE ----------------

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


# ---------------- EXTRACT NUMBERS ----------------

def extract_numbers(text):
    return re.findall(r'\d+', str(text))


# ---------------- HALLUCINATION DETECTION ----------------

def detect_hallucination(answer, retrieved_docs):

    # Build context from retrieved docs
    context_text = ""

    for doc in retrieved_docs:
        context_text += " ".join([
            str(doc.get("year", "")),
            str(doc.get("department", "")),
            str(doc.get("company", "")),
            str(doc.get("score", "")),
        ]) + " "

    context_text = context_text.lower()
    answer_text = answer.lower()

    answer_numbers = extract_numbers(answer_text)

    for num in answer_numbers:
        if num not in context_text:
            return True  # hallucinated number

    return False


# ---------------- GROUNDING CHECK ----------------

def is_grounded(answer, retrieved_docs):
    return not detect_hallucination(answer, retrieved_docs)


# ---------------- RETRIEVAL PRECISION ----------------

def retrieval_precision(retrieved_docs, ground_truth):

    relevant = 0
    gt_text = normalize_text(ground_truth)

    for doc in retrieved_docs:
        doc_text = normalize_text(str(doc))
        if gt_text in doc_text:
            relevant += 1

    if len(retrieved_docs) == 0:
        return 0

    return relevant / len(retrieved_docs)


# ---------------- MAIN EVALUATION ----------------

def evaluate(test_questions):

    rag_em = 0
    rag_f1_total = 0
    rag_grounded = 0
    rag_hallucinated = 0
    rag_retrieval_precision_total = 0

    zero_em = 0
    zero_f1_total = 0
    zero_hallucinated = 0

    few_em = 0
    few_f1_total = 0
    few_hallucinated = 0

    total = len(test_questions)

    for question, ground_truth in test_questions:

        print(f"\nEvaluating Question: {question}")

        # ---------- RAG ----------
        rag_result = answer_with_rag(question, faiss_index, docs)
        rag_answer = rag_result["answer"]
        rag_sources = rag_result.get("sources", [])
        time.sleep(12)

        print("RAG Answer:", rag_answer)

        if exact_match(rag_answer, ground_truth):
            rag_em += 1

        rag_f1_total += compute_f1(rag_answer, ground_truth)

        if is_grounded(rag_answer, rag_sources):
            rag_grounded += 1
        else:
            rag_hallucinated += 1

        rag_retrieval_precision_total += retrieval_precision(rag_sources, ground_truth)


        # ---------- FEW SHOT ----------
        few_result = answer_with_few_shot(question)
        few_answer = few_result["answer"]
        time.sleep(12)

        print("Few-Shot Answer:", few_answer)

        if exact_match(few_answer, ground_truth):
            few_em += 1

        few_f1_total += compute_f1(few_answer, ground_truth)

        # Few-shot hallucination:
        # If it gives numbers but no database access → hallucination
        if extract_numbers(few_answer):
            few_hallucinated += 1

        # ---------- ZERO SHOT ----------
        zero_result = answer_with_zero_shot(question)
        zero_answer = zero_result["answer"]
        time.sleep(12)

        print("Zero-Shot Answer:", zero_answer)

        if exact_match(zero_answer, ground_truth):
            zero_em += 1

        zero_f1_total += compute_f1(zero_answer, ground_truth)

        # hallucination check
        if extract_numbers(zero_answer):
            zero_hallucinated += 1


    # ---------------- RESULTS ----------------

    print("\n================ FINAL RESULTS ================\n")

    print("RAG Evaluation:")
    print("Exact Match %:", round((rag_em / total) * 100, 2))
    print("Average F1:", round(rag_f1_total / total, 3))
    print("Grounding Rate %:", round((rag_grounded / total) * 100, 2))
    print("Hallucination Rate %:", round((rag_hallucinated / total) * 100, 2))
    print("Average Retrieval Precision:", round(rag_retrieval_precision_total / total, 3))

    print("\nFew-Shot Evaluation:")
    print("Exact Match %:", round((few_em / total) * 100, 2))
    print("Average F1:", round(few_f1_total / total, 3))
    print("Hallucination Rate %:", round((few_hallucinated / total) * 100, 2))

    print("\nZero-Shot Evaluation:")
    print("Exact Match %:", round((zero_em / total) * 100, 2))
    print("Average F1:", round(zero_f1_total / total, 3))
    print("Hallucination Rate %:", round((zero_hallucinated / total) * 100, 2))

# ---------------- TEST DATA ----------------

if __name__ == "__main__":

    test_data = [
        ("How many companies came in 2025?", "2"),
        ("How many people got TCS in 2025?", "1"),
        ("Which company came in 2024?", "TCS"),
        ("Which department had placements in 2025?", "CSE"),
        ("Compare placements between 2024 and 2025.", "2025 had more placements"),
    ]

    evaluate(test_data)
