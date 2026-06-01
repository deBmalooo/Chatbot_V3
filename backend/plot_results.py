import matplotlib.pyplot as plt

# Replace these values with your evaluation results

models = ["RAG", "Few-Shot", "Zero-Shot"]

exact_match = [0.0, 0.0, 0.0]
f1_score = [0.02, 0.0, 0.0]

x = range(len(models))

plt.figure(figsize=(8,5))

plt.plot(x, exact_match, marker='o', label="Exact Match")
plt.plot(x, f1_score, marker='o', label="F1 Score")

plt.xticks(x, models)

plt.xlabel("Model Type")
plt.ylabel("Score")
plt.title("RAG vs Few-Shot vs Zero-Shot Performance")

plt.legend()
plt.grid(True)

plt.savefig("evaluation_graph.png")
print("Graph saved as evaluation_graph.png")
