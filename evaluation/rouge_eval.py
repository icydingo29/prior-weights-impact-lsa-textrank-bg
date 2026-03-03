import re
from collections import Counter

# NOTE: This file can be used both as a standalone script and as an importable module.
# The main entry-point for the project code is `run_rouge_eval(...)`.

def tokenize(text):
    # Simple tokenization
    return re.findall(r"[0-9A-Za-zА-Яа-яЁёІіЇїЪъЬь]+", text.lower())

def ngrams(tokens, n):
    # ngrams of words(not characters)
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def rouge_n(candidate, reference, n):
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)

    cand_ngrams = Counter(ngrams(cand_tokens, n))
    ref_ngrams = Counter(ngrams(ref_tokens, n))

    overlap = sum((cand_ngrams & ref_ngrams).values())
    cand_total = sum(cand_ngrams.values())
    ref_total = sum(ref_ngrams.values())

    # Precision: how much of the system-generated summary is relevant
    # (fraction of candidate n-grams that also appear in the reference summary)
    precision = overlap / cand_total if cand_total > 0 else 0.0

    # Recall: how much of the reference summary is covered by the system
    # (fraction of reference n-grams that are captured by the candidate summary)
    recall = overlap / ref_total if ref_total > 0 else 0.0

    # F1-score: harmonic mean of precision and recall
    # (balances coverage and exactness of the summary)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

# Longest Common Subsequence
def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    return dp[n]

def rouge_l(candidate, reference):
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)

    lcs = lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def print_scores(label, scores):
    print(f"\n== {label} ==")
    for k, v in scores.items():
        print(f"{k:>9}: {v:.4f}")


def run_rouge_eval(gold_path, tr_path, lsa_path, verbose= False):
    gold = load_text(gold_path)
    tr = load_text(tr_path)
    lsa = load_text(lsa_path)

    results = {
        "tr": {
            "rouge-1": rouge_n(tr, gold, 1),
            "rouge-2": rouge_n(tr, gold, 2),
            "rouge-L": rouge_l(tr, gold),
        },
        "lsa": {
            "rouge-1": rouge_n(lsa, gold, 1),
            "rouge-2": rouge_n(lsa, gold, 2),
            "rouge-L": rouge_l(lsa, gold),
        },
    }

    if verbose:
        print(f"\n[ROUGE] gold={gold_path} | tr={tr_path} | lsa={lsa_path}")
        print("\n--- TextRank vs Gold ---")
        print_scores("ROUGE-1", results["tr"]["rouge-1"])
        print_scores("ROUGE-2", results["tr"]["rouge-2"])
        print_scores("ROUGE-L", results["tr"]["rouge-L"])

        print("\n--- LSA vs Gold ---")
        print_scores("ROUGE-1", results["lsa"]["rouge-1"])
        print_scores("ROUGE-2", results["lsa"]["rouge-2"])
        print_scores("ROUGE-L", results["lsa"]["rouge-L"])

    return results

if __name__ == "__main__":
    run_rouge_eval(
        gold_path="evaluation/chatgpt/Brexit-00224495_10.txt",
        tr_path="evaluation/tr.txt",
        lsa_path="evaluation/lsa.txt",
        verbose=True,
    )
