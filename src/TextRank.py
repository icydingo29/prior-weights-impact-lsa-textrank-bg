from src import CoNLLReader
import numpy as np

# Power Iteration Constants
DAMPING = 0.85  # Damping factor
TOL = 1e-6 # Precision Threshold
MAX_ITER = 50 

def build_transition_matrix(edited_dicts, priors):
    n = len(edited_dicts)
    sim_matrix = np.zeros((n, n))
    norms = np.array([np.sqrt(sum(v**2 for v in d.values())) for d in edited_dicts])
    for i in range(n):
        for j in range(i + 1, n):
            common = set(edited_dicts[i].keys()) & set(edited_dicts[j].keys())
            if not common: continue

            num = sum(edited_dicts[i][w] * edited_dicts[j][w] for w in common)
            den = norms[i] * norms[j]
            score = num / den if den > 0 else 0
            sim_matrix[i][j] = sim_matrix[j][i] = score

    sum_rows = np.sum(sim_matrix, axis=1)
    P = np.zeros((n, n))
    for i in range(n):
        if sum_rows[i] > 0:
            P[i] = sim_matrix[i] / sum_rows[i]
        else:
            P[i] = priors
    return P

def power_iteration(P, priors):
    w = priors.copy()
    for _ in range(MAX_ITER):
        w_new = (1 - DAMPING) * priors + DAMPING * (P.T @ w)
        if np.linalg.norm(w_new - w, 1) < TOL:
            break
        w = w_new
    return w

def summarize_tr(original_sentences, edited_dicts, pos_dicts, ann_dicts):
    if not len(original_sentences) == len(edited_dicts) == len(pos_dicts) == len(ann_dicts):
        raise ValueError('Mismatch in sentence data!')

    n = len(original_sentences)
    if n == 0: return []

    priors = CoNLLReader.calculate_sentence_priors(original_sentences, pos_dicts, ann_dicts)

    P = build_transition_matrix(edited_dicts, priors)
    
    w = power_iteration(P, priors)

    # get indexes of the sentences for the summary
    ranked_indices = np.argsort(w)[::-1]

    return [(idx, w[idx]) for idx in ranked_indices]

if __name__ == "__main__":
    pass
