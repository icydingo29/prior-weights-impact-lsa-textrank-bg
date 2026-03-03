from src import CoNLLReader
import numpy as np

def build_tfidf_weighted_matrix(edited_dicts, priors):
    all_terms = sorted(list(set(word for d in edited_dicts for word in d.keys())))
    term_to_idx = {word: i for i, word in enumerate(all_terms)}
    num_terms, num_sentences = len(all_terms), len(edited_dicts)
    A = np.zeros((num_terms, num_sentences))

    # IDF calculation
    doc_counts = np.zeros(num_terms)
    for d in edited_dicts:
        for word in d.keys():
            doc_counts[term_to_idx[word]] += 1

    # TF-IDF calculation
    for j, d in enumerate(edited_dicts):
        for word, count in d.items():
            i = term_to_idx[word]
            idf = np.log(num_sentences / (doc_counts[i] + 1)) + 1
            A[i, j] = count * idf
    
    return A * priors, all_terms

def summarize_lsa(orig_sents, edited_dicts, pos_dicts, ann_dicts):
    n = len(orig_sents)
    if n == 0: return []

    priors = CoNLLReader.calculate_sentence_priors(orig_sents, pos_dicts, ann_dicts)
    A_weighted, _ = build_tfidf_weighted_matrix(edited_dicts, priors)
    
    # SVD
    U, S, Vt = np.linalg.svd(A_weighted, full_matrices=False)
    
    scores = np.square(Vt[0, :])
    ranked_indices = np.argsort(scores)[::-1]
    return [(idx, scores[idx]) for idx in ranked_indices]