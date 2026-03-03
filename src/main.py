from src.TextRank import summarize_tr
from src.LSA import summarize_lsa
from src.CoNLLReader import process_conll_file
import pathlib


def save_summary(path, sentences):
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(sentences).strip())


def compare_summaries(file_path, top_n = 10, tr_out_path = "evaluation/tr.txt",
                      lsa_out_path= "evaluation/lsa.txt", verbose = True):
    
    data = process_conll_file(file_path)
    orig_sents, edited_dicts, pos_dicts, ann_dicts = data

    tr_results = summarize_tr(orig_sents, edited_dicts, pos_dicts, ann_dicts)
    lsa_results = summarize_lsa(orig_sents, edited_dicts, pos_dicts, ann_dicts)
    
    tr_top_indices = ([idx for idx, _ in tr_results[:top_n]])
    lsa_top_indices = ([idx for idx, _ in lsa_results[:top_n]])
    
    if verbose:
        print(f"=== ANALYSIS FOR: {file_path} ===")
        
        print("\n[TEXTRANK SUMMARY]:")
        for idx in tr_top_indices:
            print(f"- {orig_sents[idx]}")
            
        print("\n[LSA SUMMARY]:")
        for idx in lsa_top_indices:
            print(f"- {orig_sents[idx]}")

    tr_summary = [orig_sents[idx] for idx in tr_top_indices]
    lsa_summary = [orig_sents[idx] for idx in lsa_top_indices]

    pathlib.Path(tr_out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(lsa_out_path).parent.mkdir(parents=True, exist_ok=True)

    save_summary(tr_out_path, tr_summary)
    save_summary(lsa_out_path, lsa_summary)

    set_tr = set(tr_top_indices)
    set_lsa = set(lsa_top_indices)
    overlap = set_tr.intersection(set_lsa)
    
    if verbose:
        print(f"\nOverlap: {len(overlap)}/{top_n} sentences match.")

    return tr_top_indices, lsa_top_indices

if __name__ == "__main__":
    # Example usage: provide path to a CoNLL file
    # compare_summaries("data/raw/CoNLL/your_file.conll")
    compare_summaries("data/raw/CoNLL/USElection2020-00756621.conll")
    pass