
from src import CoNLLReader
import pathlib
import matplotlib
import matplotlib.pyplot as plt
from evaluation.rouge_eval import run_rouge_eval
from src.main import compare_summaries

matplotlib.use("Agg")

TOP_N = 10
SUPPORTED_TOP_NS = (5, 10, 15)
ALPHA_GRID = (0.0, 0.5, 1.0)

def set_alpha(alpha):
    CoNLLReader.ALPHA = float(alpha)
    CoNLLReader.BETA = 1.0 - float(alpha)


def project_root():
    return pathlib.Path(__file__).resolve().parent.parent


def discover_conll_files():
    root = project_root()
    conll_dir = root / "data" / "raw" / "CoNLL"
    files = sorted(conll_dir.glob("*.conll"))
    return files


def gold_path_for(conll_path, top_n):
    root = project_root()
    base = conll_path.stem
    return root / "evaluation" / "chatgpt" / f"{base}_{top_n}.txt"


def system_out_paths(conll_path, alpha, top_n):
    root = project_root()
    base = conll_path.stem
    safe_alpha = str(alpha).replace(".", "p")
    out_dir = root / "evaluation" / "system" / base / f"alpha_{safe_alpha}"
    tr_path = out_dir / f"tr_{top_n}.txt"
    lsa_path = out_dir / f"lsa_{top_n}.txt"
    return tr_path, lsa_path


def store_f1(results, alpha, top_n, text_id, rouge_out):
    results.setdefault(alpha, {}).setdefault(top_n, {}).setdefault(text_id, {}).setdefault("tr", {})
    results.setdefault(alpha, {}).setdefault(top_n, {}).setdefault(text_id, {}).setdefault("lsa", {})

    for method in ("tr", "lsa"):
        results[alpha][top_n][text_id][method]["rouge-1"] = float(rouge_out[method]["rouge-1"]["f1"])
        results[alpha][top_n][text_id][method]["rouge-2"] = float(rouge_out[method]["rouge-2"]["f1"])
        results[alpha][top_n][text_id][method]["rouge-L"] = float(rouge_out[method]["rouge-L"]["f1"])


def mean(values):
    return sum(values) / len(values) if values else 0.0


def plot_curves(results, conll_files):
    root = project_root()
    plots_dir = root / "plots" / "rouge_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)

    text_ids = [p.stem for p in conll_files]
    x = list(SUPPORTED_TOP_NS)
    metrics = ["rouge-1", "rouge-2", "rouge-L"]

    default_alpha = getattr(CoNLLReader, "ALPHA", 0.8)
    for metric in metrics:
        tr_means = []
        lsa_means = []
        for n in x:
            tr_vals = [results[default_alpha][n][tid]["tr"][metric] for tid in text_ids]
            lsa_vals = [results[default_alpha][n][tid]["lsa"][metric] for tid in text_ids]
            tr_means.append(mean(tr_vals))
            lsa_means.append(mean(lsa_vals))

        plt.figure()
        plt.plot(x, tr_means, marker="o", label="TextRank")
        plt.plot(x, lsa_means, marker="o", label="LSA")
        plt.xlabel("top_n (sentences)")
        plt.ylabel("F1")
        plt.title(f"F1 vs top_n ({metric.upper()}) — TextRank vs LSA (alpha={default_alpha})")
        plt.xticks(x)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"type1_{metric}_tr_vs_lsa_alpha_{str(default_alpha).replace('.', 'p')}.png", dpi=200)
        plt.close()

    for metric in metrics:
        plt.figure()
        for alpha in ALPHA_GRID:
            y = []
            for n in x:
                vals = [results[alpha][n][tid]["tr"][metric] for tid in text_ids]
                y.append(mean(vals))
            plt.plot(x, y, marker="o", label=f"alpha={alpha}")
        plt.xlabel("top_n (sentences)")
        plt.ylabel("F1")
        plt.title(f"F1 vs top_n ({metric.upper()}) — TextRank (alpha sweep)")
        plt.xticks(x)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"type2_{metric}_textrank_alpha_sweep.png", dpi=200)
        plt.close()

    for metric in metrics:
        plt.figure()
        for alpha in ALPHA_GRID:
            y = []
            for n in x:
                vals = [results[alpha][n][tid]["lsa"][metric] for tid in text_ids]
                y.append(mean(vals))
            plt.plot(x, y, marker="o", label=f"alpha={alpha}")
        plt.xlabel("top_n (sentences)")
        plt.ylabel("F1")
        plt.title(f"F1 vs top_n ({metric.upper()}) — LSA (alpha sweep)")
        plt.xticks(x)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"type3_{metric}_lsa_alpha_sweep.png", dpi=200)
        plt.close()


def run():
    root = project_root()

    conll_files = discover_conll_files()
    if len(conll_files) == 0:
        raise FileNotFoundError("No .conll files found in data/raw/CoNLL")

    default_alpha = getattr(CoNLLReader, "ALPHA", 0.8)

    all_alphas = tuple(dict.fromkeys((default_alpha, *ALPHA_GRID)))

    results = {}

    for conll_path in conll_files:
        text_id = conll_path.stem
        for alpha in all_alphas:
            set_alpha(alpha)
            for n in SUPPORTED_TOP_NS:
                gold_path = gold_path_for(conll_path, n)
                if not gold_path.exists():
                    raise FileNotFoundError(
                        f"Missing gold summary file: {gold_path}. "
                        f"Expected naming: evaluation/chatgpt/{text_id}_{n}.txt"
                    )

                tr_out, lsa_out = system_out_paths(conll_path, alpha, n)

                compare_summaries(
                    str(conll_path),
                    top_n=n,
                    tr_out_path=str(tr_out),
                    lsa_out_path=str(lsa_out),
                    verbose=False,
                )

                rouge_out = run_rouge_eval(
                    gold_path=str(gold_path),
                    tr_path=str(tr_out),
                    lsa_path=str(lsa_out),
                    verbose=False,
                )

                store_f1(results, alpha=alpha, top_n=n, text_id=text_id, rouge_out=rouge_out)

        print(f"[DONE] {text_id}")

    set_alpha(default_alpha)

    plot_curves(results, conll_files)
    print(f"\nSaved plots to: {(root / 'plots' / 'rouge_curves').as_posix()}")

    return results


if __name__ == "__main__":
    run()
