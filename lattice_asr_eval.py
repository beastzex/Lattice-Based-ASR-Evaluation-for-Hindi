# ============================================================
# Lattice-Based ASR Evaluation for Hindi — Task 4
# ============================================================
import re
import unicodedata
import pandas as pd
import numpy as np
from itertools import zip_longest

# ── 1. CONFIG ────────────────────────────────────────────────
MODELS = ["Model H", "Model i", "Model k", "Model l", "Model m", "Model n"]
REFERENCE_COL = "Human"
VOTE_THRESHOLD = 0.50   # majority: >50 % of models must agree to override ref

# ── 2. TEXT NORMALISATION ────────────────────────────────────
# Punctuation / diacritics that carry no lexical meaning in Hindi ASR
_PUNCT = re.compile(r'[।॥,.!?;:\-–—"\'""''()\[\]{}।|/\\@#%^&*+=<>~`]')
# Transliteration map: common Latin loanwords found in the data
_LATIN_MAP = {
    "feedback": "फीडबैक", "pure": "प्योर", "heart": "हार्ट",
    "sir": "सर", "face": "फेस", "desktop": "डेस्कटॉप",
    "laptop": "लैपटॉप", "easy": "इजी",
}

def normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, map Latin tokens → Devanagari, return word list."""
    if not isinstance(text, str):
        return []
    text = unicodedata.normalize("NFC", text)
    text = _PUNCT.sub(" ", text)
    tokens = text.strip().split()
    out = []
    for t in tokens:
        t_lower = t.lower()
        out.append(_LATIN_MAP.get(t_lower, t))
    return [w for w in out if w]


# ── 3. WORD-LEVEL ALIGNMENT (Needleman-Wunsch / edit-path) ──
def _edit_ops(ref: list, hyp: list):
    """Return list of (op, ref_idx_or_None, hyp_idx_or_None) via DP back-trace."""
    R, H = len(ref), len(hyp)
    dp = np.zeros((R + 1, H + 1), dtype=int)
    for i in range(R + 1): dp[i][0] = i
    for j in range(H + 1): dp[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    # back-trace
    ops, i, j = [], R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            ops.append(("match", i-1, j-1)); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(("sub", i-1, j-1)); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(("del", i-1, None)); i -= 1
        else:
            ops.append(("ins", None, j-1)); j -= 1
    return list(reversed(ops))


# ── 4. LATTICE CONSTRUCTION ──────────────────────────────────
def build_lattice(ref_words: list[str], model_outputs: dict[str, list[str]],
                  vote_threshold: float = VOTE_THRESHOLD) -> list[set]:
    """
    Align every model output against the reference using word-level edit ops.
    For each reference position, collect all model words (substitutions/matches).
    Apply majority-vote to decide whether to trust the reference or model consensus.

    Returns a list of sets — each set is one lattice bin.
    """
    n_models = len(model_outputs)
    # position → {word: count}
    pos_votes: dict[int, dict[str, int]] = {i: {} for i in range(len(ref_words))}
    # positions where models insert a word (no ref counterpart)
    ins_votes: dict[int, dict[str, int]] = {}   # keyed by (ref_pos_before, order)

    for model_name, hyp_words in model_outputs.items():
        ops = _edit_ops(ref_words, hyp_words)
        ins_counter = {}   # track insertions per ref position
        for op, ri, hi in ops:
            if op in ("match", "sub"):
                w = hyp_words[hi]
                pos_votes[ri][w] = pos_votes[ri].get(w, 0) + 1
            elif op == "del":
                # model omitted this ref word — count ref word itself
                w = ref_words[ri]
                pos_votes[ri][w] = pos_votes[ri].get(w, 0) + 0  # don't add
            elif op == "ins":
                # insertion before next ref position
                anchor = ri if ri is not None else -1
                key = (anchor, ins_counter.get(anchor, 0))
                ins_counter[anchor] = ins_counter.get(anchor, 0) + 1
                if key not in ins_votes:
                    ins_votes[key] = {}
                w = hyp_words[hi]
                ins_votes[key][w] = ins_votes[key].get(w, 0) + 1

    lattice: list[set] = []

    for ri, ref_word in enumerate(ref_words):
        votes = pos_votes[ri]
        total_votes = sum(votes.values())
        bin_words: set[str] = set()

        # Check if models agree on something different from reference
        top_word, top_count = max(votes.items(), key=lambda x: x[1]) if votes else (ref_word, 0)
        model_agreement = top_count / n_models

        if model_agreement >= vote_threshold and top_word != ref_word:
            # Majority disagrees with reference → trust models, add both
            bin_words.add(top_word)
            # Still add ref if at least one model matched it
            if ref_word in votes and votes[ref_word] > 0:
                bin_words.add(ref_word)
        else:
            # Reference is trusted; add all words that appeared at this position
            bin_words.add(ref_word)
            for w, cnt in votes.items():
                if cnt >= 2:   # at least 2 models agree → valid variant
                    bin_words.add(w)

        lattice.append(bin_words)

    # Handle majority insertions (words most models add that ref lacks)
    # We insert them as extra bins only if >50% of models produced them
    ins_bins: list[tuple] = []
    for (anchor, order), votes in ins_votes.items():
        top_w, top_c = max(votes.items(), key=lambda x: x[1])
        if top_c / n_models >= vote_threshold:
            ins_bins.append((anchor, order, top_w))

    # Rebuild lattice with insertion bins spliced in
    if ins_bins:
        ins_bins.sort(key=lambda x: (x[0], x[1]))
        new_lattice: list[set] = []
        for ri, bin_set in enumerate(lattice):
            # insertions before this ref position
            for (anchor, order, w) in ins_bins:
                if anchor == ri - 1:
                    new_lattice.append({w})
            new_lattice.append(bin_set)
        lattice = new_lattice

    return lattice


# ── 5. LATTICE WER ───────────────────────────────────────────
def lattice_wer(lattice: list[set], hyp_words: list[str]) -> float:
    """
    Adapted Levenshtein distance between a hypothesis word sequence and a lattice.
    A substitution cost is 0 if the hyp word is IN the lattice bin, else 1.
    """
    L, H = len(lattice), len(hyp_words)
    dp = np.zeros((L + 1, H + 1), dtype=float)
    for i in range(L + 1): dp[i][0] = i
    for j in range(H + 1): dp[0][j] = j

    for i in range(1, L + 1):
        for j in range(1, H + 1):
            hit = hyp_words[j-1] in lattice[i-1]
            sub_cost = 0.0 if hit else 1.0
            dp[i][j] = min(
                dp[i-1][j-1] + sub_cost,   # match / sub
                dp[i-1][j]   + 1.0,         # deletion from lattice
                dp[i][j-1]   + 1.0,         # insertion in hyp
            )
    ref_len = L
    if ref_len == 0:
        return 0.0
    return float(dp[L][H]) / ref_len


# ── 6. STANDARD WER (for comparison) ────────────────────────
def standard_wer(ref_words: list[str], hyp_words: list[str]) -> float:
    R, H = len(ref_words), len(hyp_words)
    if R == 0:
        return 0.0
    dp = np.zeros((R + 1, H + 1), dtype=int)
    for i in range(R + 1): dp[i][0] = i
    for j in range(H + 1): dp[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            dp[i][j] = min(dp[i-1][j-1] + cost, dp[i-1][j] + 1, dp[i][j-1] + 1)
    return dp[R][H] / R


# ── 7. MAIN PIPELINE ─────────────────────────────────────────
def main():
    df = pd.read_excel("Question 4.xlsx", sheet_name="Task")
    df = df.dropna(subset=[REFERENCE_COL])

    results = {m: {"std_wer": [], "lat_wer": []} for m in MODELS}

    for _, row in df.iterrows():
        ref_words = normalise(str(row[REFERENCE_COL]))
        if not ref_words:
            continue

        model_norm = {m: normalise(str(row[m])) for m in MODELS if isinstance(row.get(m), str)}

        lattice = build_lattice(ref_words, model_norm)

        for m in MODELS:
            if m not in model_norm:
                continue
            hyp = model_norm[m]
            results[m]["std_wer"].append(standard_wer(ref_words, hyp))
            results[m]["lat_wer"].append(lattice_wer(lattice, hyp))

    # ── 8. SUMMARY TABLE ────────────────────────────────────
    rows = []
    for m in MODELS:
        sw = np.mean(results[m]["std_wer"]) * 100
        lw = np.mean(results[m]["lat_wer"]) * 100
        delta = sw - lw
        rows.append({
            "Model": m,
            "Standard WER (%)": round(sw, 2),
            "Lattice WER (%)":  round(lw, 2),
            "WER Reduction (pp)": round(delta, 2),
            "Verdict": "Unfairly penalised - corrected" if delta > 0.5 else
                       ("Slightly improved" if delta > 0 else "Unchanged / worse")
        })

    summary = pd.DataFrame(rows)
    print("\n" + "="*72)
    print("  LATTICE-BASED ASR EVALUATION — FINAL RESULTS")
    print("="*72)
    print(summary.to_string(index=False))
    print("="*72)

    # ── 9. SAMPLE LATTICE (first row) — written to file ────
    first_row = df.iloc[0]
    ref_w = normalise(str(first_row[REFERENCE_COL]))
    m_norm = {m: normalise(str(first_row[m])) for m in MODELS if isinstance(first_row.get(m), str)}
    lat = build_lattice(ref_w, m_norm)
    with open("sample_lattice.txt", "w", encoding="utf-8") as f:
        f.write("Sample Lattice (row 1):\n")
        f.write(f"  Reference : {ref_w}\n")
        for i, b in enumerate(lat):
            f.write(f"  Bin [{i:02d}]  : {sorted(b)}\n")
    print("\nSample lattice written to sample_lattice.txt")

    # Save results
    summary.to_csv("lattice_wer_results.csv", index=False, encoding="utf-8-sig")
    print("\nResults saved to lattice_wer_results.csv")


if __name__ == "__main__":
    main()
