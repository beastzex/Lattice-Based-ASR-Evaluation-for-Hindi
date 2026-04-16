# Lattice-Based ASR Evaluation for Hindi

A comprehensive framework for evaluating Automatic Speech Recognition (ASR) models using lattice-based word error rate (WER) calculation, specifically designed to address the limitations of standard WER in Hindi ASR evaluation.

## Overview

This project implements an advanced ASR evaluation system that goes beyond traditional word error rate metrics by:

- **Handling multiple valid transcriptions** — Accounts for orthographic variants, code-switching, and compound splitting that are common in Hindi
- **Correcting reference errors** — Uses a voting mechanism across multiple ASR models to identify and correct errors in the human reference transcription
- **Word-level alignment** — Employs word-level edit distance alignment to maintain interpretability and handle Hindi-specific linguistic phenomena

## Problem Statement

Standard ASR evaluation compares a model's output against a single ground truth string, treating every deviation as an error. This approach fails for Hindi ASR due to:

- **Orthographic freedom** — The same word can be written multiple ways (e.g., `हाँ` / `हां`)
- **Code-switching** — Hindi speakers mix English loanwords in both scripts (e.g., `फीडबैक` / `feedback`)
- **Compound splitting** — Words may be written as one or two units (e.g., `रक्षाबंधन` / `रक्षा बंधन`)
- **Human reference errors** — The annotator may produce a reference less accurate than some model outputs

## Solution: Lattice-Based Evaluation

A **lattice** replaces the flat reference string with a sequential list of **bins**, where each bin contains all valid lexical and spelling alternatives for that position. Models are penalized only if their output word doesn't appear in the corresponding bin.

### Key Features

1. **Text Normalization** — Removes evaluation-irrelevant surface differences (punctuation, diacritics, script variants)
2. **Voting Mechanism** — If ≥50% of models agree on a word and it differs from the reference, the model consensus is trusted
3. **Flexible Alignment** — Handles insertions, deletions, substitutions, and matches at the word level
4. **Lattice WER** — Computes adapted Levenshtein distance where words are correct if they appear in the corresponding lattice bin

## Project Structure

```
├── lattice_asr_eval.py         # Main evaluation script
├── sample_lattice.txt          # Example lattice format
├── data_dump.txt               # Sample input data
├── lattice_wer_results.csv     # Output results
├── Task_4_Report.md            # Detailed technical report
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

## Usage

### Input Format

The script expects:
- **Human reference** — Ground truth transcription in Devanagari script or Latin transliteration
- **Model outputs** — Transcriptions from 5+ ASR models

### Configuration

Edit these parameters in `lattice_asr_eval.py`:

```python
MODELS = ["Model H", "Model i", "Model k", "Model l", "Model m", "Model n"]
REFERENCE_COL = "Human"
VOTE_THRESHOLD = 0.50  # Majority threshold for model consensus
```

### Running the Evaluation

```bash
python lattice_asr_eval.py
```

### Output

- **Lattice structure** — All valid word alternatives per position
- **WER metrics** — Lattice WER compared to traditional WER
- **CSV results** — Detailed per-segment evaluation scores

## Alignment Strategy

The system uses **word-level alignment** (not subword or phrase) because:

- ✓ High interpretability — WER is the industry standard
- ✓ Natural lattice semantics — One bin = one spoken word position
- ✓ Handles compound splits directly
- ✓ Linguistically appropriate for agglutinative Hindi morphology

## Voting Mechanism

**Rule:** If ≥50% of models agree on a word at a position and it differs from the reference:
- The model consensus word is added to the lattice bin
- The reference word may be overridden if confidence is high
- Only words with ≥2 model votes are retained

**Why 50%?** With 5 models, this requires ≥3 models to agree—a strict majority unlikely by chance while remaining practical.

## Results Interpretation

- **Lattice WER** — Lower than traditional WER due to accepting valid alternatives
- **Consensus positions** — Where ≥3 models agree, lattice WER often differs from single-reference WER
- **Reference corrections** — Identified positions where the human reference was likely incorrect

## Dependencies

- Python 3.7+
- pandas
- numpy

## Technical Details

For in-depth explanation of the theoretical framework, alignment algorithm, voting mechanism, and empirical results, see **Task_4_Report.md**.

## Dataset

- **Audio segments:** 40 Hindi speech samples
- **Models evaluated:** 5 ASR systems + 1 human reference
- **Language:** Hindi (Devanagari script)
- **Evaluation metric:** Word Error Rate (WER)

## Future Improvements

- Support for multiple languages beyond Hindi
- Customizable voting thresholds
- Phonetic-level alignment options
- Interactive visualization of lattices
- Integration with popular ASR frameworks

## License

Internal project for research purposes.

