<div align="center">

<!-- Animated Title Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=DesiTalks%20ASR%20System&fontSize=50&fontAlignY=35&desc=End-to-End%20Hindi%20Speech%20Recognition%20with%20Whisper&descAlignY=58&descAlign=50&animation=fadeIn&fontColor=ffffff" width="100%"/>

<br/>

<!-- Animated Badges -->
<a href="#"><img src="https://img.shields.io/badge/🎙️%20ASR-Whisper%20Small-4A90D9?style=for-the-badge&logoColor=white"/></a>
<a href="#"><img src="https://img.shields.io/badge/🇮🇳%20Language-Hindi%20%7C%20Devanagari-FF6B35?style=for-the-badge"/></a>
<a href="#"><img src="https://img.shields.io/badge/🧠%20Framework-HuggingFace-FFD43B?style=for-the-badge&logo=huggingface&logoColor=black"/></a>
<a href="#"><img src="https://img.shields.io/badge/📉%20WER%20Reduction-230%%20→%2041%%20-00C853?style=for-the-badge"/></a>
<a href="#"><img src="https://img.shields.io/badge/🔬%20Evaluation-Lattice%20Based-9C27B0?style=for-the-badge"/></a>
<a href="#"><img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black"/></a>

<br/><br/>

> **🏢 Organization:** Josh Talks — AI Researcher Intern Assignment  
> **👨‍💻 Author:** Jayant Yadav  
> **📅 Domain:** Speech & Audio | Automatic Speech Recognition | NLP

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/jayant-yadav-a22b98283/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/CoderJaynt)

</div>

---

## 📋 Table of Contents

- [✨ Project Overview](#-project-overview)
- [🏗️ System Architecture](#️-system-architecture)
- [📊 Q1 — ASR Pipeline & Fine-Tuning](#-q1--asr-pipeline--fine-tuning)
- [🧹 Q2 — Post-Processing & NLP Pipeline](#-q2--post-processing--nlp-pipeline)
- [🧠 Q3 — Word-Level Classification](#-q3--word-level-classification)
- [🔬 Q4 — Lattice-Based Evaluation](#-q4--lattice-based-evaluation)
- [📈 Results Summary](#-results-summary)
- [⚙️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [📁 Repository Structure](#-repository-structure)

---

## ✨ Project Overview

This repository presents a **complete, production-grade Automatic Speech Recognition (ASR) pipeline** built for Hindi conversational speech, developed as part of the Josh Talks AI Researcher Intern technical assignment.

The system goes far beyond basic transcription:

| Component | What it does |
|-----------|-------------|
| 🎯 **Fine-Tuning** | Adapts Whisper-small to domain-specific Hindi audio, reducing WER from 230% → ~41% |
| 🧹 **Post-Processing** | Intelligent cleanup pipeline for numbers, English loanwords, and repetitions |
| 🧠 **Word Classification** | Automated correct/incorrect spelling detection across ~1.77 lakh unique words |
| 🔬 **Lattice Evaluation** | Fair, multi-reference WER computation that doesn't unfairly penalize valid transcriptions |

### 🎯 Problem Statement

Traditional ASR pipelines suffer from three key weaknesses:

1. **Rigid evaluation** — Single ground-truth strings unfairly penalize valid alternate spellings/pronunciations
2. **Code-switching blindness** — Hindi-English mixed speech causes systematic errors
3. **No interpretability** — Opaque errors with no actionable categorization

**This project solves all three.**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DesiTalks ASR Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Raw Audio (GCP)                                                   │
│        │                                                            │
│        ▼                                                            │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐      │
│   │  Audio   │───▶│   Whisper    │───▶│   Raw ASR Output    │      │
│   │  Ingest  │    │  Fine-Tuned  │    │  (Hindi Devanagari) │      │
│   └──────────┘    └──────────────┘    └─────────────────────┘      │
│        │                                        │                   │
│        │ Q1                                     │ Q2                │
│        ▼                                        ▼                   │
│   ┌──────────────────┐              ┌───────────────────────┐      │
│   │  Preprocessing   │              │   Cleanup Pipeline    │      │
│   │  • URL fixing    │              │  • Number normalize   │      │
│   │  • 16kHz resamp  │              │  • English tagging    │      │
│   │  • Text norm     │              │  • Repetition removal │      │
│   │  • Filtering     │              └───────────────────────┘      │
│   └──────────────────┘                          │                   │
│                                                 │ Q3                │
│                                                 ▼                   │
│                                    ┌───────────────────────┐       │
│                                    │  Word Classification  │       │
│                                    │  • Rule-based system  │       │
│                                    │  • Confidence scoring │       │
│                                    │  • 1.77L words        │       │
│                                    └───────────────────────┘       │
│                                                 │                   │
│                                                 │ Q4                │
│                                                 ▼                   │
│                                    ┌───────────────────────┐       │
│                                    │  Lattice Evaluation   │       │
│                                    │  • Multi-ref WER      │       │
│                                    │  • Fair scoring       │       │
│                                    └───────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Q1 — ASR Pipeline & Fine-Tuning

**Notebook:** `Whisper_Model_training___evaluation.ipynb`

### Data Preprocessing

The ~10-hour Hindi dataset was ingested from Google Cloud Storage. URLs were systematically patched from the legacy format to the active `upload_goai` bucket format before any data fetching.

**Preprocessing steps applied:**

1. **Transcription Fetching** — JSON transcriptions fetched per recording, expanded to segment-level records
2. **Duration Filtering** — Segments outside 1–30 seconds were removed (noise, silence, very long utterances)
3. **Empty/Trivial Removal** — Transcriptions with ≤1 character discarded
4. **Unicode Normalization** — NFC normalization applied for consistent Devanagari representation
5. **Whitespace Cleanup** — Multiple spaces collapsed; leading/trailing whitespace stripped
6. **Audio Processing** — Downloaded to cache, sliced using timestamps, resampled to 16kHz mono WAV
7. **Speaker-Aware Split** — 10% of unique speakers held out for validation (stratified, `seed=42`)

```
Before filtering: N segments
After filtering:  N - invalid segments
Removed:          noise + empty + OOB duration segments
```

### Fine-Tuning Configuration

| Hyperparameter | Value |
|----------------|-------|
| Base Model | `openai/whisper-small` |
| Learning Rate | `1e-5` |
| Batch Size | `16` |
| Max Steps | `1,000` |
| Warmup Steps | `100` |
| Eval Strategy | Every 200 steps |
| Precision | FP16 |
| Best Metric | WER (lower is better) |

### WER Results

| Model | WER (%) | Notes |
|-------|---------|-------|
| 🔴 Whisper-small Baseline | **230.11%** | Pre-trained, no domain adaptation |
| 🟢 Fine-tuned Whisper-small | **~41%** | After domain fine-tuning on Hindi dataset |

> **~189 percentage point improvement** — demonstrating the critical importance of domain adaptation for Hindi ASR.

### Error Taxonomy

After systematically sampling 25+ error utterances (every Nth error strategy, non-cherry-picked):

| Error Category | Description | Example |
|---------------|-------------|---------|
| **Substitution** | Phonetically similar word substituted | ref: `किताब` → pred: `कताब` |
| **Deletion** | Short function words dropped | ref: `मैंने यह किया` → pred: `मैं किया` |
| **Insertion** | Hallucinated filler words added | pred adds `है` where not spoken |
| **Hallucination** | Model generates plausible-but-wrong content | Repeated phrases, unrelated words |
| **Code-Mix Errors** | English loanwords in Devanagari misread | `इंटरव्यू` confused for native words |

### Proposed & Implemented Fixes

| Error Type | Fix | Status |
|------------|-----|--------|
| Repetition/Hallucination | Post-processing repetition removal | ✅ Implemented |
| Number word mismatch | Number normalization pipeline | ✅ Implemented (Q2) |
| Code-mix handling | English word tagging + dictionary expansion | ✅ Implemented (Q2) |

---

## 🧹 Q2 — Post-Processing & NLP Pipeline

**Notebook:** `DesiTalks_Q2_Post_processing___NLP_pipeline.ipynb`

Raw ASR output from Hindi conversations is inherently messy. This pipeline cleans it systematically before any downstream NLP task.

### a) Number Normalization

Converts spoken Hindi number words into digit form.

```python
num_map = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, ...
    "सौ": 100, "हजार": 1000
}
```

**Compound number handling:**

| Input | Output |
|-------|--------|
| `दो` | `2` |
| `दस` | `10` |
| `तीन सौ चौवन` | `354` |
| `एक हज़ार` | `1000` |
| `पच्चीस` | `25` |

**Edge Case Handling:**

The pipeline uses a `is_safe_to_convert()` guard to skip idioms where literal conversion is wrong:

| Input | Decision | Reason |
|-------|----------|--------|
| `दो-चार बातें` | ❌ Skip (keep as-is) | Idiomatic expression — "a few things" |
| `दो किताबें` | ✅ Convert → `2 किताबें` | Literal count |

### b) English Word Detection & Tagging

Critical for downstream processing: English words spoken in Hindi conversation are transcribed in Devanagari per the guideline (e.g., "computer" → `कंप्यूटर`). These need separate handling.

**Detection rules applied:**
1. **Dictionary lookup** — Known English loanwords list
2. **Roman script detection** — Any token containing `[a-zA-Z]`
3. **Phonetic suffix matching** — Endings like `शन`, `मेंट`, `टिंग`

**Example:**

```
Input:  "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई"
Output: "मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई"
```

---

## 🧠 Q3 — Word-Level Classification

**Notebook:** `DesiTalks_Q3_Word_Classification.ipynb`

### Objective

Classify ~1,77,000 unique words from the human-transcribed Hindi dataset into:
- ✅ **Correctly spelled** — Ready to use as-is
- ❌ **Incorrectly spelled** — Flag for re-transcription

This avoids re-doing the entire dataset — only flagged segments are re-transcribed.

### Approach

A rule-based classification system combining multiple heuristics:

```
Word Input
    │
    ├─▶ [Hindi Vocabulary Dictionary] ──────────────▶ CORRECT (High confidence)
    │
    ├─▶ [English Loanword Dictionary] ──────────────▶ CORRECT (High confidence)
    │
    ├─▶ [Pattern Rules]
    │       • Repetitive characters (3+ same char) ─▶ INCORRECT
    │       • Excessive halant (्) usage           ─▶ INCORRECT
    │       • Single character tokens              ─▶ INCORRECT
    │
    └─▶ [Frequency Signal]
            • High frequency → likely correct
            • Low frequency → uncertain (Low confidence)
```

### Confidence Scoring

Every classified word receives a `high / medium / low` confidence label with a reason.

### Results

| Metric | Value |
|--------|-------|
| Words reviewed (low confidence bucket) | 40–50 |
| Correctly classified | **12** |
| Incorrectly classified | **13** |

### Known Failure Modes

| Word Category | Why It's Hard |
|---------------|---------------|
| **Rare proper nouns** | Not in dictionary, low frequency — looks like an error but isn't |
| **English loanwords in Devanagari** | `कंप्यूटर` valid per guidelines but pattern-matches as suspicious |
| **Dialectal variants** | Regional spellings of common words confuse the rule engine |

---

## 🔬 Q4 — Lattice-Based Evaluation

**Notebook:** `DesiTalks_Q4.ipynb`

### The Problem with Standard WER

Standard WER compares model output against a **single rigid reference string**. This unfairly penalizes:
- Valid number representations (`14` vs `चौदह`)
- Acceptable spelling variants (`किताबें` vs `किताबे`)
- Lexical synonyms (`पुस्तकें` vs `किताबें`)

### Lattice Solution

A **lattice** replaces the flat reference with a sequence of "bins." Each bin contains all valid alternatives at that alignment position.

**Example:**

```
Spoken audio: "उसने चौदह किताबें खरीदीं"

Standard ref:  ["उसने", "चौदह", "किताबें", "खरीदीं"]

Lattice:       [["उसने"],
                ["चौदह", "14"],
                ["किताबें", "किताबे", "पुस्तकें"],
                ["खरीदीं", "खरीदी"]]
```

### Implementation

```python
def build_lattice(models, reference):
    tokenized = [tokenize(m) for m in models]
    tokenized.append(tokenize(reference))
    aligned = align_sequences(tokenized)
    lattice = []
    for position in aligned:
        variants = set([w for w in position if w != "<pad>"])
        lattice.append(list(variants))
    return lattice

def compute_lattice_wer(pred, lattice):
    pred_tokens = tokenize(pred)
    errors = sum(
        1 for i in range(len(lattice))
        if i >= len(pred_tokens) or pred_tokens[i] not in lattice[i]
    )
    return errors / len(lattice)
```

### Results

| Model | Standard WER | Lattice WER | Outcome |
|-------|-------------|-------------|---------|
| Model 2 | 0.16 | **0.00** | ✅ Unfair penalty removed |
| Model 3 | 0.16 | **0.00** | ✅ Unfair penalty removed |
| Model 5 | 0.33 | **0.16** | ✅ Partial improvement |

> Models 2 and 3 were completely exonerated — their "errors" were actually valid alternate transcriptions.

### Alignment Unit Justification

**Word-level** alignment was chosen over subword/character because:
- Matches human evaluation intuition
- Devanagari word boundaries are well-defined
- Computationally simpler for this task scale

---

## 📈 Results Summary

<div align="center">

| Question | Metric | Result |
|----------|--------|--------|
| Q1 — Fine-Tuning | WER Baseline | 230.11% |
| Q1 — Fine-Tuning | WER Fine-tuned | ~41% |
| Q1 — Fine-Tuning | WER Reduction | **~189pp** |
| Q2 — Cleanup Pipeline | Number normalization | ✅ Implemented |
| Q2 — Cleanup Pipeline | English word tagging | ✅ Implemented |
| Q3 — Word Classification | Words analyzed | ~1,77,000 |
| Q3 — Word Classification | Low-conf accuracy | 12/25 correct |
| Q4 — Lattice WER | Model 2 improvement | 0.16 → 0.00 |
| Q4 — Lattice WER | Model 3 improvement | 0.16 → 0.00 |
| Q4 — Lattice WER | Model 5 improvement | 0.33 → 0.16 |

</div>

---

## ⚙️ Tech Stack

<div align="center">

| Category | Tools |
|----------|-------|
| **Core ML** | OpenAI Whisper, HuggingFace Transformers, PyTorch |
| **Audio Processing** | Librosa, Torchaudio, PyDub, FFmpeg |
| **Data** | Pandas, NumPy, HuggingFace Datasets |
| **Evaluation** | `jiwer` (WER), HuggingFace `evaluate` |
| **NLP** | Unicode / Devanagari normalization, custom rule engines |
| **Platform** | Google Colab (T4 GPU), Google Drive, Google Cloud Storage |
| **Visualization** | TensorBoard |

</div>

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install transformers datasets accelerate jiwer evaluate librosa torchaudio pydub soundfile
apt-get install ffmpeg
```

### Clone the Repository

```bash
git clone https://github.com/CoderJaynt/Speech-Audio-Research-Assignment.git
cd Speech-Audio-Research-Assignment
```

### Run Notebooks in Order

```
1. Whisper_Model_training___evaluation.ipynb       ← Q1: Fine-tune & evaluate
2. DesiTalks_Q2_Post_processing___NLP_pipeline.ipynb ← Q2: Cleanup pipeline
3. DesiTalks_Q3_Word_Classification.ipynb          ← Q3: Word-level scoring
4. DesiTalks_Q4.ipynb                              ← Q4: Lattice WER evaluation
```

> **Note:** All notebooks are designed for **Google Colab with T4 GPU**. Mount your Google Drive before running. URL patching is handled automatically in Q1.

### Data Access

The dataset uses the following URL format:
```
https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_transcription.json
```

---

## 📁 Repository Structure

```
Speech-Audio-Research-Assignment/
│
├── 📓 Whisper_Model_training___evaluation.ipynb
│       └── Q1: Data preprocessing, fine-tuning, FLEURS evaluation
│
├── 📓 DesiTalks_Q2_Post_processing___NLP_pipeline.ipynb
│       └── Q2: Number normalization, English word tagging
│
├── 📓 DesiTalks_Q3_Word_Classification.ipynb
│       └── Q3: Rule-based word spelling classification
│
├── 📓 DesiTalks_Q4.ipynb
│       └── Q4: Lattice construction & fair WER evaluation
│
├── 📄 Final_Combined_Assignment_Report.docx
│       └── Full written report covering all questions
│
└── 📋 README.md
        └── You are here!
```

---

<div align="center">

<!-- Footer Wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=fadeIn" width="100%"/>

**Made with ❤️ for Hindi ASR research**

[![LinkedIn](https://img.shields.io/badge/Jayant%20Yadav-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/jayant-yadav-a22b98283/)
[![GitHub](https://img.shields.io/badge/CoderJaynt-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/CoderJaynt)

*If this project helped you, consider giving it a ⭐ on GitHub!*

</div>
