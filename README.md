<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Hindi%20ASR%20System&fontSize=50&fontAlignY=35&desc=End-to-End%20Hindi%20Speech%20Recognition%20with%20Whisper&descAlignY=58&descAlign=50&animation=fadeIn&fontColor=ffffff" width="100%"/>

<br/>

<a href="#"><img src="https://img.shields.io/badge/🎙️%20ASR-Whisper%20Small-4A90D9?style=for-the-badge&logoColor=white"/></a>
<a href="#"><img src="https://img.shields.io/badge/🇮🇳%20Language-Hindi%20%7C%20Devanagari-FF6B35?style=for-the-badge"/></a>
<a href="#"><img src="https://img.shields.io/badge/🧠%20Framework-HuggingFace-FFD43B?style=for-the-badge&logo=huggingface&logoColor=black"/></a>
<a href="#"><img src="https://img.shields.io/badge/📉%20WER%20Reduction-230%%20→%2041%%20-00C853?style=for-the-badge"/></a>
<a href="#"><img src="https://img.shields.io/badge/🔬%20Evaluation-Lattice%20Based-9C27B0?style=for-the-badge"/></a>
<a href="#"><img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black"/></a>

<br/><br/>

> **👨‍💻 Author:** Jayant Yadav  
> **📅 Domain:** Speech & Audio | Automatic Speech Recognition | NLP

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/jayant-yadav-a22b98283/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/CoderJaynt)

</div>

---

## 📋 Table of Contents

- [✨ Project Overview](#-project-overview)
- [📂 Dataset](#-dataset)
- [🏗️ System Architecture](#️-system-architecture)
- [📊 Phase 1 — ASR Pipeline & Fine-Tuning](#-phase-1--asr-pipeline--fine-tuning)
- [🧹 Phase 2 — Post-Processing & NLP Pipeline](#-phase-2--post-processing--nlp-pipeline)
- [🧠 Phase 3 — Word-Level Classification](#-phase-3--word-level-classification)
- [🔬 Phase 4 — Lattice-Based Evaluation](#-phase-4--lattice-based-evaluation)
- [📈 Results Summary](#-results-summary)
- [⚙️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [📁 Repository Structure](#-repository-structure)

---

## ✨ Project Overview

This repository presents a **complete, production-grade Automatic Speech Recognition (ASR) pipeline** built for Hindi conversational speech.

The system goes far beyond basic transcription:

| Component | What it does |
|-----------|-------------|
| 🎯 **Fine-Tuning** | Adapts Whisper-small to domain-specific Hindi audio, reducing WER from 230% → ~41% |
| 🧹 **Post-Processing** | Intelligent cleanup pipeline for numbers, English loanwords, and repetitions |
| 🧠 **Word Classification** | Automated correct/incorrect spelling detection across ~1.77 lakh unique words |
| 🔬 **Lattice Evaluation** | Fair, multi-reference WER computation that doesn't unfairly penalize valid transcriptions |

### 🎯 Motivation

Hindi is a low-resource language for ASR. Off-the-shelf models like Whisper, despite being multilingual, perform poorly on conversational Hindi due to:

1. **Rigid evaluation** — Single ground-truth strings unfairly penalize valid alternate spellings/pronunciations
2. **Code-switching blindness** — Hindi-English mixed speech causes systematic errors
3. **No interpretability** — Opaque errors with no actionable categorization

**This project addresses all three.**

---

## 📂 Dataset

The model was fine-tuned on **~10 hours of Hindi conversational speech data** (`FT Data - data.csv`), included in this repository.

Each record contains:
- 🎙️ **Audio** — Hindi conversational speech recordings
- 📝 **Transcription metadata** — Human-annotated Devanagari transcriptions with segment-level timestamps

**Preprocessing applied before training:**
- Duration filtering (1–30 second segments only)
- Unicode NFC normalization for consistent Devanagari representation
- Whitespace cleanup and trivial transcription removal
- Resampling to 16kHz mono WAV
- Speaker-aware train/validation split (90/10, stratified by speaker, `seed=42`)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Hindi ASR Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Raw Audio + FT Data - data.csv                                    │
│        │                                                            │
│        ▼                                                            │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐      │
│   │  Audio   │───▶│   Whisper    │───▶│   Raw ASR Output    │      │
│   │  Ingest  │    │  Fine-Tuned  │    │  (Hindi Devanagari) │      │
│   └──────────┘    └──────────────┘    └─────────────────────┘      │
│        │                                        │                   │
│        │ Phase 1                                │ Phase 2           │
│        ▼                                        ▼                   │
│   ┌──────────────────┐              ┌───────────────────────┐      │
│   │  Preprocessing   │              │   Cleanup Pipeline    │      │
│   │  • 16kHz resamp  │              │  • Number normalize   │      │
│   │  • Text norm     │              │  • English tagging    │      │
│   │  • Filtering     │              │  • Repetition removal │      │
│   └──────────────────┘              └───────────────────────┘      │
│                                                 │                   │
│                                                 │ Phase 3           │
│                                                 ▼                   │
│                                    ┌───────────────────────┐       │
│                                    │  Word Classification  │       │
│                                    │  • Rule-based system  │       │
│                                    │  • Confidence scoring │       │
│                                    │  • 1.77L words        │       │
│                                    └───────────────────────┘       │
│                                                 │                   │
│                                                 │ Phase 4           │
│                                                 ▼                   │
│                                    ┌───────────────────────┐       │
│                                    │  Lattice Evaluation   │       │
│                                    │  • Multi-ref WER      │       │
│                                    │  • Fair scoring       │       │
│                                    └───────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Phase 1 — ASR Pipeline & Fine-Tuning

**Notebook:** `Whisper_Model_training___evaluation.ipynb`

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

After systematically sampling 25+ error utterances:

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
| Number word mismatch | Number normalization pipeline | ✅ Implemented (Phase 2) |
| Code-mix handling | English word tagging + dictionary expansion | ✅ Implemented (Phase 2) |

---

## 🧹 Phase 2 — Post-Processing & NLP Pipeline

**Notebook:** `Hindi_ASR_Post_processing_NLP.ipynb`

### a) Number Normalization

Converts spoken Hindi number words into digit form.

```python
num_map = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, ...
    "सौ": 100, "हजार": 1000
}
```

| Input | Output |
|-------|--------|
| `दो` | `2` |
| `तीन सौ चौवन` | `354` |
| `एक हज़ार` | `1000` |

**Edge Case Handling:**

| Input | Decision | Reason |
|-------|----------|--------|
| `दो-चार बातें` | ❌ Skip | Idiomatic — "a few things" |
| `दो किताबें` | ✅ Convert → `2 किताबें` | Literal count |

### b) English Word Detection & Tagging

English words spoken in Hindi conversation are often transcribed in Devanagari. These need separate handling for downstream tasks.

**Detection rules:**
1. **Dictionary lookup** — Known English loanwords list
2. **Roman script detection** — Tokens containing `[a-zA-Z]`
3. **Phonetic suffix matching** — Endings like `शन`, `मेंट`, `टिंग`

```
Input:  "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई"
Output: "मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई"
```

---

## 🧠 Phase 3 — Word-Level Classification

**Notebook:** `Hindi_ASR_Word_Classification.ipynb`

### Objective

Classify ~1,77,000 unique words from human-transcribed Hindi data into correctly or incorrectly spelled — avoiding re-transcription of the entire dataset.

### Approach

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

### Known Failure Modes

| Word Category | Why It's Hard |
|---------------|---------------|
| **Rare proper nouns** | Not in dictionary, low frequency — looks like error but isn't |
| **English loanwords in Devanagari** | Valid but pattern-matches as suspicious |
| **Dialectal variants** | Regional spellings confuse the rule engine |

---

## 🔬 Phase 4 — Lattice-Based Evaluation

**Notebook:** `Hindi_ASR_Lattice_Evaluation.ipynb`

### The Problem with Standard WER

Standard WER compares against a **single rigid reference**, unfairly penalizing:
- Valid number representations (`14` vs `चौदह`)
- Acceptable spelling variants (`किताबें` vs `किताबे`)
- Lexical synonyms (`पुस्तकें` vs `किताबें`)

### Lattice Solution

A lattice replaces the flat reference with bins — each bin contains all valid alternatives at that position.

```
Spoken: "उसने चौदह किताबें खरीदीं"

Lattice: [["उसने"],
           ["चौदह", "14"],
           ["किताबें", "किताबे", "पुस्तकें"],
           ["खरीदीं", "खरीदी"]]
```

### Results

| Model | Standard WER | Lattice WER | Outcome |
|-------|-------------|-------------|---------|
| Model 2 | 0.16 | **0.00** | ✅ Unfair penalty removed |
| Model 3 | 0.16 | **0.00** | ✅ Unfair penalty removed |
| Model 5 | 0.33 | **0.16** | ✅ Partial improvement |

---

## 📈 Results Summary

<div align="center">

| Phase | Metric | Result |
|-------|--------|--------|
| Phase 1 — Fine-Tuning | WER Baseline | 230.11% |
| Phase 1 — Fine-Tuning | WER Fine-tuned | ~41% |
| Phase 1 — Fine-Tuning | WER Reduction | **~189pp** |
| Phase 2 — Cleanup Pipeline | Number normalization | ✅ Implemented |
| Phase 2 — Cleanup Pipeline | English word tagging | ✅ Implemented |
| Phase 3 — Word Classification | Words analyzed | ~1,77,000 |
| Phase 4 — Lattice WER | Model 2 improvement | 0.16 → 0.00 |
| Phase 4 — Lattice WER | Model 3 improvement | 0.16 → 0.00 |
| Phase 4 — Lattice WER | Model 5 improvement | 0.33 → 0.16 |

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
| **Platform** | Google Colab (T4 GPU), Google Drive |
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
1. Whisper_Model_training___evaluation.ipynb   ← Phase 1: Fine-tune & evaluate
2. Hindi_ASR_Post_processing_NLP.ipynb         ← Phase 2: Cleanup pipeline
3. Hindi_ASR_Word_Classification.ipynb         ← Phase 3: Word-level scoring
4. Hindi_ASR_Lattice_Evaluation.ipynb          ← Phase 4: Lattice WER evaluation
```

> **Note:** All notebooks are designed for **Google Colab with T4 GPU**. Mount your Google Drive before running.

---

## 📁 Repository Structure

```
Speech-Audio-Research-Assignment/
│
├── 📊 FT Data - data.csv
│       └── ~10 hours of Hindi conversational speech training data
│
├── 📓 Whisper_Model_training___evaluation.ipynb
│       └── Phase 1: Data preprocessing, fine-tuning, evaluation
│
├── 📓 Hindi_ASR_Post_processing_NLP.ipynb
│       └── Phase 2: Number normalization, English word tagging
│
├── 📓 Hindi_ASR_Word_Classification.ipynb
│       └── Phase 3: Rule-based word spelling classification
│
├── 📓 Hindi_ASR_Lattice_Evaluation.ipynb
│       └── Phase 4: Lattice construction & fair WER evaluation
│
└── 📋 README.md
        └── You are here!
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=fadeIn" width="100%"/>

**Made with ❤️ for Hindi ASR research**

[![LinkedIn](https://img.shields.io/badge/Jayant%20Yadav-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/jayant-yadav-a22b98283/)
[![GitHub](https://img.shields.io/badge/CoderJaynt-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/CoderJaynt)

*If this project helped you, consider giving it a ⭐ on GitHub!*

</div>
