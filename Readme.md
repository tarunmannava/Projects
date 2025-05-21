# DNA Trigram Tokenization using CUDA

This project implements a CUDA-based tokenizer that converts DNA sequences into trigram-based integer tokens for high-performance preprocessing in bioinformatics pipelines.

## 🧪 Project Overview

- **Language**: C++ with CUDA
- **Domain**: Bioinformatics (DNA Sequence Processing)
- **Goal**: Speed up DNA tokenization using GPU parallelism
- **Result**: Up to 339× faster than CPU implementation

## 📂 Files

- `code.cu`: Main CUDA program
- `sequence_samples.txt`: Sample DNA input
- `tokenization_results.txt`: Output tokens
- `DNA_Trigram_CUDA_Report.pdf`: Full technical report

## ⚙️ Compilation Instructions

```bash
nvcc -o tokenizer code.cu
./tokenizer
