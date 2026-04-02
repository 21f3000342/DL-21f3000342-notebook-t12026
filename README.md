# 🎵 Music Genre Classification

**BSDA2001P: Introduction to DL and GenAI Project**

---

## 📋 Overview

This project tackles automatic music genre classification for the **Messy Mashup Kaggle competition**. The task requires predicting the correct genre label for noisy audio mashups constructed by:

1. Selecting instrument stems from **different songs of the same genre**
2. Applying **tempo adjustments** for rhythmic synchronisation
3. Mixing the stems into a single waveform
4. Overlaying **ESC-50 environmental noise** at random positions and SNR levels

The central challenge is a significant **train-test distribution shift** — training data consists of clean, source-separated stems while test samples are pre-built noisy mashups. The solution closes this gap by reconstructing the mashup generation process during training.

**10 genres:** `blues` · `classical` · `country` · `disco` · `hiphop` · `jazz` · `metal` · `pop` · `reggae` · `rock`

---

## 📦 Dataset

The **Messy Mashup** dataset is organised into four components under the `messy_mashup/` root:

```
messy_mashup/
├── genres_stems/               # 1,000 training songs × 4 stems each
│   ├── blues/
│   │   ├── blues.00001/
│   │   │   ├── bass.wav
│   │   │   ├── drums.wav
│   │   │   ├── other.wav
│   │   │   └── vocals.wav
│   │   └── ...                 # 100 songs per genre
│   └── ...                     # 10 genres total
├── ESC-50-master/
│   └── audio/                  # 2,000 noise clips × 50 categories × 5s each
├── mashups/                    # 3,020 unlabelled test files (~30s each)
├── test.csv                    # id → filename mapping
└── sample_submission.csv
```

**Key statistics:**
- 1,000 songs · 10 genres · 100 songs/genre · 4 stems/song = **4,000 stem files**
- All stems ≈ 30 seconds · sample rates: 22,050 Hz and 44,100 Hz
- No corrupted files (< 4 KB) found across training data
- Test mashups: mean duration 28.7 s · std 3.5 s · min 8.6 s

---

## 📊 Methodology

### Core Insight: Closing the Distribution Gap

The competition is fundamentally a **domain adaptation problem**. The solution reconstructs the test data generation process during training on-the-fly:

```
For each training sample:
  1. Pick 4 songs independently from the same genre
  2. Take bass/drums/other/vocals one each from the 4 songs (cross-song mixing)
  3. Scale each stem by a random volume factor ∈ [0.75, 1.25]
  4. Sum all 4 stems → RMS-normalise to −20 dBFS
  5. Inject 1–2 ESC-50 noise clips at random temporal positions, SNR ∈ [10, 28] dB
  6. Extract a random 10-second window from the 30-second mix
  7. Apply SpecAugment on the resulting spectrogram
```

This ensures the model sees data **statistically indistinguishable from the test set** at every training step. The genre label remains unambiguous because all four stems originate from the same class.

### Augmentation Pipeline

| Augmentation | Parameters | Applied to |
|---|---|---|
| Cross-song stem recombination | 4 different songs per sample | All models |
| Per-stem volume perturbation | Uniform[0.75, 1.25] | All models |
| RMS normalisation | −20 dBFS | All models |
| ESC-50 noise injection | 1–2 clips, SNR ∈ [10, 28] dB | All models |
| Random temporal crop | 10 s window | All models |
| SpecAugment — frequency | 2 masks × up to 20–24 bins, p=0.5–0.6 | All models |
| SpecAugment — time | 2 masks × up to 40–48 frames, p=0.5–0.6 | All models |
| Mixup | Beta(0.3–0.4, 0.3–0.4), p=0.3–0.5 | All models |
| Stem dropout | Drop 1 random stem, p=0.08 | ScratchCNN only |
| Tempo shift (spectrogram-domain) | Bilinear scale ∈ [0.85, 1.15], p=0.5 | ResNet50d only |
| Patchout — time axis | Up to 40% of time patches zeroed | AST only |
| Patchout — frequency axis | Up to 20% of freq patches zeroed | AST only |

---

## 🏗️ Model Architectures

### Model 1 — Audio Spectrogram Transformer (AST) ⭐ Best

```
Checkpoint : MIT/ast-finetuned-audioset-10-10-0.4593
Backbone   : 12-layer ViT encoder, 768-dim hidden states, 12 attention heads
Input      : [1024, 128] log-mel spectrogram → 512 patches of size 16×16
Parameters : 86.5M (all trainable, no frozen layers)

Input pipeline:
  Audio (16 kHz mono) → ASTFeatureExtractor → [1024 × 128]

Classification head (original head replaced):
  pooler_output [768]
  → LayerNorm(768)
  → Dropout(0.15)
  → Linear(768 → 384) + GELU
  → Dropout(0.10)
  → Linear(384 → 10)

Training-time regularisation:
  Patchout: p_time=0.40, p_freq=0.20
  Label smoothing: ε = 0.05
  Mixup: Beta(0.3, 0.3), probability = 0.30
```

### Model 2 — ResNet50d on Log-Mel Spectrograms

```
Backbone   : ResNet50d (timm), ImageNet pretrained
Modification: in_chans=1, num_classes=10
Parameters : ~25M

Feature pipeline (computed on-GPU):
  Audio (16 kHz) → MelSpectrogram(n_fft=512, hop=160, n_mels=128, f_min=20, f_max=8000)
  → AmplitudeToDB
  → Per-sample standardisation (μ=0, σ=1)
  → unsqueeze(1) → [B, 1, 128, ~1001]
  → Bilinear tempo-shift ∈ [0.85, 1.15], p=0.5 (training only)
  → SpecAugment × 2 (training only)
```

### Model 3 — From-Scratch CNN (Baseline)

```
Block 1:  Conv2d(1 → 32,   3×3, pad=1) + BN + ReLU + MaxPool(2×2)
Block 2:  Conv2d(32 → 64,  3×3, pad=1) + BN + ReLU + MaxPool(2×2)
Block 3:  Conv2d(64 → 128, 3×3, pad=1) + BN + ReLU + MaxPool(2×2)
Block 4:  Conv2d(128→ 256, 3×3, pad=1) + BN + ReLU + MaxPool(2×2)
Block 5:  Conv2d(256→ 512, 3×3, pad=1) + BN + ReLU + MaxPool(2×2)
          AdaptiveAvgPool2d(1×1) → Flatten
Head:     Linear(512→256) + ReLU + Dropout(0.4)
          Linear(256→128) + ReLU + Dropout(0.3)
          Linear(128→10)

Total parameters: 1,735,498
```

---

## ⚙️ Training Configuration

| Hyperparameter | AST | ResNet50d | ScratchCNN |
|---|---|---|---|
| Sample rate | 16 kHz | 16 kHz | 22.05 kHz |
| Crop length | 10 s | 10 s | 10 s |
| Batch size | 48 (2× T4 GPU) | 96 | 64 (2× T4 GPU) |
| Epochs | 25 | 35 | 30 |
| Optimizer | AdamW | AdamW | AdamW |
| Peak LR | 3×10⁻⁵ | 4×10⁻⁴ | 1×10⁻³ |
| LR schedule | OneCycleLR | OneCycleLR | OneCycleLR |
| Warmup fraction | 5% | 5% | 5% |
| Weight decay | 0.02 | 0.01 | 1×10⁻⁴ |
| Label smoothing | 0.05 | 0.05 | 0.05 |
| Mixup probability | 0.30 | 0.50 | 0.30 |
| Mixup alpha | 0.30 | 0.40 | 0.30 |
| Train samples/epoch | 10,000 | 10,000 | 8,000 |
| Validation samples | 2,000 | 1,500 | 1,500 |
| TTA crops (inference) | 7 | 11 | 7 |
| Gradient clipping | 1.0 | — | 1.0 |
| Mixed precision | AMP FP16 | AMP FP16 | AMP FP16 |
| Early stopping patience | 8 | — | 10 |
| **Best Val Macro F1** | **0.9985** | **0.9993** | **0.8719** |

---

## 📈 Results

### Leaderboard Scores

| Model | Val F1 | Public LB | Private LB | Notes |
|---|---|---|---|---|
| ScratchCNN (baseline) | 0.8719 | 0.80028 | 0.82224 | 5-layer CNN from scratch |
| ResNet50d | 0.9993 | 0.96856 | 0.96332 | timm pretrained, torchaudio pipeline |
| AST (single run) | 0.9985 | 0.98191 | 0.97634 | Best individual model |
| **AST × 3 Ensemble** | — | **0.98521** | **0.98069** | **Final submission** |

---

## 🎯 Key Findings

**1. Distribution alignment is the primary lever.**
All three models converge to high accuracy because all share the same synthetic mashup augmentation. The augmentation closes the train-test distribution gap; architecture determines how efficiently the signal is exploited.

**2. AudioSet pretraining provides an enormous head start.**
The AST reaches Macro F1 > 0.90 at epoch 1 — before any domain-specific fine-tuning has meaningfully occurred. Without pretraining the ScratchCNN needs 12+ epochs to reach the same level and plateaus ~18 F1 points lower.

**3. Pretrained CNN vs. pretrained Transformer gap is narrower than expected.**
ResNet50d (25M params) achieves 0.969 LB vs. AST's 0.982 (86.5M params) — a 3.5× parameter increase buys only 1.3 absolute F1 points. For latency-constrained deployment ResNet50d is a strong alternative.

**4. Rock is the hardest genre consistently across all models.**
Its spectral overlap with blues (shared guitar timbre) and country (similar tempo) causes the most cross-genre confusion. This is the only genre that occasionally dips below F1 = 0.95 even in the strongest AST runs.

**5. On-GPU feature extraction is critical for training throughput.**
Moving MelSpectrogram computation to the GPU (ResNet50d) removes the CPU DataLoader bottleneck and allows 2× larger effective batch sizes compared to the librosa-based CPU pipeline used in ScratchCNN.

---

## 🎓 Future Work

- **Source separation at inference** — Run Demucs v4 on test mashups to re-separate stems before classification, effectively inverting the mashup process and potentially reaching 0.99+
- **Architecture diversity in ensemble** — Including ResNet50d probabilities alongside AST would introduce genuine architectural diversity for larger ensemble gains than same-architecture seeds
- **SNR calibration** — The ~0.015 gap between validation F1 (~0.998) and leaderboard F1 (~0.982) likely reflects a mismatch between the simulated SNR range [10, 28] dB and the true test distribution
- **Longer crops** — 15–20 second crops would capture longer-range rhythmic patterns at the cost of increased memory and compute
- **Hard-negative mining for confusable genres** — Targeted augmentation for the rock/blues/country cluster would directly address the most common failure mode

---

## 🚀 Deployment

A live demo is deployed on HuggingFace Spaces:

🔗 **[music-genre-classifier](https://huggingface.co/spaces/neerajs7/music-genre-classifier)**

**Features:**
- Upload or record audio
- Top 3 genre predictions with confidence scores
- Test-Time Augmentation (7 crops)
- Professional Gradio interface

