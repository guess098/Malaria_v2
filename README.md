# Two-Stage Malaria Field Screener & Species Classifier (v2)

A browser-native, two-stage deep learning pipeline that scans full Giemsa-stained thin blood smear fields, localizes infected erythrocytes, filters out non-parasite artifacts, and identifies the four classic human *Plasmodium* species (*P. falciparum*, *P. vivax*, *P. malariae*, and *P. ovale*).

All inference runs **100% client-side** via **ONNX Runtime Web** (WebGPU / WebAssembly)—supporting both **live USB UVC microscope cameras** and **uploaded micrographs** with zero server latency and full image privacy.

---

## Architecture

1. **Stage 1 — Parasite Localization (`YOLO11n`):**
   - Scans full thin-smear fields (`960×960` letterboxed input) for a single target class (`parasite`) to maximize detection sensitivity across ring, trophozoite, schizont, and gametocyte stages.
   - Exported to FP16-quantized ONNX with embedded Non-Maximum Suppression (NMS).
2. **Context-Aware Auto-Cropping:**
   - Extracts padded `224×224` RGB patches around every Stage 1 bounding box from the full-resolution source frame, keeping surrounding healthy RBCs visible as a size reference for host-cell enlargement.
3. **Stage 2 — Species Classification & Artifact Rejection (`MobileNetV2`):**
   - Fine-tuned on 5 classes (`Falciparum`, `Malariae`, `Ovale`, `Vivax`, plus mined `Uninfected` background RBC patches).
   - Rejects Stage 1 false positives (platelets, stain debris) and aggregates `softmax` probabilities across all confirmed parasites on the slide via soft voting.

---

## Dataset & Leak-Proof Split

- **Source:** [MP-IDB (Malaria Parasite Image Database)](https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis) — 210 thin-smear fields (`2592×1944`) with pathologist-annotated masks.
- **Splitting Strategy:** Stratified **80% Train (168 slides) / 20% Validation (42 slides)** performed strictly **at the slide level** prior to cropping so patches from the same smear never leak across splits.

---

## Validation Performance (42 Unseen Full Smears)

| Stage / Scope | Key Metrics |
| :--- | :--- |
| **Stage 1 (`YOLO11n` @ 960px)** | **93.53%** Recall (Sensitivity) · **89.44%** Precision · **95.20%** mAP@50 (344 ground-truth parasites) |
| **Stage 2 (`MobileNetV2` @ 224px)** | **97.40%** Crop Accuracy · **0.889** Macro F1 across 5 classes (427 patches) |
| **End-to-End Slide Verdict** | **39 / 42 (92.86%)** Slide Accuracy · **0.911** Macro F1 · **100%** *P. falciparum* Recall (21/21) |

### Per-Species End-to-End Slide Breakdown ($n = 42$)

| Species | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| *P. falciparum* | 0.913 | 1.000 | 0.955 | 21 |
| *P. vivax* | 1.000 | 0.875 | 0.933 | 8 |
| *P. malariae* | 1.000 | 0.857 | 0.923 | 7 |
| *P. ovale* | 0.833 | 0.833 | 0.833 | 6 |

---

## Repository Structure

```text
├── index.html                 # Single-file web app (live camera, upload, demo slides, ONNX pipeline)
├── stage1_yolo11n.onnx        # FP16-quantized Stage 1 YOLO11n detector (embedded NMS)
├── stage2_mobilenetv2.onnx    # FP16-quantized Stage 2 5-class MobileNetV2 classifier
├── stage2_classes.json        # Class index mapping
├── samples/                   # 4 unseen validation micrographs for 1-click browser testing
│   ├── falciparum.jpg
│   ├── vivax.jpg
│   ├── malariae.jpg
│   └── ovale.jpg
└── README.md
