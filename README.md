# Intelligent Signal Processing — End-of-Term Submission

> End-of-term practical assessment covering video-based computer vision, lossless audio compression, and automated multimedia format validation using Python and OpenCV.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Exercise 1 — Vehicle Detection & Tracking](#exercise-1--vehicle-detection--tracking)
- [Exercise 2 — Rice Coding for Lossless WAV Compression](#exercise-2--rice-coding-for-lossless-wav-compression)
- [Exercise 3 — Film Festival Format Validator](#exercise-3--film-festival-format-validator)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)

---

## Project Overview

This repository contains three practical exercises submitted as part of an Intelligent Signal Processing (ISP) end-of-term assessment. The exercises span three core domains of signal processing:

1. **Spatial/temporal signal analysis** — detecting and tracking moving vehicles in traffic footage using background modelling and contour analysis.
2. **Lossless source coding** — compressing monaural PCM audio with Rice coding and verifying byte-perfect reconstruction via SHA-256.
3. **Multimedia format compliance** — automating the validation and re-encoding of film submissions against a defined festival specification using `ffprobe` and `ffmpeg`.

Each exercise is self-contained inside its own Jupyter notebook and supporting file directory.

---

## Repository Structure

```
Intelligent_Signal_Processing/
│
├── ISP_EOT_FINAL_SUBMISSION/
│   ├── exercise 1/
│   │   └── Exercise_1_ISP_Final/
│   │       ├── EX1.1-ISP-Final.ipynb       # Task 1.1 — Vehicle detection & tracking
│   │       ├── EX1.2-ISP-Final.ipynb       # Task 1.2 — Downtown car counter
│   │       └── Exercise1_Files/
│   │           ├── Traffic_Laramie_1.mp4
│   │           └── Traffic_Laramie_2.mp4
│   │
│   ├── exercise 2/
│   │   └── Exercise_2_ISP_Final/
│   │       ├── EX2-ISP-Final.ipynb         # Rice coding compression pipeline
│   │       ├── Exercise2_Files/
│   │       │   ├── Sound1.wav
│   │       │   └── Sound2.wav
│   │       ├── outputs_no_delta/           # Compressed outputs without delta encoding
│   │       └── outputs_with_delta/         # Compressed outputs with delta encoding
│   │
│   └── exercise 3/
│       └── Exercise_3_ISP_Final/
│           ├── Ex3-ISP-Final.ipynb         # Festival format validator & converter
│           ├── Exercise3_Files/            # Input film submissions (mp4, mov, avi)
│           └── outputs/
│               ├── format_audit_report.txt
│               └── converted/             # Re-encoded compliant files
│
├── Intelligent_Signal_Processing_EOT.pdf   # Full submission report
└── Intelligent_Signal_Processing_EOT_Codes.pdf
```

---

## Exercise 1 — Vehicle Detection & Tracking

### Task 1.1 — Background Subtraction & Contour-Based Detection

**Notebook:** `EX1.1-ISP-Final.ipynb`

This task implements a `TrafficDetector` class that processes traffic video footage to detect and track moving vehicles using classical computer vision techniques.

**Pipeline:**

1. Load video and validate capture parameters (resolution, FPS, frame count).
2. Crop each frame to a manually defined Region of Interest (ROI) to exclude irrelevant scene areas.
3. Apply **Gaussian blurring** to reduce high-frequency noise prior to background modelling.
4. Run **KNN Background Subtraction** (`cv2.createBackgroundSubtractorKNN`) to isolate moving foreground pixels. The KNN model builds a multi-sample per-pixel background estimate over a configurable history window.
5. Extract vehicle contours from the foreground mask; discard contours below a minimum area threshold (`min_area = 4000 px²`) to suppress noise and pedestrians.
6. Draw bounding boxes and per-frame detection counts on output frames.
7. Write annotated video to disk (`.avi`).

**Key parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `min_area` | 4000 px² | Minimum contour area for a valid vehicle detection |
| KNN history | default | Number of past frames used to build background model |
| Gaussian kernel | configurable | Pre-blur for noise suppression |

**Input:** `Traffic_Laramie_1.mp4`, `Traffic_Laramie_2.mp4`
**Output:** Annotated `.avi` video files, saved IR test frames (`frame_with_IR.jpg`, `IR_test.jpg`)

---

### Task 1.2 — Downtown Car Counter

**Notebook:** `EX1.2-ISP-Final.ipynb`

Extends the detection pipeline with a centroid-based tracker and directional gate to count vehicles that are moving towards the downtown zone.

**Pipeline:**

1. Apply the same Gaussian blur + KNN background subtraction as Task 1.1.
2. For each detected contour, compute its centroid and associate it with an existing tracked `Car` object via nearest-neighbour matching.
3. Each `Car` object records its current and previous position, a missing-frame counter, and a `counted` flag.
4. A vehicle is counted once when its centroid crosses a predefined virtual trip wire moving in the downtown direction.
5. Vehicles that remain unmatched for a configurable number of frames are removed from the tracker.

**Key design decisions:**

- **Single-count guarantee:** The `counted` flag prevents re-counting vehicles that stall near the trip wire.
- **Robustness to occlusion:** The `frames_missing` counter allows short-term occlusions without track loss.

---

## Exercise 2 — Rice Coding for Lossless WAV Compression

**Notebook:** `EX2-ISP-Final.ipynb`

This exercise implements a complete lossless audio compression and decompression pipeline using **Rice coding**, encoding mono PCM audio into a custom binary format (`.ex2`) and verifying perfect reconstruction through SHA-256 digest comparison.

### Methodology

**1. WAV Ingestion (`read_wav_mono`)**
Reads 16-bit PCM `.wav` files. Stereo files are downmixed to mono by retaining the left channel. All metadata (sample rate, channel count, sample width) is preserved in a `WavPayload` dataclass for use during decoding.

**2. Zigzag Mapping**
Rice coding requires non-negative integers. Signed `int16` residuals are bijectively mapped to unsigned values using zigzag encoding:

```
0 → 0,  -1 → 1,  1 → 2,  -2 → 3,  2 → 4, ...
```

The inverse mapping is applied exactly during decoding.

**3. Rice Encoding**
Each unsigned sample `u` is split at Rice parameter `K`:
- **Quotient:** `q = u >> K` — encoded as `q` zero-bits followed by a stop-bit (`1`).
- **Remainder:** `r = u & ((1 << K) - 1)` — encoded as a `K`-bit fixed-width binary word.

Codeword positions are computed via a cumulative sum over codeword lengths, enabling fully vectorised bitstream packing with NumPy.

**4. Custom `.ex2` File Format**
Compressed files include:
- Magic header bytes `RCEX2` for format identification.
- Metadata block: sample rate, channels, sample width, sample count, Rice parameter `K`.
- Delta encoding flag.
- Packed Rice bitstream.

**5. Delta Encoding (optional)**
When enabled, first-order differences between consecutive samples replace the raw sample values prior to Rice coding. This typically reduces residual magnitude for correlated audio signals, improving compression ratio.

**6. Verification**
SHA-256 digests of the original and decoded `int16` sample arrays are compared to confirm byte-perfect lossless reconstruction.

### Experimental Conditions

Each `.wav` file was compressed under four conditions:

| Condition | K | Delta |
|-----------|---|-------|
| NoDelta_K2 | 2 | Off |
| NoDelta_K4 | 4 | Off |
| Delta_K2   | 2 | On  |
| Delta_K4   | 4 | On  |

**Libraries:** `wave`, `numpy`, `struct`, `hashlib`, `pandas`, `dataclasses`, `pathlib`

---

## Exercise 3 — Narbonne Film Festival Format Validator

**Notebook:** `Ex3-ISP-Final.ipynb`

This exercise automates the format validation and transcoding pipeline for the Narbonne Online Film Festival, which receives over 100 film submissions annually in heterogeneous formats.

### Festival Specification

| Field | Required Value |
|-------|---------------|
| Container | `mp4` |
| Video codec | `hevc` (H.265) |
| Audio codec | `aac` |
| Frame rate | 25.0 fps |
| Aspect ratio | 16:9 |
| Resolution | 640 × 360 px |
| Video bitrate | 2,000,000 – 5,000,000 bps |
| Audio bitrate | ≤ 256,000 bps |
| Audio channels | 2 (stereo) |

### Pipeline

1. **Tool verification** — checks that `ffprobe` and `ffmpeg` are installed and accessible on the system PATH. Installs via Homebrew (macOS), `apt` (Linux), or `choco` (Windows) if missing.
2. **Probing** — each input file (`.mp4`, `.mov`, `.avi`, `.mkv`) is passed to `ffprobe` with JSON output to extract container format, video stream parameters, and audio stream parameters.
3. **Validation** — probed values are compared field-by-field against `target_specs_required`. Non-compliance is recorded with a human-readable description of each failing field.
4. **Report generation** — a plain-text `format_audit_report.txt` is written summarising pass/fail status and specific issues for each submission.
5. **Re-encoding** — non-compliant files are automatically transcoded with `ffmpeg` to produce `<filename>_formatOK.mp4` in `outputs/converted/`, applying the required codec, resolution, frame rate, bitrate, and audio parameters.
6. **Post-conversion verification** — each converted file is re-probed and re-validated to confirm the transcoding fully resolved all compliance issues.

**Input files processed:**

| File | Original Format |
|------|----------------|
| `Cosmos_War_of_the_Planets.mp4` | MP4 |
| `Last_man_on_earth_1964.mov` | QuickTime MOV |
| `The_Gun_and_the_Pulpit.avi` | AVI |
| `The_Hill_Gang_Rides_Again.mp4` | MP4 |
| `Voyage_to_the_Planet_of_Prehistoric_Women.mp4` | MP4 |

**Libraries:** `json`, `subprocess`, `pathlib`, `math`, `glob`, `platform`

---

## Dependencies

Install the required Python packages before running any notebook:

```bash
pip install numpy opencv-python pandas tabulate
```

`ffmpeg` and `ffprobe` are required for Exercise 3. On most systems:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

Python version: **3.8+** recommended.

---

## How to Run

Each exercise is a standalone Jupyter notebook. Open the notebook for the desired exercise and run all cells in order:

```bash
# Exercise 1.1
jupyter notebook "ISP_EOT_FINAL_SUBMISSION/exercise 1/Exercise_1_ISP_Final/EX1.1-ISP-Final.ipynb"

# Exercise 1.2
jupyter notebook "ISP_EOT_FINAL_SUBMISSION/exercise 1/Exercise_1_ISP_Final/EX1.2-ISP-Final.ipynb"

# Exercise 2
jupyter notebook "ISP_EOT_FINAL_SUBMISSION/exercise 2/Exercise_2_ISP_Final/EX2-ISP-Final.ipynb"

# Exercise 3
jupyter notebook "ISP_EOT_FINAL_SUBMISSION/exercise 3/Exercise_3_ISP_Final/Ex3-ISP-Final.ipynb"
```

Each notebook expects its accompanying `Exercise*_Files/` directory to be present at the relative paths defined within the notebook. No additional configuration is required.

---

*Submitted as part of the Intelligent Signal Processing End-of-Term Assessment.*
