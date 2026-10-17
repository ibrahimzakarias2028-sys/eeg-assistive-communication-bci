# SSVEP-Based Brain-Computer Interface for Assistive Communication

#### data source: https://www.sciencedirect.com/science/article/pii/S2352340921001104#article

#### Description:
> **Turning brainwaves into voice — giving a voice back to those who need it most.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7%2B-green)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🧠 Overview & Mission

This project explores how **Brain-Computer Interfaces (BCIs)** can bridge the gap between the human mind and the external world — especially for individuals who have lost all voluntary muscle control.

Designed for patients suffering from **Complete Quadriplegia**, **Amyotrophic Lateral Sclerosis (ALS)**, and **Locked-in Syndrome (LIS)**, this system is an **SSVEP-based BCI assistant** that translates a user's neural responses into spoken language. By focusing their gaze on a flickering visual stimulus, a patient can express fundamental needs — water, food, bathroom, or sleep — without moving a single muscle.

> *"Technology is at its best when it restores what was once thought lost."*

---

## 📁 Project Files Description

To adhere to a modular and testable architecture, the project is structured into three primary files:

*   **`project.py`**: The core application housing the main function and the signal processing pipeline. It handles file validation, raw EEG data ingestion, signal detrending, Butterworth SOS bandpass filtering, Welch's Power Spectral Density (PSD) estimation, target frequency extraction, and offline Text-to-Speech (TTS) synthesis.
*   **`test_project.py`**: A comprehensive test suite utilizing `pytest`. It contains custom unit tests (`test_validate_file`, `test_read_data`, and `test_extract_target_frequency`) to rigorously verify data integrity, file handling exceptions, and the mathematical accuracy of the frequency classification logic.
*   **`requirements.txt`**: A pip-installable dependencies file listing the external libraries required to execute the pipeline (`numpy`, `scipy`, `pyttsx3`, `pytest`).

---

## 🔬 The Science Behind It: SSVEP

The system relies on **Steady-State Visually Evoked Potentials (SSVEP)** — a neurophysiological phenomenon where the visual cortex generates electrical oscillations matching the exact frequency of a repetitive visual stimulus.

By decoding this signature from a single-channel EEG signal, we can infer which stimulus the patient is looking at.

| Frequency (Hz) | Command | Spoken Output |
|:---:|:---|:---|
| **8.57** | 💧 Water | *"I want water"* |
| **10.00** | 🍽️ Food | *"I want to eat"* |
| **12.00** | 🚻 Bathroom | *"I want to go to bathroom"* |
| **15.00** | 😴 Sleep | *"I want to sleep"* |

---

## 🏗️ System Architecture & Signal Processing Pipeline

The software follows a rigorous clinical signal processing pipeline:

### 1. Data Ingestion & Validation
Accepts raw single-channel EEG recordings in CSV/text format. It programmatically verifies file existence and ensures pure numeric content (preventing array casting errors).

### 2. Signal Preprocessing
- **Linear Detrending**: Utilizes `scipy.signal.detrend` to eliminate DC offset and low-frequency baseline drift caused by electrode impedance fluctuations.
- **Butterworth Bandpass Filtering**: A 4th-order filter isolates the 7.5–30.0 Hz band.

### 3. Spectral Analysis & Classification
Applies **Welch's Method** (1024-sample Hann window, 50% overlap at `fs = 256 Hz`) to transform the discrete-time signal into a high-resolution Power Spectral Density (PSD). The system measures spectral power at the four target frequencies and classifies the user's intent using maximum power via `np.argmax`.

### 4. Voice Output
Maps the classified frequency to its corresponding command and synthesizes natural speech using the `pyttsx3` offline engine.

---

## 🚀 Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the system: `python project.py`
3. Execute unit tests: `pytest test_project.py`

---

## ⚖️ Design Choices & Engineering Trade-offs

During development, several technical alternatives were debated to ensure reliability:

| Alternative Considered | Why It Was Rejected & The Final Choice |
|:---|:---|
| **Plain FFT (`scipy.fft`)** | Raw FFT produces high-variance spectral estimates. **Welch's method** was chosen because segment averaging significantly reduces noise variance, which is critical for detecting weak SSVEP peaks in spontaneous background EEG. |
| **Direct-form (ba) Filter** | Direct-form coefficients suffer from numerical instability in high-order filters. **SOS (Second-Order Sections)** cascades were chosen for superior numerical precision. |
| **`filtfilt` vs `lfilter`** | Standard filtering introduces phase delay. `sosfiltfilt` (forward-backward zero-phase filtering) was explicitly chosen to prevent temporal distortion. |
| **Cloud-based TTS APIs** | Cloud TTS introduces latency and requires internet access. `pyttsx3` was initially used, then replaced with **Piper TTS** for higher voice naturalness while maintaining zero-latency, fully offline execution. |

---

## 🙏 Acknowledgments

This project was born from a deep belief that engineering should serve humanity. I am grateful to the open-source scientific Python community for providing the tools that make neurotechnology accessible.[cite: 1]

> *"The best way to predict the future is to code it."*[cite: 1]

<p align="center">
  <em>Built with 💻, 🧠, and curiosity.</em><br>
  <strong>Computer Engineering | Neurotechnology | Assistive Technology</strong>[cite: 1]
</p>
