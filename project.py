"""SSVEP-Based Brain-Computer Interface (BCI) for Assistive Communication.

File: project.py
Author: Ibrahim Zakaria Hassan Mohamed
License: MIT
Domain: Biomedical Engineering / Assistive Neurotechnology

Overview
--------
This module provides an end-to-end signal processing and classification pipeline
for single-channel Electroencephalography (EEG) data. It decodes Steady-State
Visually Evoked Potentials (SSVEP) into real-time offline text-to-speech output,
enabling non-verbal assistive communication.

Target Population
-----------------
Designed specifically for patients suffering from complete quadriplegia,
Amyotrophic Lateral Sclerosis (ALS), or Locked-in Syndrome (LIS) who retain full
cognitive function but lack voluntary motor and speech control.
"""

import sys
from typing import Tuple
import numpy as np
from scipy import signal
from piper import PiperVoice, SynthesisConfig
import sounddevice as sd

# ---------------------------------------------------------------------------
# Global Pipeline Parameters & Hardware Configuration
# ---------------------------------------------------------------------------
SAMPLING_RATE: int = 256  # Hz — Satisfies Nyquist-Shannon sampling theorem for SSVEP

STIMULUS_FREQUENCIES: np.ndarray = np.array([8.57, 10.00, 12.00, 15.00])

FREQUENCY_TO_COMMAND: dict[float, str] = {
    8.57: "I want water",
    10.00: "I want to eat",
    12.00: "I want to go to bathroom",
    15.00: "I want to sleep",
}


def main() -> None:
    """Orchestrate the EEG signal processing and speech synthesis pipeline."""
    file_path = input("Enter The File Name: ")
    validate_file(file_path)

    raw_data = read_data(file_path)
    filtered_data = filter_signal(raw_data)
    frequencies, power_spectrum = compute_power_spectrum(filtered_data)

    target_frequency = extract_target_frequency(frequencies, power_spectrum)
    command_text = to_speech(target_frequency)

    print(f"\n[BCI Output]: {command_text}")

def validate_file(file_path: str) -> None:
    """Validate input dataset existence and verify numerical data integrity.

    Parameters
    ----------
    file_path : str
        Relative or absolute path to the target text/CSV file.

    Raises
    ------
    SystemExit
        If the target file does not exist or contains non-numeric alphabetic characters.

    Notes
    -----
    Non-numeric artifacts (such as CSV column headers or string timestamps) corrupt
    array casting during file parsing and distort frequency analysis.
    """
    try:
        with open(file_path, mode="r") as file:
            content = file.read()
    except FileNotFoundError:
        sys.exit("Error: File does not exist. Please check the file path.")

    if any(char.isalpha() for char in content):
        sys.exit("Error: Non-numeric content detected. Please remove text headers/labels.")


def read_data(file_path: str) -> np.ndarray:
    """Load raw 1-D time-series EEG signal samples from disk.

    Parameters
    ----------
    file_path : str
        Validated file path containing numerical EEG voltage values.

    Returns
    -------
    np.ndarray
        1-D array containing raw discrete-time EEG voltage measurements.

    Raises
    ------
    ValueError
        If the loaded file contains zero numerical data points.
    """
    data = np.loadtxt(file_path)
    if data.size == 0:
        raise ValueError("Loaded EEG dataset is empty.")
    return data


def filter_signal(raw_data: np.ndarray) -> np.ndarray:
    """Pre-process raw EEG signal via linear detrending and bandpass filtering.

    Parameters
    ----------
    raw_data : np.ndarray
        Raw 1-D time-series EEG voltage signal.

    Returns
    -------
    np.ndarray
        Filtered 1-D signal constrained to the 7.50–30.00 Hz passband.

    Notes
    -----
    - **Linear Detrending**: Removes low-frequency DC offset and baseline drift
      caused by electrode impedance variations and subject body movements.
    - **Filter Design**: Implements a 4th-order Butterworth bandpass filter configured
      in Second-Order Sections (SOS) to avoid numerical instability inherent in
      high-order direct-form (ba) transfer functions.
    - **Zero-Phase Filtering**: Employs `sosfiltfilt` (forward-backward filtering) to
      completely eliminate group delay distortion, preserving accurate phase alignment
      for SSVEP target detection.
    """
    # Step 1: Remove linear trend and DC bias
    detrended_data = signal.detrend(
        data=raw_data,
        axis=-1,
        type='linear',
        bp=0,
        overwrite_data=False
    )

    # Step 2: Define Butterworth SOS filter bounds
    lowcut_freq: float = 7.50   # Lower cutoff frequency (Hz)
    highcut_freq: float = 30.00 # Upper cutoff frequency (Hz)

    bandpass_filter = signal.butter(
        N=4,
        Wn=(lowcut_freq, highcut_freq),
        btype='bandpass',
        output='sos',
        fs=SAMPLING_RATE
    )

    # Step 3: Forward-backward zero-phase filtering
    filtered_data = signal.sosfiltfilt(
        sos=bandpass_filter,
        x=detrended_data,
        axis=-1,
        padtype='odd'
    )

    return filtered_data


def compute_power_spectrum(filtered_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate Power Spectral Density (PSD) using Welch's periodogram method.

    Parameters
    ----------
    filtered_data : np.ndarray
        Cleaned 1-D time-series EEG signal.

    Returns
    -------
    frequencies : np.ndarray
        Array of discrete frequency bins up to the Nyquist limit.
    power_spectrum : np.ndarray
        Power spectral density values (V^2/Hz) corresponding to frequency bins.

    Notes
    -----
    Welch's method divides the signal into overlapping segments, computes modified
    periodograms, and averages them. This reduces spectral variance significantly
    compared to standard FFT, making it ideal for extracting weak SSVEP peaks
    embedded in spontaneous cortical activity.
    """
    frequencies, power_spectrum = signal.welch(
        x=filtered_data,
        fs=SAMPLING_RATE,
        window='hann',
        nperseg=1024,
        noverlap=512,
        nfft=1024,
        detrend='constant',
        return_onesided=True,
        scaling='spectrum',
        axis=-1,
        average='mean'
    )

    return frequencies, power_spectrum


def extract_target_frequency(frequencies: np.ndarray, power_spectrum: np.ndarray) -> float:
    """Classify user intent by identifying the target frequency with peak spectral power.

    Parameters
    ----------
    frequencies : np.ndarray
        Array of computed frequency bins from PSD analysis.
    power_spectrum : np.ndarray
        Corresponding power spectral density values.

    Returns
    -------
    float
        The stimulus frequency (Hz) exhibiting the highest spectral power.

    Notes
    -----
    Uses exact bin index mapping via absolute difference minimization (`argmin`)
    to evaluate power at predefined target frequencies (8.57, 10.00, 12.00, 15.00 Hz),
    followed by `argmax` to classify the user's intent.
    """
    target_powers = []

    for target_freq in STIMULUS_FREQUENCIES:
        freq_differences = np.absolute(frequencies - target_freq)
        closest_index = freq_differences.argmin()
        target_powers.append(power_spectrum[closest_index])

    target_powers_arr = np.array(target_powers)
    target_power_index = np.argmax(target_powers_arr)

    return float(STIMULUS_FREQUENCIES[target_power_index])


def to_speech(detected_frequency: float) -> str:
    """Synthesize and stream audio output for the command mapped to a detected frequency.

    Parameters
    ----------
    detected_frequency : float
        The stimulus frequency (Hz) identified from SSVEP classification.

    Returns
    -------
    str
        The verbal command corresponding to the detected frequency.

    Notes
    -----
    Retrieves the textual command from `FREQUENCY_TO_COMMAND`, synthesizes 
    PCM audio using the Piper TTS model, converts raw audio chunks into 
    16-bit integer NumPy arrays, and plays them via `sounddevice`.
    """
    command_text = FREQUENCY_TO_COMMAND[detected_frequency]
    
    # Load Piper TTS model and synthesis configuration
    voice = PiperVoice.load(r"F:\Python\Project\en_US-lessac-medium.onnx")

    syn_config = SynthesisConfig(
        volume=0.5,
        length_scale=1.5,
        noise_scale=1.0,
        noise_w_scale=1.0,
        normalize_audio=True
    )

    for chunk in voice.synthesize(command_text, syn_config=syn_config):
        audio = np.frombuffer(
            chunk.audio_int16_bytes,
            dtype=np.int16,
            count=-1, 
            offset=0,
            like=None
        )

        # Stream and play the synthesized audio chunk
        sd.play(audio, chunk.sample_rate)
        sd.wait()

    return command_text


if __name__ == "__main__":
    main()
