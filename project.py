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
from sklearn.cross_decomposition import CCA
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

    raw_data= read_data(file_path)
    filtered_data = filter_signal(raw_data)
    stimulus_waves = hertz_to_waves(STIMULUS_FREQUENCIES, len(filtered_data))
    X_c, Y_c = cca(stimulus_waves, filtered_data)
    target_frequency = detect_frequency(X_c, Y_c)
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


def hertz_to_waves(stimulus_frequencies: np.ndarray, data_length: int) -> list:
    """Generate sinusoidal reference signals for canonical correlation analysis.

    Parameters
    ----------
    stimulus_frequencies : np.ndarray
        1-D array of target SSVEP stimulus frequencies (Hz).
    data_length : int
        Total number of samples in the filtered EEG recording, used to match temporal duration.

    Returns
    -------
    list of np.ndarray
        A list containing 1-D sinusoidal reference wave arrays for each stimulus frequency.

    Notes
    -----
    - **Time Synchronization**: Constructs a high-resolution time array scaled dynamically
      to the exact duration of the EEG signal ($\text{data\_length} / \text{SAMPLING\_RATE}$).
    - **Reference Modeling**: Generates pure sine waves ($\sin(2\pi f t)$) to serve as temporal
      template references for frequency-specific phase alignment in CCA.
    """
    stimulus_waves = []
    time_array = np.linspace(
        0,
        data_length / SAMPLING_RATE,
        num=data_length,
        endpoint=True,
        retstep=False
    )
    for freq in stimulus_frequencies:
        wave = np.sin(2 * np.pi * freq * time_array)  
        stimulus_waves.append(wave)

    return stimulus_waves


def cca(stimulus_waves: list, filtered_data: np.ndarray) -> Tuple[list, list]:
    """Execute multi-template Canonical Correlation Analysis (CCA) for frequency recognition.

    Parameters
    ----------
    stimulus_waves : list of np.ndarray
        List of 1-D sinusoidal reference waves for each target frequency.
    filtered_data : np.ndarray
        Pre-processed 1-D EEG voltage signal.

    Returns
    -------
    Tuple[list, list]
        - `Xs_C`: List of transformed canonical variables corresponding to reference waves.
        - `Ys_C`: List of transformed canonical variables corresponding to the EEG signal.

    Notes
    -----
    - **Dimensionality Reshaping**: Converts 1-D vectors into 2D column matrices (`(-1, 1)`)
      required by scikit-learn estimator interfaces.
    - **Independent Projections**: Iterates through each stimulus reference independently,
      fitting a 1-component CCA model to maximize linear correlation between the reference
      template and the multichannel/single-channel EEG data structure.
    """
    reshaped_stimulus_waves = []
    for stimulus_wave in stimulus_waves:
        reshaped_stimulus_wave = np.reshape(stimulus_wave, (-1, 1))
        reshaped_stimulus_waves.append(reshaped_stimulus_wave)
    reshaped_filtered_data = np.reshape(filtered_data, (-1, 1))
    
    Xs_C, Ys_C = [], []
    for wave in reshaped_stimulus_waves:
        cca_model = CCA(n_components=1)
        cca_model.fit(wave, reshaped_filtered_data)
        X_c, Y_c = cca_model.transform(wave, reshaped_filtered_data)
        Xs_C.append(X_c)
        Ys_C.append(Y_c)
        
    return Xs_C, Ys_C


def detect_frequency(X_c: list, Y_c: list) -> float:
    """Classify the target SSVEP frequency by evaluating maximal canonical correlation coefficients.

    Parameters
    ----------
    X_c : list of np.ndarray
        List of transformed reference canonical variables.
    Y_c : list of np.ndarray
        List of transformed EEG canonical variables.

    Returns
    -------
    float
        The classified target frequency corresponding to the maximum correlation score.

    Notes
    -----
    - **Vector Flattening**: Employs `.ravel()` to safely cast 2D transformed components back
      into 1-D vectors for Pearson correlation matrix extraction via `np.corrcoef`.
    - **Decision Rule**: Uses `np.argmax` to find the reference frequency index yielding the
      strongest linear association with the recorded EEG segment.
    """
    correlation_coefficients = []
    for x, y in zip(X_c, Y_c):
        correlation = np.corrcoef(x.ravel(), y.ravel())[0, 1]
        correlation_coefficients.append(correlation)
        
    index_of_max_correlation = np.argmax(correlation_coefficients, axis=None, out=None, keepdims=False)
    detected_frequency = STIMULUS_FREQUENCIES[index_of_max_correlation]
    
    return detected_frequency


def to_speech(detected_frequency: float) -> str:
    """Synthesize real-time offline text-to-speech output matching the decoded BCI command.

    Parameters
    ----------
    detected_frequency : float
        The classified frequency mapped to an assistive communication command.

    Returns
    -------
    str
        The textual command associated with the target frequency.

    Notes
    -----
    - **Engine Integration**: Utilizes the offline ONNX-based `PiperVoice` neural synthesizer
      for low-latency voice rendering.
    - **Acoustic Profiling**: Configures custom synthesis parameters (volume scale, length scale,
      and noise variance) for clear, paced auditory articulation via `sounddevice`.
    """
    command_text = FREQUENCY_TO_COMMAND[detected_frequency]
    
    voice = PiperVoice.load(r"F:\Python\Project\en_US-lessac-medium.onnx")

    syn_config = SynthesisConfig(
        volume=0.5,  # half as loud
        length_scale=1.5,  # slower speech rate
        noise_scale=1.0,  # more audio variation
        noise_w_scale=1.0,  # more speaking variation
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

        sd.play(audio, chunk.sample_rate)
        sd.wait()

    return command_text


if __name__ == "__main__":
    main()
