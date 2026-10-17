"""Unit tests for the eeg signal processing pipeline (cs50p final project).

This module contains unit tests executed with pytest to verify file validation,
data ingestion, and digital signal processing (dsp) filtering components.
"""

import numpy as np
import pytest
from project import (
    filter_signal,
    read_data,
    validate_file,
)


def test_validate_file():
    """Test file path existence checks and non-numeric content handling."""
    # Verify SystemExit when target file does not exist
    with pytest.raises(SystemExit) as exc_info:
        validate_file("anonymous_data.csv")
    assert exc_info.value.code == "Error: File does not exist. Please check the file path."

    # Verify SystemExit when file contains alphanumeric or text headers
    sample_file = "alphanumeric_data.txt"
    with open(sample_file, "w") as f:
        f.write("F1 566 349")

    with pytest.raises(SystemExit) as exc_info:
        validate_file(sample_file)
    assert exc_info.value.code == "Error: Non-numeric content detected. Please remove text headers/labels."


def test_read_data():
    """Test reading raw voltage data from valid and empty text files."""
    # Verify ValueError when attempting to parse an empty file
    empty_file = "empty_file.txt"
    with open(empty_file, "w"):
        pass

    with pytest.raises(ValueError):
        read_data(empty_file)

    # Verify correct array parsing from valid numeric file input
    sample_voltages = [566, 581, 558, 599, 599, 525, 547, 549]
    data_file = "data.txt"
    with open(data_file, "w") as f:
        for val in sample_voltages:
            f.write(f"{val}\n")

    assert np.array_equal(read_data(data_file), np.array(sample_voltages))


def test_filter_signal():
    """Test bandpass signal filtering using synthetic multi-frequency wave."""
    # Generate 1-second discrete time axis at 256 Hz sampling rate
    time_array = np.linspace(0, 1, num=256, endpoint=True, retstep=False)

    # Define constituent frequencies (in Hz)
    low_freq = 3      # Low-frequency baseline artifact
    target_freq = 12  # Target brainwave signal (Alpha band)
    high_freq = 50    # High-frequency power-line noise

    # Synthesize individual sinusoidal components
    low_wave = np.sin(2 * np.pi * low_freq * time_array)
    target_wave = np.sin(2 * np.pi * target_freq * time_array)
    high_wave = np.sin(2 * np.pi * high_freq * time_array)

    # Construct raw composite signal
    raw_wave = low_wave + target_wave + high_wave

    # Apply digital signal processing filter
    filtered_signal = filter_signal(raw_wave)

    # Perform FFT transformation and extract magnitude spectrum
    fft_result = np.fft.rfft(filtered_signal)
    absolute_spectrum = np.abs(fft_result)
    highest_power_index = np.argmax(absolute_spectrum)

    # Generate corresponding discrete frequency bins
    virtual_frequencies = np.fft.rfftfreq(n=256, d=1 / 256)

    # Extract dominant peak frequency and verify match against target
    dominant_freq = virtual_frequencies[highest_power_index]
    assert np.isclose(target_freq, dominant_freq)
