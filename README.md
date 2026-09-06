# Pulsar Single-Pulse fitting tool

A Python toolkit designed for radio pulsar single-pulse analysis, featuring Continuous Wavelet Transform (CWT) peak fitting, histogram-driven profile reconstruction, and modulation index calculation. This repository was
developed to process the publicly available MSPES data.

## Directory Structure

```text
├── modules/
│   ├── mod_index.py             # Bootstrap uncertainty & modulation index calculations
│   ├── pulse_fitting.py         # Multi-Gaussian profile fitting & CWT peak tracking
│   └── polarization_analysis.py # Baseline subtraction, linear polarization (L), PA, & plotting
├── batch_pulsar_analysis.py     # Batch processing runner for multiple pulsars (PDF compilation)
├── process_single_pulsar.py     # Detailed diagnostic analysis script for single pulsars
└── write_pulsestack.py          # Ingestion script to generate .npy pulse stacks and stats
```

# Dependencies
Python 3.x

numpy

scipy

matplotlib

psrqpy (for querying the ATNF pulsar catalog)

Install required packages via pip:
```text
pip install numpy scipy matplotlib psrqpy
```
# Usage

1. Generate a single pulse stack for a pulsar from MSPES format and store it in NumPy format:
   ```text
   python process_single_pulsar.py <directory> <psrname> <threshold>
   ```
2. Process single pulses of a pulsar using the NumPy pulse stack
   ```text
   python process_single_pulsar.py <directory> <psrname> <threshold>
   ```
3. Process all the pulsars in a folder containing NumPy-format single-pulse stacks:
   ```text
   python batch_pulsar_analysis.py <threshold> <directory>
   ```
