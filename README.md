# Resonant Order in the Enochic Calendar: Computational Modeling of Harmonic Stability and Precessional Coupling

**Author:** Demetrius A. Bell
**Affiliation:** Independent Researcher

**Paper:** Available on arXiv at [link once posted]

---

## Description

This repository contains the computational analysis code and supplementary materials for research investigating the 364-day Enochic calendar's correction mechanism and its mathematical relationship to Earth's axial precession. The analysis demonstrates that the 294-day correction period creates a **phase-locked oscillator system** that transforms unbounded linear calendar drift into bounded sinusoidal oscillation through harmonic coupling.

### Core Research Question

How does a simple 364-day calendar with a 294-day correction mechanism produce a self-correcting, bounded oscillation system? This work demonstrates that the calendar's correction period (294 days = 42 weeks) creates resonances with Earth's ~25,772-year precession cycle, causing the system to phase-lock and remain bounded perpetually (±7.6° oscillation over 70,000+ years).

### Key Findings

- **Bounded Oscillation**: Maximum drift of ±7.6° (~7.5 days) over 70,000+ years
- **Harmonic Resonance**: Three dominant periods (14,700 / 29,400 / 9,800 years) align with precession fractions
- **Structural Stability**: >99% of Monte Carlo trials maintain bounded behavior under parameter perturbation
- **Model Accuracy**: 7-harmonic FFT model explains 95.8% of variance (R² = 0.958)
- **Predictive Power**: 98.3% accuracy in amplitude prediction without calibration

---

## Repository Structure

```
enoch-calendar/
├── code/
│   ├── src/                           # Core Python modules
│   │   ├── classifier.py              # Parametric coupled oscillator model
│   │   ├── enoch.py                   # Calendar generation functions
│   │   ├── harmonic_analysis.py       # FFT-based harmonic analysis
│   │   ├── lunar.py                   # Lunar cycle calculations
│   │   ├── monte_carlo.py             # Monte Carlo robustness testing
│   │   ├── sensitivity.py             # 3-parameter sensitivity analysis
│   │   ├── publication_style.py       # Publication figure styling
│   │   ├── calc.py                    # Utility functions
│   │   └── enoch_config.py            # Configuration settings
│   ├── notebooks/                     # Jupyter analysis pipeline
│   │   ├── 01-parametric-sweep-global.ipynb
│   │   ├── 02-monte-carlo-robustness.ipynb
│   │   ├── 03-sensitivity-3parameter.ipynb
│   │   ├── 04-publication-figures.ipynb
│   │   ├── 05-fft-spectrum-catalog.ipynb
│   │   ├── 06-fft-model-selection.ipynb
│   │   ├── 07-fft-optimal-model.ipynb
│   │   ├── 08-294-resonance-analysis.ipynb
│   │   └── 09-visualization-dashboard.ipynb
│   ├── notebooks/outputs/             # Analysis results
│   │   ├── csvs/                      # Data tables
│   │   ├── figures/                   # Publication-ready figures
│   │   ├── tables/                    # LaTeX tables
│   │   └── models/                    # Serialized model files
│   └── requirements.txt               # Python dependencies
├── pyproject.toml                     # Project metadata
└── requirements.yml                   # Conda environment specification
```

---

## Computational Methods

### 1. Parametric Coupled Oscillator Model
Models the calendar as a coupled oscillator system where:
- **Calendar oscillator**: Annual drift = tropical year − 364 days
- **Correction oscillator**: Period-driven corrections every 294 days
- **Coupling mechanism**: Harmonic alignment with precession fractions

### 2. Fast Fourier Transform (FFT) Analysis
- Converts time-domain ecliptic longitude data to frequency domain
- Identifies dominant periodic components (3 primary + 4 secondary harmonics)
- Fixed-period curve fitting to prevent overfitting

### 3. Monte Carlo Robustness Testing
- 10,000 stochastic trials with parameter perturbations
- Tests structural stability under measurement uncertainty
- Validates >99% bounded oscillation persistence

### 4. Sensitivity Analysis
Three-parameter grid search examining:
- Vernal offset (0–364 days): Minimal impact (phase independence)
- Correction period (290–298 days): Moderate sensitivity
- Calendar year (360–368 days): High structural dependency

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Enoch Year | 364 days | Calendar year length (52 weeks) |
| Correction Period | 294 days | Interval between corrections (42 weeks) |
| Earth Tropical Year | 365.24219 days | Actual solar year |
| Precession Period | 25,772 years | Earth's axial precession cycle |
| Predicted Amplitude | 7.73° | Model prediction |
| Observed Amplitude | 7.6° | Actual maximum oscillation |

---

## Installation

### Prerequisites
- Python 3.10+
- Swiss Ephemeris data files (for astronomical calculations)

### Setup

```bash
# Clone the repository
git clone https://github.com/meetri/enoch-364day-calendar.git
cd enoch-calendar/code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure Swiss Ephemeris path (edit with your local path)
# Edit src/enoch_config.py
```

### Dependencies

**Core Scientific Stack:**
- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0

**Astronomical Calculations:**
- pyswisseph >= 2.10.3 (Swiss Ephemeris bindings)

**Visualization:**
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- jupyter >= 1.0.0

---

## Usage

### Running the Analysis Pipeline

```bash
cd code/notebooks

# Launch Jupyter
jupyter notebook

# Execute notebooks in order (01 through 09)
```

### Programmatic Execution

```bash
# Execute all notebooks
for nb in notebooks/*.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb"
done
```

### Using Individual Modules

```python
from src.classifier import classify_tropical_year, parameter_sweep
from src.enoch import enoch_calendar_frame
from src.harmonic_analysis import HarmonicAnalyzer

# Generate calendar data
calendar_df = enoch_calendar_frame(start_year=2000, num_years=100)

# Classify Earth's tropical year
result = classify_tropical_year(365.24219)
print(f"Classification: {result['classification']}")
print(f"Predicted Amplitude: {result['predicted_amplitude']:.2f}°")

# Perform harmonic analysis
analyzer = HarmonicAnalyzer(calendar_df['ecliptic_longitude'])
fft_results = analyzer.perform_fft()
```

---

## Output Files

Analysis outputs are saved to `code/notebooks/outputs/`:

| Directory | Contents |
|-----------|----------|
| `csvs/` | Data tables (parametric sweep, Monte Carlo, sensitivity results) |
| `figures/` | Publication-ready PDF/PNG figures |
| `tables/` | LaTeX-formatted tables for manuscript |
| `models/` | Serialized fitted models |

### Key Output Files

- `publication_summary_table.csv` - Model comparison statistics
- `optimal_model_params.csv` - 7-harmonic fitted parameters
- `precession_coupling_table.csv` - Harmonic alignment analysis
- `monte_carlo_robustness_summary.csv` - Robustness statistics
- `extrapolation_predictions.csv` - 70,000-year predictions

---

## Notebook Descriptions

| Notebook | Purpose |
|----------|---------|
| 01-parametric-sweep-global | Tests mechanism across tropical year parameter space (350–380 days) |
| 02-monte-carlo-robustness | 10,000 Monte Carlo trials for structural stability |
| 03-sensitivity-3parameter | Grid search over vernal offset, correction period, calendar year |
| 04-publication-figures | Generates all publication-quality figures and tables |
| 05-fft-spectrum-catalog | FFT analysis identifying dominant harmonic frequencies |
| 06-fft-model-selection | Model selection using AIC/BIC criteria |
| 07-fft-optimal-model | Fits optimal 7-harmonic model with 70,000-year extrapolation |
| 08-294-resonance-analysis | Detailed analysis of 294-day correction resonance |
| 09-visualization-dashboard | Interactive exploration dashboard |

---

## Citation

If you use this code, please cite:

```bibtex
@article{bell2025resonant,
  title={Resonant Order in the Enochic Calendar: Computational Modeling of Harmonic Stability and Precessional Coupling},
  author={Bell, Demetrius A.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## License

This code is provided as supplementary material for academic research. Please contact the author for licensing inquiries.

---

## Acknowledgments

This research utilizes the Swiss Ephemeris for high-precision astronomical calculations.
