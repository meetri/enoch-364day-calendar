# Enoch Calendar Analysis - Supplementary Code

Computational analysis supporting the manuscript: *"Phase-Locking in the 364-Day Calendar: A Parametric Coupled Oscillator Model"*

## Overview

This repository contains Python code for analyzing the 364-day calendar's correction mechanism and its harmonic resonance with Earth's precession cycle. The analysis demonstrates that the 294-day correction period creates bounded oscillation through phase-locking to precession harmonics.

## Repository Structure

```
code/
├── src/                    # Python modules
│   ├── classifier.py       # Parametric coupled oscillator model
│   ├── enoch.py            # Calendar generation functions
│   ├── harmonic_analysis.py # FFT-based harmonic analysis
│   ├── lunar.py            # Lunar cycle calculations
│   ├── monte_carlo.py      # Monte Carlo robustness testing
│   ├── sensitivity.py      # 3-parameter sensitivity analysis
│   ├── publication_style.py # Publication-ready figure styling
│   └── calc.py             # Utility functions
├── notebooks/              # Jupyter notebooks (analysis pipeline)
│   ├── 01-parametric-sweep-global.ipynb
│   ├── 02-monte-carlo-robustness.ipynb
│   ├── 03-sensitivity-3parameter.ipynb
│   ├── 04-publication-figures.ipynb
│   ├── 05-fft-spectrum-catalog.ipynb
│   ├── 06-fft-model-selection.ipynb
│   ├── 07-fft-optimal-model.ipynb
│   ├── 08-294-resonance-analysis.ipynb
│   └── 09-visualization-dashboard.ipynb
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Swiss Ephemeris data files (for astronomical calculations)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd enoch-calendar/code
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure Swiss Ephemeris path:
   
   Edit `src/enoch_config.py` to set the path to your Swiss Ephemeris data files:
   ```python
   SWISS_EPH_PATH = "/path/to/your/ephemeris/files"
   ```

## Running the Analysis

### Using Jupyter

Start Jupyter and open notebooks in sequence:
```bash
cd notebooks
jupyter notebook
```

### Running All Notebooks Programmatically

To execute all notebooks and verify reproducibility:
```bash
cd notebooks
for nb in *.ipynb; do
    echo "Running $nb..."
    jupyter nbconvert --to notebook --execute "$nb" --output "/tmp/test_${nb}"
done
```

## Notebooks Description

| Notebook | Description |
|----------|-------------|
| 01-parametric-sweep-global | Tests calendar mechanism across tropical year parameter space |
| 02-monte-carlo-robustness | Monte Carlo simulation testing structural stability |
| 03-sensitivity-3parameter | Sensitivity analysis for vernal offset, correction period, calendar year |
| 04-publication-figures | Generates publication-ready figures |
| 05-fft-spectrum-catalog | FFT analysis to identify dominant harmonic frequencies |
| 06-fft-model-selection | Model selection using AIC/BIC criteria |
| 07-fft-optimal-model | Fits optimal multi-harmonic model |
| 08-294-resonance-analysis | Analyzes the 294-day correction period resonance |
| 09-visualization-dashboard | Interactive visualization dashboard |

## Output Files

All outputs are written to `notebooks/outputs/`:
- `csvs/` - Data tables and analysis results
- `figures/` - Publication-ready figures (PDF and PNG)
- `tables/` - LaTeX tables for manuscript
- `models/` - Serialized model files

## Key Results

The analysis demonstrates:
1. **Bounded Oscillation**: The 294-day correction transforms linear drift into bounded sinusoidal oscillation
2. **Precession Coupling**: Three dominant harmonics (13,965 / 27,930 / 9,310 years) align with precession fractions
3. **Structural Stability**: Mechanism works across parameter perturbations (Monte Carlo: >99% bounded)
4. **Phase Independence**: Vernal equinox offset has minimal effect on amplitude

## License

This code is provided as supplementary material for academic research.

## Citation

If you use this code, please cite the associated manuscript.
