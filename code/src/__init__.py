"""
Enoch Calendar Analysis Package

Computational validation of the 364-day Enochic calendar with 294-day
correction mechanism, demonstrating bounded oscillation through harmonic
coupling to Earth's axial precession.

Modules:
    calendar: Calendar generation and correction mechanism
    ephemeris: Solar position calculations using JPL DE441
    classifier: Harmonic alignment and amplitude prediction
    harmonic_analysis: FFT analysis and harmonic extraction
    monte_carlo: Monte Carlo robustness testing
    visualization: Plotting utilities for manuscript figures
"""

__version__ = "1.0.0"
__author__ = "Demetrius A. Bell"

# Package-level constants (used across all modules)
ENOCH_YEAR = 364  # Calendar structure (52 weeks × 7 days)
CORRECTION_PERIOD = 294  # Days between corrections (42 weeks)
PRECESSION_PERIOD = 25772  # Earth's axial precession (years)
EARTH_TROPICAL_YEAR = 365.24219  # Earth's tropical year (days)
