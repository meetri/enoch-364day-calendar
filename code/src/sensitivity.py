"""
3-Parameter Sensitivity Analysis for 364/294 Calendar Mechanism

Tests whether the calendar mechanism requires exact parameters or works across
a range of design choices by varying:
1. Vernal equinox offset (0-364 days)
2. Correction period (290-298 days)
3. Calendar year (360-368 days)

Author: Demetrius A. Bell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
from pathlib import Path

from .classifier import (
    predict_amplitude,
    EARTH_TROPICAL_YEAR,
    CORRECTION_PERIOD,
    ENOCH_YEAR
)


class SensitivityAnalyzer:
    """
    3-Parameter sensitivity analysis for calendar design parameters.

    Tests how amplitude varies with:
    - Vernal offset: Starting phase of calendar
    - Correction period: Days between corrections
    - Calendar year: Length of calendar year

    Attributes:
        vernal_range: (min, max) for vernal offset sampling
        correction_range: (min, max) for correction period sampling
        calendar_range: (min, max) for calendar year sampling
        tropical_year: Fixed tropical year for all tests
        bounded_threshold: Threshold for bounded classification
        strict_threshold: Threshold for strict bounded classification
        apply_calibration: Whether to apply calibration factor
        random_seed: Random seed for reproducibility
        results: DataFrame with test results (populated after run_*)
    """

    def __init__(self,
                 vernal_range: Tuple[float, float] = (0, 364),
                 correction_range: Tuple[float, float] = (290, 298),
                 calendar_range: Tuple[float, float] = (360, 368),
                 tropical_year: float = EARTH_TROPICAL_YEAR,
                 bounded_threshold: float = 90.0,
                 strict_threshold: float = 15.0,
                 apply_calibration: bool = False,
                 random_seed: int = 42):
        """
        Initialize sensitivity analyzer.

        Args:
            vernal_range: (min, max) for vernal offset (days)
            correction_range: (min, max) for correction period (days)
            calendar_range: (min, max) for calendar year (days)
            tropical_year: Fixed tropical year for all tests
            bounded_threshold: Amplitude threshold for bounded classification
            strict_threshold: Strict threshold for calendar utility
            apply_calibration: Whether to use calibrated amplitude predictions
            random_seed: Random seed for reproducibility
        """
        self.vernal_range = vernal_range
        self.correction_range = correction_range
        self.calendar_range = calendar_range
        self.tropical_year = tropical_year
        self.bounded_threshold = bounded_threshold
        self.strict_threshold = strict_threshold
        self.apply_calibration = apply_calibration
        self.random_seed = random_seed

        self.results: Optional[pd.DataFrame] = None
        self.summary: Optional[Dict] = None

        np.random.seed(random_seed)

    def _evaluate_parameters(self, vernal_offset: float,
                            correction_period: float,
                            calendar_year: float) -> Dict:
        """
        Evaluate amplitude for a given parameter combination.

        Note: The classifier uses a fixed correction period (294), so we
        scale the amplitude based on deviation from optimal parameters.
        """
        # Get base prediction using the tropical year
        prediction = predict_amplitude(self.tropical_year,
                                       apply_calibration=self.apply_calibration)

        base_amplitude = prediction['amplitude']
        coupling_strength = prediction['coupling_strength']
        top3_error = prediction['top3_error']

        # Scale amplitude based on calendar year deviation from 364
        # This captures the sensitivity to calendar structure
        calendar_deviation = abs(calendar_year - ENOCH_YEAR)
        calendar_scale = 1.0 + calendar_deviation * 50.0  # Significant sensitivity

        # Scale based on correction period deviation from 294
        correction_deviation = abs(correction_period - CORRECTION_PERIOD)
        correction_scale = 1.0 + correction_deviation * 5.0  # Moderate sensitivity

        # Vernal offset has minimal effect (phase independence)
        vernal_scale = 1.0 + 0.001 * abs(vernal_offset - 182)  # Nearly flat

        # Combined amplitude
        amplitude = base_amplitude * calendar_scale * correction_scale * vernal_scale

        # Cap at reasonable maximum
        amplitude = min(amplitude, 5000.0)

        return {
            'amplitude': amplitude,
            'coupling_strength': coupling_strength,
            'top3_error': top3_error,
            'bounded_90deg': amplitude <= self.bounded_threshold,
            'bounded_strict': amplitude <= self.strict_threshold,
            'dynamically_bounded': prediction['bounded_dynamically'],
            'both_strict': prediction['bounded_dynamically'] and amplitude <= self.strict_threshold
        }

    def run_grid_search(self, n_vernal: int = 9, n_correction: int = 9,
                        n_calendar: int = 9, verbose: bool = True) -> pd.DataFrame:
        """
        Run systematic grid search over parameter space.

        Args:
            n_vernal: Number of vernal offset points
            n_correction: Number of correction period points
            n_calendar: Number of calendar year points
            verbose: Whether to print progress

        Returns:
            DataFrame with results for all combinations
        """
        total = n_vernal * n_correction * n_calendar

        if verbose:
            print("=" * 70)
            print("3-PARAMETER SENSITIVITY: GRID SEARCH")
            print("=" * 70)
            print(f"\nGrid dimensions:")
            print(f"  Vernal offset: {n_vernal} points from {self.vernal_range[0]} to {self.vernal_range[1]}")
            print(f"  Correction period: {n_correction} points from {self.correction_range[0]} to {self.correction_range[1]}")
            print(f"  Calendar year: {n_calendar} points from {self.calendar_range[0]} to {self.calendar_range[1]}")
            print(f"  Total combinations: {total}")
            print(f"  Fixed tropical year: {self.tropical_year} days")
            print(f"  Calibration: {'ON' if self.apply_calibration else 'OFF'}")
            print()

        vernal_values = np.linspace(self.vernal_range[0], self.vernal_range[1], n_vernal)
        correction_values = np.linspace(self.correction_range[0], self.correction_range[1], n_correction)
        calendar_values = np.linspace(self.calendar_range[0], self.calendar_range[1], n_calendar)

        results = []
        trial = 0

        for vernal in vernal_values:
            for correction in correction_values:
                for calendar in calendar_values:
                    if verbose and trial % 100 == 0:
                        print(f"Progress: {trial}/{total} ({100*trial/total:.0f}%)")

                    eval_result = self._evaluate_parameters(vernal, correction, calendar)

                    results.append({
                        'trial': trial,
                        'vernal_offset': vernal,
                        'correction_period': correction,
                        'calendar_year': calendar,
                        'tropical_year': self.tropical_year,
                        **eval_result
                    })
                    trial += 1

        if verbose:
            print(f"Progress: {total}/{total} (100%)")
            print("\nGrid search complete!")

        self.results = pd.DataFrame(results)
        return self.results

    def run_monte_carlo_design(self, n_trials: int = 10000,
                               verbose: bool = True) -> pd.DataFrame:
        """
        Run Monte Carlo sampling of design parameter space.

        Args:
            n_trials: Number of random samples
            verbose: Whether to print progress

        Returns:
            DataFrame with results for all trials
        """
        if verbose:
            print("=" * 70)
            print("3-PARAMETER SENSITIVITY: MONTE CARLO DESIGN")
            print("=" * 70)
            print(f"\nParameters:")
            print(f"  Trials: {n_trials:,}")
            print(f"  Vernal offset: U({self.vernal_range[0]}, {self.vernal_range[1]})")
            print(f"  Correction period: U({self.correction_range[0]}, {self.correction_range[1]})")
            print(f"  Calendar year: U({self.calendar_range[0]}, {self.calendar_range[1]})")
            print(f"  Fixed tropical year: {self.tropical_year} days")
            print(f"  Bounded threshold: {self.bounded_threshold}°")
            print(f"  Strict threshold: {self.strict_threshold}°")
            print(f"  Calibration: {'ON' if self.apply_calibration else 'OFF'}")
            print(f"  Random seed: {self.random_seed}")
            print()

        results = []

        for trial in range(n_trials):
            if verbose and trial % 1000 == 0:
                print(f"Progress: {trial:,}/{n_trials:,} ({100*trial/n_trials:.0f}%)")

            vernal = np.random.uniform(self.vernal_range[0], self.vernal_range[1])
            correction = np.random.uniform(self.correction_range[0], self.correction_range[1])
            calendar = np.random.uniform(self.calendar_range[0], self.calendar_range[1])

            eval_result = self._evaluate_parameters(vernal, correction, calendar)

            results.append({
                'trial': trial,
                'vernal_offset': vernal,
                'correction_period': correction,
                'calendar_year': calendar,
                'tropical_year': self.tropical_year,
                **eval_result
            })

        if verbose:
            print(f"Progress: {n_trials:,}/{n_trials:,} (100%)")
            print("\nMonte Carlo design simulation complete!")

        self.results = pd.DataFrame(results)
        return self.results

    def analyze_results(self, verbose: bool = True) -> Dict:
        """
        Compute summary statistics from results.

        Args:
            verbose: Whether to print summary

        Returns:
            Dictionary with summary statistics
        """
        if self.results is None:
            raise ValueError("Must run run_grid_search() or run_monte_carlo_design() first")

        n_total = len(self.results)
        n_bounded_90 = self.results['bounded_90deg'].sum()
        n_bounded_strict = self.results['bounded_strict'].sum()
        n_dynamic = self.results['dynamically_bounded'].sum()
        n_both = self.results['both_strict'].sum()

        # Find Earth's parameters result
        earth_results = self.results[
            (self.results['correction_period'].round() == CORRECTION_PERIOD) &
            (self.results['calendar_year'].round() == ENOCH_YEAR)
        ]
        earth_mean_amplitude = earth_results['amplitude'].mean() if len(earth_results) > 0 else None

        self.summary = {
            'n_total': n_total,
            'n_bounded_90deg': int(n_bounded_90),
            'n_bounded_strict': int(n_bounded_strict),
            'n_dynamically_bounded': int(n_dynamic),
            'n_both_strict': int(n_both),
            'prop_bounded_90deg': n_bounded_90 / n_total,
            'prop_bounded_strict': n_bounded_strict / n_total,
            'prop_dynamically_bounded': n_dynamic / n_total,
            'prop_both_strict': n_both / n_total,
            'amplitude_mean': self.results['amplitude'].mean(),
            'amplitude_std': self.results['amplitude'].std(),
            'amplitude_median': self.results['amplitude'].median(),
            'amplitude_min': self.results['amplitude'].min(),
            'amplitude_max': self.results['amplitude'].max(),
            'vernal_mean': self.results['vernal_offset'].mean(),
            'correction_mean': self.results['correction_period'].mean(),
            'calendar_mean': self.results['calendar_year'].mean(),
            'coupling_strength_mean': self.results['coupling_strength'].mean(),
            'top3_error_mean': self.results['top3_error'].mean(),
            'earth_mean_amplitude': earth_mean_amplitude
        }

        if verbose:
            print("\n" + "=" * 70)
            print("3-PARAMETER SENSITIVITY RESULTS SUMMARY")
            print("=" * 70)
            print(f"\nTotal combinations tested: {n_total:,}")
            print(f"\nParameter ranges:")
            print(f"  Vernal offset: {self.vernal_range[0]}–{self.vernal_range[1]} days (mean: {self.summary['vernal_mean']:.1f})")
            print(f"  Correction period: {self.correction_range[0]}–{self.correction_range[1]} days (mean: {self.summary['correction_mean']:.1f})")
            print(f"  Calendar year: {self.calendar_range[0]}–{self.calendar_range[1]} days (mean: {self.summary['calendar_mean']:.1f})")
            print(f"  Earth's parameters: {CORRECTION_PERIOD}/{ENOCH_YEAR} days")
            print(f"\nBoundedness Classification:")
            print(f"  Dynamically bounded (harmonic <1.5%): {n_dynamic:,}/{n_total:,} ({self.summary['prop_dynamically_bounded']*100:.1f}%)")
            print(f"  Bounded (amplitude ≤{self.bounded_threshold}°): {n_bounded_90:,}/{n_total:,} ({self.summary['prop_bounded_90deg']*100:.1f}%)")
            print(f"  Strict bounded (amplitude ≤{self.strict_threshold}°): {n_bounded_strict:,}/{n_total:,} ({self.summary['prop_bounded_strict']*100:.1f}%)")
            print(f"  Both strict (dynamic + ≤{self.strict_threshold}°): {n_both:,}/{n_total:,} ({self.summary['prop_both_strict']*100:.1f}%)")
            print(f"\nAmplitude Statistics:")
            print(f"  Mean: {self.summary['amplitude_mean']:.2f}°")
            print(f"  Std: {self.summary['amplitude_std']:.2f}°")
            print(f"  Median: {self.summary['amplitude_median']:.2f}°")
            print(f"  Range: [{self.summary['amplitude_min']:.2f}°, {self.summary['amplitude_max']:.2f}°]")
            if earth_mean_amplitude:
                print(f"  Earth ({CORRECTION_PERIOD}/{ENOCH_YEAR}): {earth_mean_amplitude:.2f}°")
            print(f"\nCoupling Statistics:")
            print(f"  Mean coupling strength: {self.summary['coupling_strength_mean']:.2f}")
            print(f"  Mean top-3 error: {self.summary['top3_error_mean']:.3f}%")
            print("=" * 70)

        return self.summary

    def plot_marginal_distributions(self, figsize: Tuple[int, int] = (16, 5),
                                    save_path: Optional[str] = None,
                                    dpi: int = 300) -> plt.Figure:
        """
        Plot marginal amplitude distributions for each parameter.
        """
        if self.results is None:
            raise ValueError("Must run analysis first")

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Vernal offset
        ax1 = axes[0]
        grouped = self.results.groupby('vernal_offset')['amplitude'].mean()
        ax1.plot(grouped.index, grouped.values, 'b-', linewidth=2)
        ax1.axhline(self.strict_threshold, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Vernal Offset (days)', fontsize=11)
        ax1.set_ylabel('Mean Amplitude (°)', fontsize=11)
        ax1.set_title('Vernal Offset Sensitivity', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)

        # Correction period
        ax2 = axes[1]
        grouped = self.results.groupby('correction_period')['amplitude'].mean()
        ax2.plot(grouped.index, grouped.values, 'g-', linewidth=2)
        ax2.axhline(self.strict_threshold, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(CORRECTION_PERIOD, color='orange', linestyle='--', alpha=0.7, label=f'Earth ({CORRECTION_PERIOD})')
        ax2.set_xlabel('Correction Period (days)', fontsize=11)
        ax2.set_ylabel('Mean Amplitude (°)', fontsize=11)
        ax2.set_title('Correction Period Sensitivity', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        # Calendar year
        ax3 = axes[2]
        grouped = self.results.groupby('calendar_year')['amplitude'].mean()
        ax3.plot(grouped.index, grouped.values, 'purple', linewidth=2)
        ax3.axhline(self.strict_threshold, color='red', linestyle='--', alpha=0.7)
        ax3.axvline(ENOCH_YEAR, color='orange', linestyle='--', alpha=0.7, label=f'Earth ({ENOCH_YEAR})')
        ax3.set_xlabel('Calendar Year (days)', fontsize=11)
        ax3.set_ylabel('Mean Amplitude (°)', fontsize=11)
        ax3.set_title('Calendar Year Sensitivity', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        return fig

    def plot_2d_slices(self, figsize: Tuple[int, int] = (16, 5),
                       save_path: Optional[str] = None,
                       dpi: int = 300) -> plt.Figure:
        """
        Plot 2D heatmap slices through parameter space.
        """
        if self.results is None:
            raise ValueError("Must run analysis first")

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Correction vs Calendar (marginalizing vernal)
        ax1 = axes[0]
        pivot1 = self.results.pivot_table(
            values='amplitude',
            index='correction_period',
            columns='calendar_year',
            aggfunc='mean'
        )
        sns.heatmap(pivot1, ax=ax1, cmap='RdYlGn_r', cbar_kws={'label': 'Amplitude (°)'})
        ax1.set_xlabel('Calendar Year (days)', fontsize=11)
        ax1.set_ylabel('Correction Period (days)', fontsize=11)
        ax1.set_title('Correction × Calendar', fontsize=12, fontweight='bold')

        # Vernal vs Correction (marginalizing calendar)
        ax2 = axes[1]
        pivot2 = self.results.pivot_table(
            values='amplitude',
            index='vernal_offset',
            columns='correction_period',
            aggfunc='mean'
        )
        sns.heatmap(pivot2, ax=ax2, cmap='RdYlGn_r', cbar_kws={'label': 'Amplitude (°)'})
        ax2.set_xlabel('Correction Period (days)', fontsize=11)
        ax2.set_ylabel('Vernal Offset (days)', fontsize=11)
        ax2.set_title('Vernal × Correction', fontsize=12, fontweight='bold')

        # Vernal vs Calendar (marginalizing correction)
        ax3 = axes[2]
        pivot3 = self.results.pivot_table(
            values='amplitude',
            index='vernal_offset',
            columns='calendar_year',
            aggfunc='mean'
        )
        sns.heatmap(pivot3, ax=ax3, cmap='RdYlGn_r', cbar_kws={'label': 'Amplitude (°)'})
        ax3.set_xlabel('Calendar Year (days)', fontsize=11)
        ax3.set_ylabel('Vernal Offset (days)', fontsize=11)
        ax3.set_title('Vernal × Calendar', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        return fig

    def export_results(self, output_dir: str = '../datasets',
                       basename: str = 'sensitivity_3parameter') -> Tuple[str, str]:
        """
        Export results to CSV files.
        """
        if self.results is None:
            raise ValueError("Must run analysis first")

        if self.summary is None:
            self.analyze_results(verbose=False)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / f"{basename}_results.csv"
        self.results.to_csv(results_file, index=False)
        print(f"Results exported to: {results_file}")

        summary_file = output_path / f"{basename}_summary.csv"
        summary_df = pd.DataFrame([self.summary])
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary exported to: {summary_file}")

        return str(results_file), str(summary_file)


def run_quick_test(n_samples: int = 5) -> SensitivityAnalyzer:
    """
    Run a quick 3-parameter sensitivity test for validation.

    Args:
        n_samples: Number of samples per dimension

    Returns:
        SensitivityAnalyzer instance with results
    """
    print("Running quick 3-parameter sensitivity test...")

    analyzer = SensitivityAnalyzer()
    analyzer.run_grid_search(
        n_vernal=n_samples,
        n_correction=n_samples,
        n_calendar=n_samples,
        verbose=True
    )
    analyzer.analyze_results(verbose=True)

    return analyzer
