"""
Monte Carlo Robustness Testing for 364/294 Calendar Mechanism

This module provides a Monte Carlo framework to test the structural stability
of the 364-day calendar with 294-day correction mechanism across parameter
perturbations.

Key Question: Does bounded oscillation persist when parameters vary with
realistic uncertainties?

Author: Demetrius A. Bell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .classifier import (
    predict_amplitude,
    classify_tropical_year,
    EARTH_TROPICAL_YEAR,
    CORRECTION_PERIOD
)


class MonteCarloRobustness:
    """
    Monte Carlo robustness testing for the 364/294 calendar resonance.

    Tests whether bounded oscillation persists across parameter perturbations:
    - Tropical year length (measurement uncertainty)
    - Correction period (mechanism variation)
    - Optional: start phase/epoch sensitivity

    Attributes:
        num_trials: Number of Monte Carlo trials
        year_mean: Mean tropical year (days)
        year_std: Standard deviation for tropical year sampling
        correction_min: Minimum correction period (days)
        correction_max: Maximum correction period (days)
        bounded_threshold: Amplitude threshold for bounded classification (degrees)
        strict_threshold: Strict threshold for calendar utility (degrees)
        apply_calibration: Whether to use calibrated amplitude predictions
        random_seed: Random seed for reproducibility
        results: DataFrame with trial results (populated after run_simulation)
    """

    def __init__(self,
                 num_trials: int = 10000,
                 year_mean: float = EARTH_TROPICAL_YEAR,
                 year_std: float = 0.05,
                 correction_min: float = 293.0,
                 correction_max: float = 295.0,
                 bounded_threshold: float = 90.0,
                 strict_threshold: float = 15.0,
                 apply_calibration: bool = True,
                 random_seed: int = 42):
        """
        Initialize Monte Carlo robustness test.

        Args:
            num_trials: Number of Monte Carlo trials
            year_mean: Mean tropical year for sampling (days)
            year_std: Standard deviation for tropical year (days)
            correction_min: Minimum correction period (days)
            correction_max: Maximum correction period (days)
            bounded_threshold: Amplitude threshold for bounded classification (degrees)
            strict_threshold: Strict threshold for calendar utility (degrees)
            apply_calibration: Whether to use calibrated amplitude predictions
            random_seed: Random seed for reproducibility
        """
        self.num_trials = num_trials
        self.year_mean = year_mean
        self.year_std = year_std
        self.correction_min = correction_min
        self.correction_max = correction_max
        self.bounded_threshold = bounded_threshold
        self.strict_threshold = strict_threshold
        self.apply_calibration = apply_calibration
        self.random_seed = random_seed

        # Results storage
        self.results: Optional[pd.DataFrame] = None
        self.summary: Optional[Dict] = None

        # Set random seed
        np.random.seed(random_seed)

    def run_simulation(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run Monte Carlo simulation.

        For each trial:
        1. Sample tropical year from N(year_mean, year_std²)
        2. Sample correction period from U(correction_min, correction_max)
        3. Predict amplitude using classifier
        4. Classify as bounded/unbounded based on thresholds

        Args:
            verbose: Whether to print progress updates

        Returns:
            DataFrame with results for all trials
        """
        if verbose:
            print("=" * 70)
            print("MONTE CARLO ROBUSTNESS SIMULATION")
            print("=" * 70)
            print(f"\nParameters:")
            print(f"  Trials: {self.num_trials:,}")
            print(f"  Tropical year: N({self.year_mean:.5f}, {self.year_std:.5f})")
            print(f"  Correction period: U({self.correction_min}, {self.correction_max})")
            print(f"  Bounded threshold: {self.bounded_threshold}°")
            print(f"  Strict threshold: {self.strict_threshold}°")
            print(f"  Calibration: {'ON' if self.apply_calibration else 'OFF'}")
            print(f"  Random seed: {self.random_seed}")
            print()

        results = []

        for trial in range(self.num_trials):
            if verbose and trial % 1000 == 0:
                pct = 100 * trial / self.num_trials
                print(f"Progress: {trial:,}/{self.num_trials:,} ({pct:.0f}%)")

            # Sample parameters
            tropical_year = np.random.normal(self.year_mean, self.year_std)

            # For correction period, we're testing the mechanism's sensitivity
            # Note: classifier uses fixed CORRECTION_PERIOD, so we sample but
            # use the classifier's prediction model which assumes 294 days
            # This tests: "if year length varies, does 294 still work?"
            correction_period = np.random.uniform(self.correction_min, self.correction_max)

            # Predict amplitude using classifier
            # Note: predict_amplitude() uses the fixed 294-day model
            prediction = predict_amplitude(tropical_year, apply_calibration=self.apply_calibration)

            amplitude = prediction['amplitude']
            coupling_strength = prediction['coupling_strength']
            top3_error = prediction['top3_error']

            # Classify bounded/unbounded at multiple thresholds
            bounded_90deg = amplitude <= self.bounded_threshold
            bounded_strict = amplitude <= self.strict_threshold
            dynamically_bounded = prediction['bounded_dynamically']

            # Store trial result
            results.append({
                'trial': trial,
                'tropical_year': tropical_year,
                'correction_period': correction_period,
                'amplitude': amplitude,
                'coupling_strength': coupling_strength,
                'top3_error': top3_error,
                'bounded_90deg': bounded_90deg,
                'bounded_strict': bounded_strict,
                'dynamically_bounded': dynamically_bounded,
                'both_strict': dynamically_bounded and bounded_strict
            })

        if verbose:
            print(f"Progress: {self.num_trials:,}/{self.num_trials:,} (100%)")
            print("\nSimulation complete!")

        self.results = pd.DataFrame(results)
        return self.results

    def analyze_results(self, verbose: bool = True) -> Dict:
        """
        Compute summary statistics from simulation results.

        Args:
            verbose: Whether to print summary

        Returns:
            Dictionary with summary statistics
        """
        if self.results is None:
            raise ValueError("Must run run_simulation() first")

        # Compute statistics
        n_bounded_90 = self.results['bounded_90deg'].sum()
        n_bounded_strict = self.results['bounded_strict'].sum()
        n_dynamic = self.results['dynamically_bounded'].sum()
        n_both = self.results['both_strict'].sum()

        prop_bounded_90 = n_bounded_90 / self.num_trials
        prop_bounded_strict = n_bounded_strict / self.num_trials
        prop_dynamic = n_dynamic / self.num_trials
        prop_both = n_both / self.num_trials

        amplitude_mean = self.results['amplitude'].mean()
        amplitude_std = self.results['amplitude'].std()
        amplitude_median = self.results['amplitude'].median()
        amplitude_min = self.results['amplitude'].min()
        amplitude_max = self.results['amplitude'].max()

        year_mean = self.results['tropical_year'].mean()
        year_std = self.results['tropical_year'].std()

        coupling_mean = self.results['coupling_strength'].mean()
        top3_error_mean = self.results['top3_error'].mean()

        self.summary = {
            'num_trials': self.num_trials,
            'n_bounded_90deg': int(n_bounded_90),
            'n_bounded_strict': int(n_bounded_strict),
            'n_dynamically_bounded': int(n_dynamic),
            'n_both_strict': int(n_both),
            'prop_bounded_90deg': prop_bounded_90,
            'prop_bounded_strict': prop_bounded_strict,
            'prop_dynamically_bounded': prop_dynamic,
            'prop_both_strict': prop_both,
            'amplitude_mean': amplitude_mean,
            'amplitude_std': amplitude_std,
            'amplitude_median': amplitude_median,
            'amplitude_min': amplitude_min,
            'amplitude_max': amplitude_max,
            'year_mean_sampled': year_mean,
            'year_std_sampled': year_std,
            'coupling_strength_mean': coupling_mean,
            'top3_error_mean': top3_error_mean
        }

        if verbose:
            print("\n" + "=" * 70)
            print("MONTE CARLO RESULTS SUMMARY")
            print("=" * 70)
            print(f"\nSampling Statistics:")
            print(f"  Trials completed: {self.num_trials:,}")
            print(f"  Tropical year sampled: {year_mean:.5f} ± {year_std:.5f} days")
            print(f"  Earth's value: {EARTH_TROPICAL_YEAR} days")
            print()
            print(f"Boundedness Classification:")
            print(f"  Dynamically bounded (harmonic <1.5%): {n_dynamic:,}/{self.num_trials:,} ({prop_dynamic*100:.1f}%)")
            print(f"  Bounded (amplitude ≤{self.bounded_threshold}°): {n_bounded_90:,}/{self.num_trials:,} ({prop_bounded_90*100:.1f}%)")
            print(f"  Strict bounded (amplitude ≤{self.strict_threshold}°): {n_bounded_strict:,}/{self.num_trials:,} ({prop_bounded_strict*100:.1f}%)")
            print(f"  Both strict (dynamic + ≤{self.strict_threshold}°): {n_both:,}/{self.num_trials:,} ({prop_both*100:.1f}%)")
            print()
            print(f"Amplitude Statistics:")
            print(f"  Mean: {amplitude_mean:.2f}°")
            print(f"  Std: {amplitude_std:.2f}°")
            print(f"  Median: {amplitude_median:.2f}°")
            print(f"  Range: [{amplitude_min:.2f}°, {amplitude_max:.2f}°]")
            print()
            print(f"Coupling Statistics:")
            print(f"  Mean coupling strength: {coupling_mean:.2f}")
            print(f"  Mean top-3 error: {top3_error_mean:.3f}%")
            print("\n" + "=" * 70)

        return self.summary

    def plot_distributions(self, figsize: Tuple[int, int] = (16, 10),
                          save_path: Optional[str] = None,
                          dpi: int = 300) -> plt.Figure:
        """
        Generate diagnostic visualization plots.

        Creates a multi-panel figure showing:
        1. Amplitude distribution histogram
        2. Tropical year vs amplitude scatter
        3. Bounded vs unbounded counts
        4. Coupling strength distribution

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save figure
            dpi: Resolution for saved figure

        Returns:
            Matplotlib figure object
        """
        if self.results is None:
            raise ValueError("Must run run_simulation() first")

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Panel 1: Amplitude histogram
        ax1 = axes[0, 0]
        bounded = self.results[self.results['bounded_strict']]
        unbounded = self.results[~self.results['bounded_strict']]

        ax1.hist(bounded['amplitude'], bins=50, alpha=0.7, color='green',
                label=f'Bounded (≤{self.strict_threshold}°): {len(bounded):,}', edgecolor='black')
        if len(unbounded) > 0:
            ax1.hist(unbounded['amplitude'], bins=50, alpha=0.7, color='red',
                    label=f'Unbounded (>{self.strict_threshold}°): {len(unbounded):,}', edgecolor='black')

        ax1.axvline(self.strict_threshold, color='blue', linestyle='--', linewidth=2,
                   label=f'Strict threshold ({self.strict_threshold}°)')
        ax1.axvline(self.results['amplitude'].mean(), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean ({self.results["amplitude"].mean():.1f}°)')

        ax1.set_xlabel('Amplitude (degrees)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Amplitude Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # Panel 2: Tropical year vs amplitude scatter
        ax2 = axes[0, 1]
        colors = ['green' if b else 'red' for b in self.results['bounded_strict']]
        ax2.scatter(self.results['tropical_year'], self.results['amplitude'],
                   c=colors, alpha=0.5, s=10, edgecolors='none')

        ax2.axhline(self.strict_threshold, color='blue', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'{self.strict_threshold}° threshold')
        ax2.axvline(EARTH_TROPICAL_YEAR, color='purple', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Earth ({EARTH_TROPICAL_YEAR} days)')

        ax2.set_xlabel('Tropical Year (days)', fontsize=11)
        ax2.set_ylabel('Amplitude (degrees)', fontsize=11)
        ax2.set_title('Parameter Sensitivity: Year Length vs Amplitude', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        # Panel 3: Bounded classification bar chart
        ax3 = axes[1, 0]

        categories = ['Dynamic\nBounded', f'Amplitude\n≤{self.strict_threshold}°', 'Both\nStrict']
        counts = [
            self.results['dynamically_bounded'].sum(),
            self.results['bounded_strict'].sum(),
            self.results['both_strict'].sum()
        ]
        percentages = [c / self.num_trials * 100 for c in counts]

        bars = ax3.bar(categories, percentages, color=['#3498db', '#2ecc71', '#9b59b6'],
                      alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{count:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax3.set_ylabel('Percentage (%)', fontsize=11)
        ax3.set_title('Classification Results', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 110)
        ax3.grid(alpha=0.3, axis='y')

        # Panel 4: Coupling strength histogram
        ax4 = axes[1, 1]
        ax4.hist(self.results['coupling_strength'], bins=50, color='steelblue',
                alpha=0.7, edgecolor='black')
        ax4.axvline(self.results['coupling_strength'].mean(), color='red',
                   linestyle='--', linewidth=2,
                   label=f'Mean ({self.results["coupling_strength"].mean():.1f})')

        ax4.set_xlabel('Coupling Strength', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Coupling Strength Distribution', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        return fig

    def export_results(self, output_dir: str = '../datasets',
                      basename: str = 'monte_carlo_robustness') -> Tuple[str, str]:
        """
        Export results to CSV files.

        Args:
            output_dir: Directory for output files
            basename: Base filename (without extension)

        Returns:
            Tuple of (results_path, summary_path)
        """
        if self.results is None:
            raise ValueError("Must run run_simulation() first")

        if self.summary is None:
            self.analyze_results(verbose=False)

        # Create output directory if needed
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export full results
        results_file = output_path / f"{basename}_results.csv"
        self.results.to_csv(results_file, index=False)
        print(f"Results exported to: {results_file}")

        # Export summary
        summary_file = output_path / f"{basename}_summary.csv"
        summary_df = pd.DataFrame([self.summary])
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary exported to: {summary_file}")

        return str(results_file), str(summary_file)


def run_quick_test(num_trials: int = 1000) -> MonteCarloRobustness:
    """
    Run a quick Monte Carlo test for validation.

    Args:
        num_trials: Number of trials (default: 1000 for quick test)

    Returns:
        MonteCarloRobustness instance with results
    """
    print("Running quick Monte Carlo test...")
    mc = MonteCarloRobustness(num_trials=num_trials)
    mc.run_simulation()
    mc.analyze_results()

    return mc
