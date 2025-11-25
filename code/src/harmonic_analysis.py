"""
Harmonic Analysis Module for Enoch Calendar Phase-Locking Study

This module provides FFT-based analysis to identify dominant harmonic frequencies
in the Enoch calendar's oscillation pattern. It was used to discover the three
fundamental periods (13,965 / 27,930 / 9,310 years) that demonstrate the calendar's
phase-locking to Earth's precession cycle.

The analysis reveals that the 294-day correction mechanism transforms unbounded
linear drift into bounded sinusoidal oscillation through harmonic resonance.
"""

import warnings

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from typing import Dict, List, Optional


class HarmonicAnalyzer:
    """
    Analyzes periodic signals using Fast Fourier Transform (FFT) to identify
    dominant harmonic frequencies and fit multi-frequency sinusoidal models.

    This class encapsulates the methodology used to discover that the Enoch
    calendar's 294-day correction creates a perpetual self-correcting system
    phase-locked to Earth's precession cycle.

    Usage:
        analyzer = HarmonicAnalyzer(years, ecliptic_longitudes)
        periods = analyzer.perform_fft(n_peaks=3)
        results = analyzer.fit_multi_harmonic(periods)
    """

    def __init__(self, years: np.ndarray, values: np.ndarray):
        """
        Initialize the harmonic analyzer with time series data.

        Parameters:
        -----------
        years : np.ndarray
            Time values (typically solar years). Must be evenly spaced.
        values : np.ndarray
            Measured values (typically ecliptic longitude in degrees).
            Should have same length as years.

        Example:
        --------
        # From calendar data with 27,930 Day 1 events
        years = day1_data['solar_year'].values
        ecliptic = day1_data['sun_ecliptic_longitude_neg'].values
        analyzer = HarmonicAnalyzer(years, ecliptic)
        """
        if len(years) != len(values):
            raise ValueError(f"years and values must have same length (got {len(years)} vs {len(values)})")

        self.years = np.array(years, dtype=float)
        self.values = np.array(values, dtype=float)
        self.n_points = len(years)

        # Storage for analysis results
        self.fft_frequencies = None
        self.fft_power = None
        self.dominant_periods = None
        self.fitted_params = None
        self.fitted_values = None
        self.r_squared = None
        self.rmse = None

        # Diagnostics for sampling cadence
        self._sampling_stats = {}
        self.sample_spacing = None
        self.irregular_sampling = False

        self._prepare_sampling_metadata()

    def _prepare_sampling_metadata(self) -> None:
        """Analyze sampling cadence to support FFT diagnostics."""
        if self.n_points < 2:
            self.sample_spacing = 1.0
            self._sampling_stats = {'message': 'insufficient data for spacing diagnostics'}
            return

        # Ensure monotonic ordering using a stable sort so ties keep original order
        order = np.argsort(self.years, kind='mergesort')
        if not np.all(order == np.arange(self.n_points)):
            self.years = self.years[order]
            self.values = self.values[order]

        diffs = np.diff(self.years)
        positive_diffs = diffs[diffs > 0]

        if positive_diffs.size == 0:
            raise ValueError("Year values must contain an increasing trend for harmonic analysis")

        # Robust estimate of cadence: use median of strictly positive steps
        cadence = float(np.median(positive_diffs))
        self.sample_spacing = cadence if cadence > 0 else 1.0

        zero_steps = int(np.sum(np.isclose(diffs, 0.0)))
        double_steps = int(np.sum(np.isclose(diffs, 2.0)))
        max_step = float(np.max(diffs))
        min_step = float(np.min(diffs))

        # Flag irregular sampling when variance is larger than tolerance (~1 day on a yearly scale)
        tolerance = 1e-3  # about 0.36 days in year units
        irregular_count = int(np.sum(np.abs(diffs - self.sample_spacing) > tolerance))
        self.irregular_sampling = irregular_count > 0

        self._sampling_stats = {
            'zero_steps': zero_steps,
            'double_steps': double_steps,
            'min_step': min_step,
            'max_step': max_step,
            'median_step': self.sample_spacing,
            'irregular_samples': irregular_count,
            'total_samples': self.n_points
        }

        if self.irregular_sampling:
            warnings.warn(
                "Detected irregular year spacing ({} anomalies, median step {:.6f}). "
                "FFT results now use the median cadence; consider resampling for highest fidelity.".format(
                    irregular_count, self.sample_spacing
                ),
                RuntimeWarning
            )

    def perform_fft(
        self,
        n_peaks: int = 3,
        *,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None
    ) -> List[float]:
        """
        Perform Fast Fourier Transform to identify dominant periodic components.

        FFT converts time-domain signal (ecliptic longitude vs. year) into
        frequency-domain (amplitude vs. frequency), revealing periodic patterns
        that may not be obvious in the raw data.

        Parameters:
        -----------
        n_peaks : int
            Number of dominant frequencies to return (default: 3)
            The original analysis found 3 harmonics: 13,965 / 27,930 / 9,310 years

        Returns:
        --------
        periods : List[float]
            Dominant periods in years, sorted by power (strongest first)

        Notes:
        ------
        - Removes DC component (mean) to focus on oscillations
        - Only analyzes positive frequencies (negative are mirror images)
        - Power spectrum = magnitude of complex FFT coefficients
        - Period = 1 / frequency (e.g., freq=0.00007161 → period=13,965 years)

        Example:
        --------
        periods = analyzer.perform_fft(n_peaks=3)
        # Result: [13965.0, 27930.0, 9310.0]
        """
        # Remove DC component (mean) - we only care about oscillations around zero
        # This eliminates any constant offset in the ecliptic longitude
        signal = self.values - np.mean(self.values)

        # Perform FFT - converts time series to frequency spectrum
        yf = fft(signal)

        # Use measured cadence when building the frequency grid
        xf = fftfreq(self.n_points, d=self.sample_spacing if self.sample_spacing else 1.0)

        # Only analyze positive frequencies (negative frequencies are redundant)
        positive_mask = xf > 0
        positive_freqs = xf[positive_mask]

        # Calculate power spectrum = magnitude of complex FFT coefficients
        positive_power = np.abs(yf[positive_mask])

        # Convert to periods and optionally filter by bounds
        periods = 1.0 / positive_freqs
        period_mask = np.ones_like(periods, dtype=bool)
        if min_period is not None:
            period_mask &= periods >= min_period
        if max_period is not None:
            period_mask &= periods <= max_period

        positive_freqs = positive_freqs[period_mask]
        positive_power = positive_power[period_mask]
        periods = periods[period_mask]

        if positive_power.size == 0:
            raise ValueError("No frequencies remaining after applying period constraints")

        # Store for later analysis
        self.fft_frequencies = positive_freqs
        self.fft_power = positive_power

        if n_peaks > positive_power.size:
            warnings.warn(
                f"Requested {n_peaks} FFT peaks but only {positive_power.size} available after filtering; "
                f"reducing to {positive_power.size}.",
                RuntimeWarning
            )
            n_peaks = positive_power.size

        # Find the N strongest peaks in the power spectrum
        top_indices = np.argsort(positive_power)[-n_peaks:][::-1]

        # Extract the dominant frequencies and powers
        dominant_freqs = positive_freqs[top_indices]
        dominant_power = positive_power[top_indices]
        dominant_periods = periods[top_indices]

        # Convert to plain Python floats for downstream callers
        self.dominant_periods = [float(period) for period in dominant_periods]

        # Print discovery summary
        print(f"=== FFT Analysis Results ===")
        print(f"Analyzed {self.n_points} data points spanning {self.years[-1] - self.years[0]:.0f} years")
        print(f"Sampling cadence (median): {self.sample_spacing:.6f} years")
        print(f"\nTop {n_peaks} dominant periodic components:")
        for i, (period, power) in enumerate(zip(self.dominant_periods, dominant_power), 1):
            print(f"  {i}. Period: {period:>10.1f} years (power: {power:.2e})")

        return self.dominant_periods

    def fit_multi_harmonic(self, periods: Optional[List[float]] = None) -> Dict:
        """
        Fit a multi-frequency sinusoidal model to the data.

        The model equation for N harmonics is:
            y(t) = Σ[A_i * sin(2π * t / P_i + φ_i)] + offset

        Where:
            A_i = amplitude of harmonic i (degrees)
            P_i = period of harmonic i (years) - FIXED from FFT
            φ_i = phase shift of harmonic i (radians)
            offset = mean baseline (degrees)

        Parameters:
        -----------
        periods : List[float], optional
            Harmonic periods to fit (in years). If None, uses results from perform_fft()
            Example: [13965, 27930, 9310]

        Returns:
        --------
        results : Dict
            Dictionary containing:
                'amplitudes': List of fitted amplitudes (degrees)
                'phases': List of fitted phase shifts (radians)
                'offset': Baseline offset (degrees)
                'r_squared': Coefficient of determination (0-1)
                'rmse': Root mean square error (degrees)
                'fitted_values': Model predictions for all years

        Notes:
        ------
        - Uses scipy's curve_fit with Levenberg-Marquardt optimization
        - Periods are FIXED (from FFT); only amplitudes and phases are fitted
        - This differs from allowing periods to vary, which can lead to overfitting
        - The fixed-period approach is more stable for extrapolation

        Example:
        --------
        results = analyzer.fit_multi_harmonic([13965, 27930, 9310])
        print(f"R² = {results['r_squared']:.6f}")
        """
        if periods is None:
            if self.dominant_periods is None:
                raise ValueError("Must call perform_fft() first or provide periods explicitly")
            periods = self.dominant_periods

        periods = [float(p) for p in periods]
        n_harmonics = len(periods)

        # Define multi-harmonic model function
        # This is what curve_fit will try to match to the data
        def multi_harmonic_model(x, *params):
            """
            Multi-frequency sine wave model.

            Parameters are packed as: [A1, phi1, A2, phi2, ..., An, phin, offset]
            where n = number of harmonics
            """
            result = np.zeros_like(x, dtype=float)

            # Add each harmonic component
            for i in range(n_harmonics):
                amplitude = params[i * 2]      # A_i
                phase = params[i * 2 + 1]      # φ_i
                period = periods[i]             # P_i (fixed)

                # Add: A_i * sin(2π * x / P_i + φ_i)
                result += amplitude * np.sin(2 * np.pi * x / period + phase)

            # Add constant offset (last parameter)
            offset = params[-1]
            result += offset

            return result

        # Initial parameter guesses
        # Good initial guesses help optimization converge faster
        amplitude_guess = (self.values.max() - self.values.min()) / 2  # Half of data range
        offset_guess = np.mean(self.values)                            # Data mean

        initial_params = []
        for _ in range(n_harmonics):
            initial_params.extend([amplitude_guess * 0.5, 0.0])  # [amplitude, phase] for each harmonic
        initial_params.append(offset_guess)  # offset at end

        # Fit the model using non-linear least squares
        # maxfev=50000 allows more iterations for convergence
        try:
            fitted_params, covariance = curve_fit(
                multi_harmonic_model,
                self.years,
                self.values,
                p0=initial_params,
                maxfev=50000
            )
        except RuntimeError as e:
            raise RuntimeError(f"Curve fitting failed to converge: {e}")

        # Store fitted parameters
        self.fitted_params = fitted_params
        # Keep harmonic periods in sync with the fitted parameter vector
        self.dominant_periods = periods

        # Calculate fitted values (model predictions)
        self.fitted_values = multi_harmonic_model(self.years, *fitted_params)

        # Calculate goodness of fit metrics
        residuals = self.values - self.fitted_values

        # R² (coefficient of determination): 1.0 = perfect fit, 0.0 = no better than mean
        ss_residual = np.sum(residuals**2)
        ss_total = np.sum((self.values - np.mean(self.values))**2)
        self.r_squared = 1.0 - (ss_residual / ss_total)

        # RMSE (root mean square error): average prediction error in same units as data
        self.rmse = np.sqrt(np.mean(residuals**2))

        # Parse fitted parameters into readable format
        amplitudes = []
        phases = []
        for i in range(n_harmonics):
            amplitudes.append(fitted_params[i * 2])
            phases.append(fitted_params[i * 2 + 1])
        offset = fitted_params[-1]

        # Print fit summary
        print(f"\n=== Multi-Harmonic Fit Results ===")
        print(f"Fitted {n_harmonics}-frequency model")
        for i, (period, amp, phase) in enumerate(zip(periods, amplitudes, phases), 1):
            print(f"  Component {i}: Period={period:>10.1f} yr, Amplitude={amp:+7.3f}°, Phase={phase:+7.3f} rad")
        print(f"  Offset: {offset:+7.3f}°")
        print(f"\nGoodness of fit:")
        print(f"  R² = {self.r_squared:.6f}")
        print(f"  RMSE = {self.rmse:.3f}°")

        # Return results dictionary
        return {
            'periods': periods,
            'amplitudes': amplitudes,
            'phases': phases,
            'offset': offset,
            'r_squared': self.r_squared,
            'rmse': self.rmse,
            'fitted_values': self.fitted_values,
            'residuals': residuals
        }

    def predict(self, years: np.ndarray) -> np.ndarray:
        """
        Generate predictions for new time points using the fitted model.

        This allows extrapolation beyond the original data range, which is how
        we predict calendar behavior 70,000 years into the future.

        Parameters:
        -----------
        years : np.ndarray
            Years to predict (can be outside original range)

        Returns:
        --------
        predictions : np.ndarray
            Model predictions at specified years

        Example:
        --------
        # Predict from -20,000 to +50,000
        future_years = np.arange(-20000, 50000, 50)
        predictions = analyzer.predict(future_years)
        """
        if self.fitted_params is None:
            raise ValueError("Must call fit_multi_harmonic() before making predictions")

        if self.dominant_periods is None:
            raise ValueError("No periods available - run perform_fft() first")

        n_harmonics = len(self.dominant_periods)
        expected_params = 2 * n_harmonics + 1
        if len(self.fitted_params) != expected_params:
            raise ValueError(
                f"Fitted parameter vector length {len(self.fitted_params)} does not match "
                f"{n_harmonics} harmonics"
            )
        result = np.zeros_like(years, dtype=float)

        # Reconstruct the model
        for i in range(n_harmonics):
            amplitude = self.fitted_params[i * 2]
            phase = self.fitted_params[i * 2 + 1]
            period = self.dominant_periods[i]

            result += amplitude * np.sin(2 * np.pi * years / period + phase)

        # Add offset
        offset = self.fitted_params[-1]
        result += offset

        return result

    def analyze_residuals(self, n_additional_peaks: int = 5) -> Dict:
        """
        Analyze residuals from the fitted model to identify additional
        harmonic frequencies that might improve fit quality.

        This performs FFT on the residuals (observed - predicted) to check
        if there are significant periodic patterns that weren't captured by
        the current model. This is key to improving R² beyond 0.82.

        Parameters:
        -----------
        n_additional_peaks : int
            Number of additional frequency peaks to identify in residuals

        Returns:
        --------
        residual_analysis : Dict
            Dictionary containing:
                'residual_periods': List of periods found in residuals
                'residual_powers': List of corresponding power values
                'max_residual_power': Maximum power in residual spectrum
                'significant': Boolean indicating if additional harmonics are significant

        Notes:
        ------
        - Residuals should ideally be white noise (no structure)
        - If residuals show periodic structure, model is missing frequencies
        - A significant peak suggests adding that frequency could improve fit
        - Use this to decide whether to add 4th, 5th harmonics, etc.

        Example:
        --------
        residual_info = analyzer.analyze_residuals(n_additional_peaks=5)
        if residual_info['significant']:
            print("Additional harmonics may improve fit")
        """
        if self.fitted_values is None:
            raise ValueError("Must fit model first with fit_multi_harmonic()")

        # Calculate residuals
        residuals = self.values - self.fitted_values

        # Perform FFT on residuals
        N = len(residuals)
        yf_residual = fft(residuals - np.mean(residuals))
        xf_residual = fftfreq(N, d=1.0)

        # Analyze positive frequencies only
        positive_mask = xf_residual > 0
        positive_freqs_res = xf_residual[positive_mask]
        positive_power_res = np.abs(yf_residual[positive_mask])

        # Find top peaks in residual spectrum
        top_residual_indices = np.argsort(positive_power_res)[-n_additional_peaks:][::-1]
        residual_freqs = positive_freqs_res[top_residual_indices]
        residual_powers = positive_power_res[top_residual_indices]

        # Convert to periods
        residual_periods = [1.0 / freq for freq in residual_freqs if freq > 0]

        # Determine if residuals show significant periodic structure
        # Compare max residual power to original signal power
        max_residual_power = np.max(positive_power_res)
        max_original_power = np.max(self.fft_power) if self.fft_power is not None else 0

        # If residual power is > 10% of original power, consider it significant
        significance_threshold = 0.1
        is_significant = (max_residual_power / max_original_power) > significance_threshold

        print(f"=== Residual Analysis ===")
        print(f"Current model R² = {self.r_squared:.6f}")
        print(f"Residual RMSE = {self.rmse:.3f}°")
        print(f"\nTop {n_additional_peaks} periods in residuals:")
        for i, (period, power) in enumerate(zip(residual_periods, residual_powers), 1):
            print(f"  {i}. Period: {period:>10.1f} years (power: {power:.2e})")

        print(f"\nSignificance assessment:")
        print(f"  Max residual power: {max_residual_power:.2e}")
        print(f"  Max original power: {max_original_power:.2e}")
        print(f"  Ratio: {max_residual_power / max_original_power:.2%}")
        print(f"  Significant (>{significance_threshold:.0%}): {is_significant}")

        if is_significant:
            print(f"\n  → Additional harmonics may improve model fit")
        else:
            print(f"\n  → Residuals appear to be noise; current model is sufficient")

        return {
            'residual_periods': residual_periods,
            'residual_powers': residual_powers.tolist(),
            'max_residual_power': max_residual_power,
            'max_original_power': max_original_power,
            'power_ratio': max_residual_power / max_original_power,
            'significant': is_significant
        }

    def test_additional_harmonics(self, max_harmonics: int = 7) -> Dict:
        """
        Systematically test models with 1 to max_harmonics frequencies
        and compare their performance using information criteria (AIC/BIC).

        This helps answer: "Should we add a 4th harmonic? 5th? When to stop?"

        Parameters:
        -----------
        max_harmonics : int
            Maximum number of harmonics to test (default: 7)

        Returns:
        --------
        comparison : Dict
            Dictionary with keys for each n_harmonics tested, containing:
                'periods': Periods used
                'r_squared': Fit quality
                'rmse': Root mean square error
                'aic': Akaike Information Criterion
                'bic': Bayesian Information Criterion
                'n_parameters': Number of fitted parameters
                'best_by_aic': Boolean, True if this is best by AIC
                'best_by_bic': Boolean, True if this is best by BIC

        Notes:
        ------
        - AIC/BIC penalize model complexity (more parameters)
        - Lower AIC/BIC = better model balancing fit vs. complexity
        - BIC penalizes complexity more than AIC
        - Best model maximizes R² while minimizing AIC/BIC

        Example:
        --------
        comparison = analyzer.test_additional_harmonics(max_harmonics=7)
        best_n = min(comparison.keys(), key=lambda k: comparison[k]['bic'])
        print(f"Optimal number of harmonics (by BIC): {best_n}")
        """
        if self.fft_frequencies is None or self.fft_power is None:
            raise ValueError("Must run perform_fft() first")

        # Get top N frequencies from original FFT
        n_candidates = min(max_harmonics * 2, len(self.fft_power))  # Get more than needed
        top_indices = np.argsort(self.fft_power)[-n_candidates:][::-1]
        candidate_freqs = self.fft_frequencies[top_indices]
        candidate_periods = [1.0 / freq for freq in candidate_freqs if freq > 0]

        results = {}

        print(f"=== Testing Models with 1 to {max_harmonics} Harmonics ===\n")

        for n in range(1, max_harmonics + 1):
            # Use top N periods
            periods_to_test = candidate_periods[:n]

            # Fit model with these periods
            try:
                fit_result = self.fit_multi_harmonic(periods_to_test)

                # Calculate information criteria
                n_params = 2 * n + 1  # n amplitudes + n phases + 1 offset
                n_data = self.n_points
                rss = np.sum(fit_result['residuals']**2)

                # AIC = n * ln(RSS/n) + 2k
                # where k = number of parameters
                aic = n_data * np.log(rss / n_data) + 2 * n_params

                # BIC = n * ln(RSS/n) + k * ln(n)
                # BIC penalizes complexity more than AIC
                bic = n_data * np.log(rss / n_data) + n_params * np.log(n_data)

                results[n] = {
                    'periods': periods_to_test,
                    'r_squared': fit_result['r_squared'],
                    'rmse': fit_result['rmse'],
                    'aic': aic,
                    'bic': bic,
                    'n_parameters': n_params,
                    'rss': rss
                }

                print(f"  {n} harmonics: R²={fit_result['r_squared']:.6f}, "
                      f"RMSE={fit_result['rmse']:.3f}°, AIC={aic:.1f}, BIC={bic:.1f}")

            except Exception as e:
                print(f"  {n} harmonics: FAILED ({e})")
                continue

        # Identify best models
        if results:
            best_aic_n = min(results.keys(), key=lambda k: results[k]['aic'])
            best_bic_n = min(results.keys(), key=lambda k: results[k]['bic'])
            best_r2_n = max(results.keys(), key=lambda k: results[k]['r_squared'])

            for n in results:
                results[n]['best_by_aic'] = (n == best_aic_n)
                results[n]['best_by_bic'] = (n == best_bic_n)
                results[n]['best_by_r2'] = (n == best_r2_n)

            print(f"\n=== Model Selection Results ===")
            print(f"  Best by AIC: {best_aic_n} harmonics (prefers fit quality)")
            print(f"  Best by BIC: {best_bic_n} harmonics (prefers simplicity)")
            print(f"  Best by R²:  {best_r2_n} harmonics (may overfit)")
            print(f"\nRecommendation: Use {best_bic_n} harmonics (BIC criterion)")

        return results

    def get_sampling_diagnostics(self) -> Dict:
        """Return information about observed sampling cadence."""
        return dict(self._sampling_stats)

    def find_harmonic_at_period(
        self,
        target_period: float,
        residuals: Optional[np.ndarray] = None,
        tolerance: float = 0.05
    ) -> Optional[tuple[float, float]]:
        """
        Search for harmonic component at specific period using least squares.

        This uses the least-squares projection method to detect if a sinusoidal
        component at the target period is present in the residuals.

        Parameters:
        -----------
        target_period : float
            Period to search for (in years)
        residuals : Optional[np.ndarray]
            Residuals to search in. If None, uses current model residuals.
        tolerance : float
            Minimum amplitude to consider significant (degrees)

        Returns:
        --------
        result : Optional[tuple[float, float]]
            (amplitude, phase) if significant, None otherwise

        Notes:
        ------
        - Fits sin and cos components separately using least squares
        - Converts to amplitude-phase form
        - Only returns result if amplitude exceeds tolerance

        Example:
        --------
        # Search for 294-year harmonic in baseline residuals
        result = analyzer.find_harmonic_at_period(294.0, tolerance=0.05)
        if result:
            amplitude, phase = result
            print(f"Found 294-year harmonic: A={amplitude:.3f}°, φ={phase:.3f} rad")
        """
        if residuals is None:
            if self.fitted_values is None:
                raise ValueError("Must fit model first or provide residuals")
            residuals = self.values - self.fitted_values

        omega = 2 * np.pi / target_period

        # Fit sine and cosine components
        sin_component = np.sin(omega * self.years)
        cos_component = np.cos(omega * self.years)

        # Least squares fit
        A_sin = np.dot(residuals, sin_component) / np.dot(sin_component, sin_component)
        A_cos = np.dot(residuals, cos_component) / np.dot(cos_component, cos_component)

        # Convert to amplitude and phase
        amplitude = np.sqrt(A_sin**2 + A_cos**2)
        phase = np.arctan2(A_cos, A_sin)

        return (amplitude, phase) if amplitude > tolerance else None

    def search_294_multiples(
        self,
        n_range: tuple[int, int] = (1, 100),
        tolerance: float = 0.05,
        max_period_fraction: float = 0.8,
        use_residuals: bool = True,
        verbose: bool = False
    ) -> list[dict]:
        """
        Systematically search for significant 294-year multiples.

        This iteratively searches for harmonics at periods 294×n for n in the
        specified range. Each detected harmonic is removed from the residuals
        before searching for the next one (greedy forward selection).

        Parameters:
        -----------
        n_range : tuple[int, int]
            Range of multiples to search (min_n, max_n)
        tolerance : float
            Minimum amplitude to consider significant (degrees)
        max_period_fraction : float
            Skip periods > this fraction of data duration
        use_residuals : bool
            If True, search in current model residuals; if False, search in raw data
        verbose : bool
            If True, print search progress

        Returns:
        --------
        detected : list[dict]
            List of dicts with keys: n, period, amplitude, phase

        Notes:
        ------
        - Searches systematically through 294×n multiples
        - Uses iterative residual removal (greedy selection)
        - Skips periods too long for reliable detection
        - Returns only harmonics exceeding significance threshold

        Example:
        --------
        # After fitting baseline 3-harmonic model
        analyzer.fit_multi_harmonic([14700, 29400, 9800])

        # Search for additional 294-year multiples
        detected = analyzer.search_294_multiples(
            n_range=(1, 100),
            tolerance=0.05,
            verbose=True
        )

        print(f"Found {len(detected)} additional harmonics")
        for h in detected:
            print(f"  n={h['n']}: Period={h['period']:.0f} yr, Amp={h['amplitude']:.3f}°")
        """
        if use_residuals and self.fitted_values is not None:
            search_signal = self.values - self.fitted_values
        else:
            search_signal = self.values.copy()

        detected = []
        current_residuals = search_signal.copy()
        duration = self.years[-1] - self.years[0]

        if verbose:
            print(f"\nSystematic search for 294-year multiples (n={n_range[0]} to {n_range[1]}):")
            print("=" * 80)

        for n in range(n_range[0], n_range[1] + 1):
            target_period = 294 * n

            # Skip if period exceeds dataset duration
            if target_period > duration * max_period_fraction:
                if verbose and n <= 20:
                    print(f"  n={n:2d}: Period={target_period:6.0f} yr - SKIPPED (too long)")
                continue

            result = self.find_harmonic_at_period(
                target_period,
                current_residuals,
                tolerance
            )

            if result is not None:
                amplitude, phase = result
                detected.append({
                    'n': n,
                    'period': target_period,
                    'amplitude': amplitude,
                    'phase': phase
                })

                if verbose and n <= 20:
                    print(f"  n={n:2d}: Period={target_period:6.0f} yr - DETECTED (A={amplitude:.4f}°)")

                # Remove this harmonic from residuals for next iteration
                omega = 2 * np.pi / target_period
                harmonic_signal = amplitude * np.sin(omega * self.years + phase)
                current_residuals -= harmonic_signal
            else:
                if verbose and n <= 20:
                    print(f"  n={n:2d}: Period={target_period:6.0f} yr - not significant")

        if verbose:
            print(f"\n✓ Detected {len(detected)} significant harmonics (amplitude > {tolerance}°)")

        return detected

    def get_summary(self) -> Dict:
        """
        Get a complete summary of the analysis results.

        Returns:
        --------
        summary : Dict
            Complete analysis results including FFT findings and fit quality
        """
        if self.dominant_periods is None or self.fitted_params is None:
            raise ValueError("Must run perform_fft() and fit_multi_harmonic() first")

        n_harmonics = len(self.dominant_periods)

        amplitudes = []
        phases = []
        for i in range(n_harmonics):
            amplitudes.append(self.fitted_params[i * 2])
            phases.append(self.fitted_params[i * 2 + 1])

        return {
            'n_data_points': self.n_points,
            'year_range': (self.years[0], self.years[-1]),
            'year_span': self.years[-1] - self.years[0],
            'dominant_periods': self.dominant_periods,
            'amplitudes': amplitudes,
            'phases': phases,
            'offset': self.fitted_params[-1],
            'r_squared': self.r_squared,
            'rmse': self.rmse,
            'value_range': (self.values.min(), self.values.max())
        }
