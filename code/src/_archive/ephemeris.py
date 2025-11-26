"""
Solar Ephemeris Calculations using JPL DE441

This module provides high-precision solar position calculations using
the JPL DE441 planetary ephemeris via Swiss Ephemeris library.

Key functionality:
- Solar ecliptic longitude at calendar Day 1 events
- Deviation from vernal equinox (0°)
- Time series validation across precession timescales
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Optional
from src.enoch_config import SWISS_EPH_PATH

try:
    import swisseph as swe
    HAS_SWISSEPH = True
except ImportError:
    HAS_SWISSEPH = False
    print("Warning: pyswisseph not installed. Ephemeris calculations will not work.")
    print("Install with: pip install pyswisseph")

# Set Swiss Ephemeris path from environment
if HAS_SWISSEPH:
    ephe_path = SWISS_EPH_PATH
    if ephe_path and os.path.exists(ephe_path):
        swe.set_ephe_path(ephe_path)
        print(f"Swiss Ephemeris path set to: {ephe_path}")
    else:
        print(f"Warning: EPHEMERIS_PATH not found or invalid: {ephe_path}")
        print("Swiss Ephemeris will use default paths.")


# Ephemeris constants
VERNAL_EQUINOX_LONGITUDE = 0.0  # degrees (J2000.0 frame)
SECONDS_PER_DAY = 86400.0


def calculate_solar_longitude(jd: float, use_terrestrial_time: bool = True) -> tuple:
    """
    Calculate solar ecliptic longitude at given Julian Date.

    Args:
        jd: Julian Date (TT or UT depending on use_terrestrial_time)
        use_terrestrial_time: Use Terrestrial Time (True) or Universal Time (False)

    Returns:
        tuple of (longitude_degrees, distance_au, speed_deg_per_day)

    Raises:
        ImportError: If pyswisseph not installed
    """
    if not HAS_SWISSEPH:
        raise ImportError("pyswisseph required for ephemeris calculations")

    # Select calculation function
    calc_func = swe.calc if use_terrestrial_time else swe.calc_ut

    # Calculate Sun position
    # Returns: (longitude, latitude, distance, speed_long, speed_lat, speed_dist)
    sun_data = calc_func(jd, swe.SUN, 0)

    longitude = sun_data[0][0]  # Ecliptic longitude (degrees)
    distance = sun_data[0][2]   # Distance (AU)
    speed = sun_data[0][3]      # Longitude speed (deg/day)

    return longitude, distance, speed


def calculate_deviation_from_equinox(longitude: float) -> float:
    """
    Calculate signed angular deviation from vernal equinox (0°).

    Converts ecliptic longitude to signed deviation:
    - 0° → 0° (exactly at equinox)
    - 10° → +10° (ahead of equinox)
    - 350° → -10° (behind equinox)

    Range: [-180°, +180°]

    Args:
        longitude: Ecliptic longitude in degrees [0, 360)

    Returns:
        Signed deviation from 0° in degrees
    """
    # Normalize to [0, 360)
    longitude = longitude % 360

    # Convert to signed range [-180, 180]
    if longitude > 180:
        return longitude - 360
    else:
        return longitude


def gregorian_to_julian_date(year: int, month: int = 3, day: int = 20,
                             hour: float = 12.0) -> float:
    """
    Convert Gregorian calendar date to Julian Date.

    Args:
        year: Year (can be negative for BCE)
        month: Month (1-12, default: 3 for March)
        day: Day of month (default: 20 for approximate equinox)
        hour: Hour of day in decimal (default: 12.0 for noon)

    Returns:
        Julian Date (TT)
    """
    if not HAS_SWISSEPH:
        raise ImportError("pyswisseph required for date conversion")

    # Swiss Ephemeris uses Gregorian calendar for dates after 1582-10-15
    # Julian calendar for earlier dates (handles automatically)
    jd = swe.julday(year, month, day, hour)

    return jd


def calculate_vernal_equinox_jd(year: int, use_terrestrial_time: bool = True) -> float:
    """
    Calculate the precise Julian Date of the vernal equinox for a given year.

    This finds the exact moment when the Sun's ecliptic longitude = 0°.

    Args:
        year: Year (CE, can be negative for BCE)
        use_terrestrial_time: Use TT (True) or UT (False)

    Returns:
        Julian Date of vernal equinox (TT or UT)
    """
    if not HAS_SWISSEPH:
        raise ImportError("pyswisseph required for equinox calculation")

    # Approximate JD for March 20 of the given year (starting point for search)
    approx_jd = gregorian_to_julian_date(year, 3, 20, 0.0)

    # Search for exact moment when Sun longitude = 0°
    # Swiss Ephemeris flag for tropical zodiac (0° = vernal equinox)
    flags = 0  # Tropical zodiac (default)

    # Use swe.solcross_ut or swe.solcross for equinox/solstice calculations
    # For vernal equinox: longitude = 0°
    # Search window: ±30 days from March 20

    calc_func = swe.calc if use_terrestrial_time else swe.calc_ut

    # Binary search for when Sun longitude crosses 0°
    search_jd = approx_jd
    tolerance = 1e-6  # About 0.1 seconds precision
    max_iterations = 50

    for _ in range(max_iterations):
        sun_data = calc_func(search_jd, swe.SUN, flags)
        longitude = sun_data[0][0]

        # Handle wraparound: 359° → -1° relative to 0°
        if longitude > 180:
            longitude = longitude - 360

        # Check if we're close enough
        if abs(longitude) < tolerance:
            return search_jd

        # Estimate correction needed
        # Sun moves ~1° per day, so days_offset ≈ -longitude
        days_offset = -longitude / 0.9856  # Average solar motion
        search_jd += days_offset

        # Safety check: don't search beyond ±45 days
        if abs(search_jd - approx_jd) > 45:
            break

    # If search didn't converge, return approximate value
    return search_jd


def calculate_timeseries_deviations(calendar_df: pd.DataFrame,
                                    start_year: int = -12762,
                                    use_terrestrial_time: bool = True,
                                    use_actual_equinox: bool = True,
                                    verbose: bool = True) -> pd.DataFrame:
    """
    Calculate solar longitude deviations for all Day 1 events in calendar.

    Args:
        calendar_df: Calendar dataframe with 'enoch_year' and Day 1 flags
        start_year: Starting year CE (negative for BCE)
        use_terrestrial_time: Use TT instead of UT (recommended)
        use_actual_equinox: Use actual vernal equinox JD (True) or March 20 approximation (False)
        verbose: Print progress

    Returns:
        DataFrame with year, JD, longitude, deviation for each Day 1 event
    """
    if not HAS_SWISSEPH:
        raise ImportError("pyswisseph required for ephemeris calculations")

    # Get Day 1 events (solar-corrected)
    day1_events = calendar_df[calendar_df['is_day1_solar']].copy()

    if verbose:
        print(f"Calculating solar positions for {len(day1_events)} Day 1 events...")
        print(f"Start year: {start_year} CE")
        print(f"Time system: {'Terrestrial Time (TT)' if use_terrestrial_time else 'Universal Time (UT)'}")
        print(f"Base epoch: {'Actual vernal equinox' if use_actual_equinox else 'March 20 approximation'}")

    results = []

    # Calculate base Julian Date for start year
    if use_actual_equinox:
        if verbose:
            print(f"Calculating actual vernal equinox for year {start_year}...")
        base_jd = calculate_vernal_equinox_jd(start_year, use_terrestrial_time)
        if verbose:
            print(f"  Equinox JD: {base_jd:.6f}")
    else:
        base_jd = gregorian_to_julian_date(start_year, 3, 20, 12.0)

    for idx, row in day1_events.iterrows():
        enoch_year = row['enoch_year']
        daycount = row['daycount']

        # Julian Date for this event (approximate)
        # Each Enoch year ≈ 364 days, but actual spacing determined by calendar
        jd = base_jd + daycount

        try:
            # Calculate solar position
            longitude, distance, speed = calculate_solar_longitude(jd, use_terrestrial_time)

            # Calculate deviation from vernal equinox
            deviation = calculate_deviation_from_equinox(longitude)

            # Calendar year (CE)
            calendar_year = start_year + enoch_year

            results.append({
                'enoch_year': enoch_year,
                'calendar_year': calendar_year,
                'daycount': daycount,
                'jd': jd,
                'longitude': longitude,
                'deviation': deviation,
                'distance_au': distance,
                'speed_deg_per_day': speed
            })

            if verbose and len(results) % 1000 == 0:
                print(f"  Progress: {len(results)}/{len(day1_events)} ({100*len(results)/len(day1_events):.1f}%)")

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed at enoch_year={enoch_year}, JD={jd}: {e}")
            continue

    if verbose:
        print(f"  Done! Calculated {len(results)} solar positions.")

    return pd.DataFrame(results)


def analyze_bounded_oscillation(timeseries_df: pd.DataFrame) -> dict:
    """
    Analyze bounded oscillation characteristics from timeseries.

    Args:
        timeseries_df: DataFrame with 'deviation' column

    Returns:
        dict with oscillation statistics
    """
    deviations = timeseries_df['deviation'].values

    # Amplitude statistics
    amplitude_min = deviations.min()
    amplitude_max = deviations.max()
    amplitude_peak_to_peak = amplitude_max - amplitude_min
    amplitude_mean = deviations.mean()
    amplitude_std = deviations.std()

    # Linear regression for drift rate
    years = np.arange(len(deviations))
    coeffs = np.polyfit(years, deviations, 1)
    drift_rate = coeffs[0]  # degrees per year

    # Precision windows
    total = len(deviations)
    within_3deg = np.sum(np.abs(deviations) <= 3) / total * 100
    within_1deg = np.sum(np.abs(deviations) <= 1) / total * 100
    within_05deg = np.sum(np.abs(deviations) <= 0.5) / total * 100

    return {
        'n_points': total,
        'amplitude_min': amplitude_min,
        'amplitude_max': amplitude_max,
        'amplitude_peak_to_peak': amplitude_peak_to_peak,
        'amplitude_mean': amplitude_mean,
        'amplitude_std': amplitude_std,
        'drift_rate_deg_per_year': drift_rate,
        'drift_rate_deg_per_millennium': drift_rate * 1000,
        'coverage_3deg_pct': within_3deg,
        'coverage_1deg_pct': within_1deg,
        'coverage_05deg_pct': within_05deg,
    }


if __name__ == '__main__':
    # Demo: Calculate solar position for a specific date
    if HAS_SWISSEPH:
        print("=" * 70)
        print("EPHEMERIS DEMO: Solar Position Calculation")
        print("=" * 70)
        print()

        # Calculate for vernal equinox 2000 CE
        year = 2000
        jd = gregorian_to_julian_date(year, 3, 20, 12.0)
        print(f"Date: March 20, {year} (noon)")
        print(f"Julian Date: {jd:.2f}")
        print()

        longitude, distance, speed = calculate_solar_longitude(jd, use_terrestrial_time=True)
        deviation = calculate_deviation_from_equinox(longitude)

        print(f"Solar ecliptic longitude: {longitude:.4f}°")
        print(f"Deviation from equinox (0°): {deviation:.4f}°")
        print(f"Sun-Earth distance: {distance:.6f} AU")
        print(f"Longitude speed: {speed:.6f} deg/day")
        print()
        print("=" * 70)
    else:
        print("Install pyswisseph to run ephemeris demo:")
        print("  pip install pyswisseph")


def get_solar_ecliptic_longitude(jd: float) -> float:
    """
    Get solar ecliptic longitude at given Julian Date.
    
    Wrapper function for notebooks that need simple longitude retrieval.
    
    Args:
        jd: Julian Date
        
    Returns:
        Solar ecliptic longitude in degrees (0-360)
    """
    lon, dist, speed = calculate_solar_longitude(jd, use_terrestrial_time=True)
    return lon
