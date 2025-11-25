"""
Enoch Calendar Generation with 294-Day Correction Mechanism

This module generates the 364-day Enochic calendar with systematic
backward shifts every 294 days to maintain solar alignment.

Calendar structure:
- 364 days per year (52 weeks × 7 days)
- 4 quarters of 91 days each (13 weeks)
- Day 1 = vernal equinox

Correction mechanism:
- Every 294 days, shift calendar reference frame backward by 1 day
- Over 294 years: 364 total shifts = one complete year removed
- Effective period: 293 tropical years ≈ 294 calendar years
"""

import numpy as np
import pandas as pd
from typing import Tuple


# Calendar constants
ENOCH_YEAR = 364  # Days per calendar year
CORRECTION_PERIOD = 294  # Days between corrections
ENOCH_QUARTER = 91  # Days per quarter (13 weeks)
ENOCH_WEEK = 7  # Days per week


def generate_calendar_frame(num_cycles: int = 100,
                            correction_period: int = CORRECTION_PERIOD) -> pd.DataFrame:
    """
    Generate Enoch calendar with 294-day correction mechanism.

    IMPORTANT: Day 1 (vernal equinox) falls on Wednesday (day-of-week 4).
    This is achieved via the +3 offset in week calculations.

    Args:
        num_cycles: Number of 294-year cycles to generate
        correction_period: Days between corrections (default: 294)

    Returns:
        DataFrame with calendar structure and solar corrections
    """
    total_days = ENOCH_YEAR * correction_period * num_cycles

    df = pd.DataFrame()
    df['daycount'] = np.arange(total_days)

    # Calendar year (0-indexed)
    df['enoch_year'] = df['daycount'] // ENOCH_YEAR

    # Day of year (1-364)
    df['enoch_doy'] = (df['daycount'] % ENOCH_YEAR) + 1

    # Quarter (1-4) and day within quarter (1-91)
    df['quarter'] = ((df['enoch_doy'] - 1) // ENOCH_QUARTER) + 1
    df['quarter_day'] = ((df['enoch_doy'] - 1) % ENOCH_QUARTER) + 1

    # Week calculations with +3 offset
    # This makes Day 1 (daycount=0) fall on Wednesday (day 4)
    # Without offset: daycount=0 → Sunday (day 1)
    # With +3 offset: daycount=0 → (0+3)%7+1 = 4 = Wednesday ✓
    df['week_index'] = (df['daycount'] + 3) // ENOCH_WEEK
    df['week_of_year'] = ((df['enoch_doy'] - 1) // ENOCH_WEEK) + 1
    df['day_of_week'] = ((df['daycount'] + 3) % ENOCH_WEEK) + 1

    # Solar correction: backward shift every 294 days
    # Number of shifts that have occurred by this day (with +3 offset)
    df['correction_shifts'] = (df['daycount'] + 3) // correction_period

    # Solar-corrected day of year (with wraparound)
    # Shift = number of corrections mod 364 (wraps around each year)
    df['solar_shift'] = df['correction_shifts'] % ENOCH_YEAR
    df['solar_doy'] = ((df['enoch_doy'] - df['solar_shift'] - 1) % ENOCH_YEAR) + 1

    # Flag Day 1 events (for solar position calculations)
    df['is_day1_calendar'] = df['enoch_doy'] == 1
    df['is_day1_solar'] = df['solar_doy'] == 1

    return df


def get_day1_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract Day 1 events from calendar dataframe.

    Args:
        df: Calendar dataframe from generate_calendar_frame()

    Returns:
        Tuple of (calendar_day1, solar_day1) dataframes
    """
    calendar_day1 = df[df['is_day1_calendar']].copy()
    solar_day1 = df[df['is_day1_solar']].copy()

    return calendar_day1, solar_day1


def calculate_effective_tropical_year(correction_period: int = CORRECTION_PERIOD,
                                      calendar_year: int = ENOCH_YEAR) -> float:
    """
    Calculate the effective tropical year from correction mechanism.

    The mechanism removes one day every correction_period days, yielding:
    T_effective = (correction_period × calendar_year) / (correction_period - 1)

    For Earth (294-day correction):
    T_effective = (294 × 364) / 293 = 365.2423 days

    Args:
        correction_period: Days between corrections
        calendar_year: Calendar year length

    Returns:
        Effective tropical year in days
    """
    if correction_period <= 1:
        return float('inf')

    return (correction_period * calendar_year) / (correction_period - 1)


def validate_correction_mechanism(correction_period: int = CORRECTION_PERIOD,
                                  calendar_year: int = ENOCH_YEAR,
                                  earth_tropical_year: float = 365.24219) -> dict:
    """
    Validate the correction mechanism mathematics.

    Args:
        correction_period: Days between corrections
        calendar_year: Calendar year length
        earth_tropical_year: Actual Earth tropical year

    Returns:
        dict with validation metrics
    """
    # Calculate effective tropical year
    t_effective = calculate_effective_tropical_year(correction_period, calendar_year)

    # Error relative to actual tropical year
    error_days = t_effective - earth_tropical_year
    error_pct = 100 * error_days / earth_tropical_year

    # Number of years in one complete cycle
    cycle_years = correction_period

    # Number of corrections in one cycle
    num_corrections = (cycle_years * calendar_year) // correction_period

    # Total days removed in one cycle
    days_removed = num_corrections

    return {
        'correction_period': correction_period,
        'calendar_year': calendar_year,
        'effective_tropical_year': t_effective,
        'actual_tropical_year': earth_tropical_year,
        'error_days': error_days,
        'error_pct': error_pct,
        'cycle_years': cycle_years,
        'corrections_per_cycle': num_corrections,
        'days_removed_per_cycle': days_removed,
        'precision_decimal_places': int(-np.log10(abs(error_days))) if abs(error_days) > 0 else 10
    }


def generate_enoch_calendar(start_year: int,
                           end_year: int,
                           year_length_days: int = ENOCH_YEAR,
                           correction_cycle_years: int = CORRECTION_PERIOD) -> pd.DataFrame:
    """
    Generate Enoch calendar with Julian Dates for specified year range.

    This is a wrapper function for notebooks that need calendar generation
    with custom parameters and JD values for Day 1 events.

    Args:
        start_year: Starting calendar year (CE)
        end_year: Ending calendar year (CE, inclusive)
        year_length_days: Length of calendar year in days (default: 364)
        correction_cycle_years: Correction cycle in years (default: 294)

    Returns:
        DataFrame with columns:
            - calendar_year: Year number (CE)
            - jd_day1: Julian Date for Day 1 of each year
            - enoch_doy: Day of year (1-364)
    """
    # Calculate number of years
    n_years = end_year - start_year + 1

    # Reference: J2000.0 = JD 2451545.0 = 2000-01-01 12:00 TT
    # Vernal equinox 2000 ≈ 2000-03-20 07:35 UTC ≈ JD 2451623.8
    jd_reference = 2451623.8  # Approximate vernal equinox 2000
    year_reference = 2000

    # Calculate effective tropical year
    eff_trop_year = (year_length_days * correction_cycle_years) / (correction_cycle_years - 1)

    # Generate calendar years
    calendar_years = np.arange(start_year, end_year + 1)

    # Calculate JD for Day 1 of each year
    # Days from reference year
    years_from_ref = calendar_years - year_reference
    jd_day1 = jd_reference + years_from_ref * eff_trop_year

    # Create DataFrame
    df = pd.DataFrame({
        'calendar_year': calendar_years,
        'jd_day1': jd_day1
    })

    return df


if __name__ == '__main__':
    # Demo: Show the correction mechanism for Earth
    print("=" * 70)
    print("ENOCH CALENDAR CORRECTION MECHANISM")
    print("=" * 70)
    print()

    validation = validate_correction_mechanism()

    print(f"Calendar structure: {validation['calendar_year']} days (52 weeks × 7 days)")
    print(f"Correction period: {validation['correction_period']} days (42 weeks)")
    print()
    print(f"Effective tropical year: {validation['effective_tropical_year']:.6f} days")
    print(f"Actual Earth tropical year: {validation['actual_tropical_year']} days")
    print(f"Error: {validation['error_days']:.6f} days ({validation['error_pct']:.4f}%)")
    print(f"Precision: {validation['precision_decimal_places']} decimal places")
    print()
    print(f"One complete cycle: {validation['cycle_years']} calendar years")
    print(f"Corrections per cycle: {validation['corrections_per_cycle']}")
    print(f"Days removed per cycle: {validation['days_removed_per_cycle']}")
    print()
    print("=" * 70)
