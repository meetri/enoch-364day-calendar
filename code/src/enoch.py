import swisseph as swe
import pandas as pd
import numpy as np
from .lunar import add_lunar_cycles, calculate_lunar_month_years
from .calc import count_in_between, get_enoch_month_day


def merge_astronomic_data(df, YEAR_START, use_tt=True):

    # TODO: get solar equinox YEAR_START directly
    nm_df = calculate_lunar_month_years(swe, YEAR_START, 1, use_tt=use_tt)

    base_jd = nm_df["last_eq_jd"].iloc[0]
    result = df.copy()

    # Handle windback: if enoch_year_start provided, adjust daycount offset
    result["jd_noon"] = (int(base_jd) + result["daycount"]).astype(float)

    # Sunrise-based days:
    # Day starts at sunrise (~6:00 local time)
    # Jerusalem/Qumran sunrise averages ~4:00 UTC (winter) to ~3:00 UTC (summer)
    MIDDLE_EAST_SUNRISE_OFFSET = -(35.0/360.0 + 6.0/24.0)  # -0.347 days
    result["jd"] = result["jd_noon"] + MIDDLE_EAST_SUNRISE_OFFSET

    # ======================================================================
    # Calculate solar position for Enoch Day 1 and Solar Day 1
    # This shows Sun's ecliptic longitude at year starts (raw and corrected)
    mask = (result['enoch_doy'] == 1) | (result['enoch_solar_doy'] == 1)

    # Initialize columns with NaN
    result['sun_ecliptic_longitude'] = np.nan
    result['sun_distance_au'] = np.nan

    # Filter to valid JD range (ephemeris coverage starts at base_jd)
    valid_indices = result[mask].index

    # Build lists for valid rows (efficient for large datasets)
    sun_longitudes = []
    sun_distances = []

    # Use appropriate time system based on use_tt flag
    calc_func = swe.calc if use_tt else swe.calc_ut

    # Calculate only for valid JD range (no error handling needed)
    for jd in result.loc[valid_indices, 'jd']:
        sun_data = calc_func(jd, swe.SUN, 0)
        sun_longitudes.append(sun_data[0][0])  # longitude
        sun_distances.append(sun_data[0][2])   # distance

    result.loc[valid_indices, 'sun_ecliptic_longitude'] = sun_longitudes
    result.loc[valid_indices, 'sun_distance_au'] = sun_distances

    # Calculate signed angular distance from 0° (vernal equinox)
    # Converts 350° → -10°, keeps 10° → +10°
    # Range: [-180, +180] where negative = clockwise from 0°
    def signed_angular_distance(longitude):
        """Convert ecliptic longitude to signed distance from 0° (vernal equinox)"""
        longitude = longitude % 360
        if longitude > 180:
            return longitude - 360
        else:
            return longitude

    # Convert ecliptic longitude to signed range [-180, 180] for easier visualization
    # Values > 180 (e.g. 350°) become negative (e.g. -10°)
    result["sun_ecliptic_longitude_neg"] = np.where(
        result["sun_ecliptic_longitude"] <= 180,
        result["sun_ecliptic_longitude"],
        result["sun_ecliptic_longitude"] - 360
    )

    return result


def enoch_calendar_frame(num_cycles=1, correction_cycle=294):
    df = pd.DataFrame()
    df["daycount"] = range(0, 364 * correction_cycle * num_cycles)
    df["enoch_year"] = df["daycount"] // 364

    df["week_index"] = (df["daycount"] + 3) // 7
    df["enoch_dow"] = ((df["daycount"] + 3) % 7) + 1

    df["enoch_doy"] = (df["daycount"] % 364) + 1

    # Calculate month and day from enoch_doy (1-364)
    month_day_results = df["enoch_doy"].apply(lambda d: get_enoch_month_day(d - 1))
    df["enoch_month"] = [m for m, d in month_day_results]
    df["enoch_day"] = [d for m, d in month_day_results]

    shifts = (df["daycount"] + 3) // correction_cycle
    df["enoch_solar_shift"] = shifts % 364

    # Calculate solar-corrected day of year (1-364, no negatives)
    mod = (df["enoch_doy"] - df["enoch_solar_shift"] - 1) % 364 + 1
    df["enoch_solar_doy"] = mod
    df["enoch_solar_neg_doy"] = np.where(mod <= 182, mod - 1, mod - 365)

    mod2 = df["enoch_doy"]
    df["enoch_neg_doy"] = np.where(mod2 <= 182, mod2 - 1, mod2 - 365)

    # Calculate month and day from enoch_solar_doy (1-364)
    solar_month_day_results = df["enoch_solar_doy"].apply(lambda d: get_enoch_month_day(d - 1))
    df["enoch_solar_month"] = [m for m, d in solar_month_day_results]
    df["enoch_solar_day"] = [d for m, d in solar_month_day_results]

    df["enoch_priest"] = (df["week_index"] % 24) + 1
    df["enoch_priest_year"] = ((df["week_index"] // 52) % 6) + 1
    df["enoch_priest_jubilee"] = ((df["week_index"] // (24 * 13)) % 49) + 1
    df[f"{correction_cycle}_day"] = (((df["daycount"] + 3) % correction_cycle)) + 1
    df[f"{correction_cycle}_week"] = ((((df["daycount"] + 3) % (correction_cycle * 7))) // 7) + 1
    df[f"{correction_cycle}_year"] = ((((df["daycount"] + 3) % (correction_cycle * 52 * 7))) // 364) + 1

    return df


def enoch_calendar_flexible(
    num_days=1, correction_cycle=294, calendar_year=364, vernal_offset=0
):
    """
    Generalized Enoch calendar with flexible parameters for sensitivity testing.

    Unlike enoch_calendar_frame(), this function allows variation in core design
    parameters to test structural sensitivity of the calendar mechanism.

    For astronomical validation with actual sky positions, use enoch_calendar_frame()
    with merge_astronomic_data(). For sensitivity testing and parameter exploration,
    use this function.

    Args:
        num_cycles: Number of correction cycles to generate
        correction_cycle: Days between corrections (correction period, default: 294)
        calendar_year: Days in calendar year structure (default: 364)
        vernal_offset: Starting day offset from vernal equinox (default: 0)
                      This tests phase independence of the oscillation

    Returns:
        DataFrame with calendar structure including:
        - daycount: Sequential day counter
        - enoch_year: Calendar year number
        - enoch_doy: Day of year (1 to calendar_year)
        - enoch_solar_shift: Accumulated correction shifts
        - enoch_solar_doy: Solar-corrected day of year
        - Additional calendar structure (weeks, months, priests, etc.)

    Example:
        # Test Mars calendar: 672-day year with 336-day correction
        mars_df = enoch_calendar_flexible(
            num_cycles=100,
            calendar_year=672,
            correction_cycle=336,
            vernal_offset=0
        )

        # Test phase sensitivity: Earth calendar starting at different phases
        phase_test_df = enoch_calendar_flexible(
            num_cycles=100,
            vernal_offset=90  # Start 90 days after vernal equinox
        )
    """
    df = pd.DataFrame()
    df["daycount"] = range(0, num_days)
    df["enoch_year"] = df["daycount"] // calendar_year

    df["week_index"] = (df["daycount"] + 3) // 7
    df["enoch_dow"] = ((df["daycount"] + 3) % 7) + 1

    df["enoch_doy"] = (df["daycount"] % calendar_year) + 1

    # Calculate month and day from enoch_doy (1-calendar_year)
    # Note: get_enoch_month_day assumes 364-day structure
    # For non-364 calendars, this will approximate month boundaries
    month_day_results = df["enoch_doy"].apply(lambda d: get_enoch_month_day((d - 1) % 364))
    df["enoch_month"] = [m for m, d in month_day_results]
    df["enoch_day"] = [d for m, d in month_day_results]

    # Apply correction shifts (with optional vernal offset)
    shifts = (df["daycount"] + 3 + vernal_offset) // correction_cycle
    df["enoch_solar_shift"] = shifts % calendar_year

    # Calculate solar-corrected day of year (1-calendar_year, no negatives)
    mod = (df["enoch_doy"] - df["enoch_solar_shift"] - 1) % calendar_year + 1
    df["enoch_solar_doy"] = mod

    # Calculate negative DOY for visualization (centered around day 1/calendar_year)
    midpoint = calendar_year // 2
    df["enoch_solar_neg_doy"] = np.where(mod <= midpoint, mod - 1, mod - (calendar_year + 1))
    df["enoch_neg_doy"] = np.where(
        df["enoch_doy"] <= midpoint,
        df["enoch_doy"] - 1,
        df["enoch_doy"] - (calendar_year + 1)
    )

    # Calculate month and day from enoch_solar_doy (1-calendar_year)
    solar_month_day_results = df["enoch_solar_doy"].apply(lambda d: get_enoch_month_day((d - 1) % 364))
    df["enoch_solar_month"] = [m for m, d in solar_month_day_results]
    df["enoch_solar_day"] = [d for m, d in solar_month_day_results]

    df["enoch_priest"] = (df["week_index"] % 24) + 1
    df["enoch_priest_year"] = ((df["week_index"] // 52) % 6) + 1
    df["enoch_priest_jubilee"] = ((df["week_index"] // (24 * 13)) % 49) + 1
    df[f"{correction_cycle}_day"] = (((df["daycount"] + 3) % correction_cycle)) + 1
    df[f"{correction_cycle}_week"] = ((((df["daycount"] + 3) % (correction_cycle * 7))) // 7) + 1
    df[f"{correction_cycle}_year"] = ((((df["daycount"] + 3) % (correction_cycle * 52 * 7))) // calendar_year) + 1

    return df
