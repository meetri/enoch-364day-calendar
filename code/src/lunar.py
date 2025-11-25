import math
import pandas as pd
import numpy as np
from .calc import calculate_local_minima, count_in_between

SYNODIC = 29.530588861  # days
J0 = 2451550.09765  # JDE of a known new moon (2000-01-06 18:14 UT)


def _mod360(x):
    return x % 360.0


def _signed(a, b):
    """Smallest signed angle a-b in (-180, +180]"""
    return (a - b + 180.0) % 360.0 - 180.0


def _sun_moon_lon_speed(swe, jd, flags):
    """
    Return (λ_sun, ω_sun, λ_moon, ω_moon).
    NOTE: longitude at [0], speed in longitude at [3]!
    """
    sun = swe.calc_ut(jd, swe.SUN, flags=flags | swe.FLG_SPEED)[0]
    moon = swe.calc_ut(jd, swe.MOON, flags=flags | swe.FLG_SPEED)[0]
    ls, vs = _mod360(sun[0]), sun[3]   # deg, deg/day
    lm, vm = _mod360(moon[0]), moon[3]  # deg, deg/day
    return ls, vs, lm, vm


def find_new_moons_fast(
        swe,
        start_jd,
        end_jd,
        tol_minutes=0.1,
        flags=None,
        max_iter=12):
    """
    Fast new-moon finder using Newton on Moon-Sun elongation with correct
    speeds. Returns list of JD(UT) new-moon times within [start_jd, end_jd].
    """
    if flags is None:
        flags = getattr(swe, "FLG_JPLEPH", 0)

    # Convert time tolerance to an equivalent angle tolerance (very small)
    tol_days = tol_minutes / (24.0 * 60.0)
    tol_deg = tol_days * (360.0 / SYNODIC)

    res = []

    # Slightly extend k-range to avoid edge misses after Newton refinement
    k_start = math.floor((start_jd - J0) / SYNODIC) - 2
    k_end = math.ceil((end_jd - J0) / SYNODIC) + 2

    for k in range(k_start, k_end + 1):
        jd = J0 + k * SYNODIC  # seed (already close to a new moon in TT/UT)

        # Newton iterations on f(jd) = λ_moon - λ_sun = 0
        for _ in range(max_iter):
            ls, vs, lm, vm = _sun_moon_lon_speed(swe, jd, flags)
            f = _signed(lm, ls)          # deg
            df = (vm - vs)                # deg/day (relative angular speed)

            # Guard against pathological derivative (shouldn't happen in
            # practice)
            if abs(df) < 1e-6:
                break

            step = f / df                 # days
            jd -= step

            if abs(f) < tol_deg or abs(step) < tol_days:
                break

        if start_jd <= jd <= end_jd:
            res.append(jd)

    # Sort & unique just in case (near boundaries)
    res = sorted(set(res))
    return res


def add_lunar_cycles(nm_df):
    # group into lunar months
    lunar_year_cycle_df = calculate_local_minima(
        nm_df, "lunar_year_cycle", "last_eq_to_nm")

    # group into lunar years
    lunar_minor_cycle_df = calculate_local_minima(
        lunar_year_cycle_df[
            lunar_year_cycle_df.lunar_year_cycle
        ], "lunar_minor_cycle", "last_eq_to_nm"
    )

    lunar_year_cycle_indices = lunar_year_cycle_df[
        lunar_year_cycle_df['lunar_year_cycle']
    ].index

    minor_cycle_indices = lunar_minor_cycle_df[
        lunar_minor_cycle_df['lunar_minor_cycle']
    ].index

    nm_df['lunar_year_length'] = np.nan
    nm_df["lunar_year_cycle"] = False
    nm_df["lunar_minor_cycle"] = False
    nm_df["lunar_major_cycle"] = False
    nm_df["lunar_grand_cycle"] = False

    major_cycle_indices = lunar_minor_cycle_df[lunar_minor_cycle_df['lunar_minor_cycle']][::12].index
    grand_cycle_indices = lunar_minor_cycle_df[lunar_minor_cycle_df['lunar_minor_cycle']][::108].index

    nm_df.loc[lunar_year_cycle_indices, 'lunar_year_cycle'] = True
    nm_df.loc[lunar_year_cycle_indices, 'lunar_year_cycle'] = True
    nm_df.loc[minor_cycle_indices, 'lunar_minor_cycle'] = True
    nm_df.loc[major_cycle_indices, 'lunar_major_cycle'] = True
    nm_df.loc[grand_cycle_indices, 'lunar_grand_cycle'] = True

    year_mask = nm_df['lunar_year_cycle']

    nm_df.loc[year_mask,
              'lunar_year_length'] = nm_df.loc[year_mask,
                                               'nm_jd'].diff().shift(-1)
    nm_df['lunar_year_length'] = nm_df['lunar_year_length'].ffill()

    nm_df = count_in_between(nm_df, "lunar_year_cycle", "lunar_month")
    nm_df = count_in_between(nm_df, "lunar_minor_cycle", "lunar_minor_months")
    nm_df = count_in_between(nm_df, "lunar_major_cycle", "lunar_major_months")
    nm_df = count_in_between(nm_df, "lunar_grand_cycle", "lunar_grand_months")

    return nm_df


def calculate_lunar_month_years(
        swe,
        year_start,
        num_years,
        use_tt=True,
        debug=False):
    """
    Extended version of calculate_lunar_month_years with proper time scale handling
    for accurate long-term calculations.

    Parameters:
    - swe: Swiss Ephemeris object
    - year_start: Starting year for calculations
    - num_years: Number of years to calculate
    - use_tt: If True, use Terrestrial Time (TT) for astronomical accuracy.
              If False, use Universal Time (UT) for Earth rotation time.
    - debug: If True, include additional debugging information in output

    Returns:
    - DataFrame with lunar and equinox data, including time scale corrections
    """
    nm = []
    last_nm = None

    jd_start = swe.julday(year_start, 1, 1)

    # Calculate initial equinoxes
    if use_tt:
        # For TT calculations, we need to correct the equinox times
        last_eq_ut = swe.solcross_ut(
            x2cross=0, tjdut=swe.julday(
                year_start - 1, 1, 1))
        delta_t_last = swe.deltat(last_eq_ut)
        last_eq = last_eq_ut + delta_t_last  # Convert to TT

        next_eq_ut = swe.solcross_ut(
            x2cross=0, tjdut=swe.julday(
                year_start, 1, 1))
        delta_t_next = swe.deltat(next_eq_ut)
        next_eq = next_eq_ut + delta_t_next  # Convert to TT
    else:
        last_eq = swe.solcross_ut(
            x2cross=0, tjdut=swe.julday(
                year_start - 1, 1, 1))
        next_eq = swe.solcross_ut(
            x2cross=0, tjdut=swe.julday(
                year_start, 1, 1))

    for jyear in range(0, num_years):
        nm_start = jd_start
        nm_end = jd_start + 360  # lunar month is always 12 months ie. 354 days

        # Find new moons (these are already in UT)
        nm_conj = find_new_moons_fast(swe, nm_start, nm_end)

        for idx, m in enumerate(nm_conj):
            # Date of visible crescent moon
            nm_jd_ut = m + 1

            # Convert new moon time to appropriate time scale
            if use_tt:
                delta_t_nm = swe.deltat(nm_jd_ut)
                nm_jd = nm_jd_ut + delta_t_nm  # Convert to TT
            else:
                nm_jd = nm_jd_ut

            # Check if we need to update equinox
            if nm_jd > next_eq:
                last_eq = next_eq
                if use_tt:
                    # Calculate next equinox in TT
                    next_eq_ut = swe.solcross_ut(
                        x2cross=0, tjdut=next_eq_ut + 1)
                    delta_t_next = swe.deltat(next_eq_ut)
                    next_eq = next_eq_ut + delta_t_next
                else:
                    next_eq = swe.solcross_ut(x2cross=0, tjdut=next_eq + 1)

            # Calculate days in month
            if last_nm:
                days_in_month = nm_jd - last_nm
            else:
                if idx + 1 < len(nm_conj):
                    next_nm_ut = nm_conj[idx + 1] + 1
                    if use_tt:
                        delta_t_next_nm = swe.deltat(next_nm_ut)
                        next_nm = next_nm_ut + delta_t_next_nm
                    else:
                        next_nm = next_nm_ut
                    days_in_month = next_nm - nm_jd
                else:
                    days_in_month = 29.5  # Default approximate value

            last_nm = nm_jd

            # Get calendar dates for display
            # Always use UT for calendar display to match Earth rotation
            eq_str_ut = swe.revjul(nm_jd_ut, swe.GREG_CAL)

            # Calculate both UT and TT dates for equinoxes if in debug mode
            if debug:
                if use_tt:
                    # We have TT, calculate back to UT for comparison
                    last_eq_ut = last_eq - swe.deltat(last_eq)
                    next_eq_ut = next_eq - swe.deltat(next_eq)
                else:
                    # We have UT, calculate forward to TT for comparison
                    last_eq_tt = last_eq + swe.deltat(last_eq)
                    next_eq_tt = next_eq + swe.deltat(next_eq)

                last_eq_date_ut = swe.revjul(
                    last_eq_ut if use_tt else last_eq, swe.GREG_CAL)
                last_eq_date_tt = swe.revjul(
                    last_eq if use_tt else last_eq_tt, swe.GREG_CAL)
                next_eq_date_ut = swe.revjul(
                    next_eq_ut if use_tt else next_eq, swe.GREG_CAL)
                next_eq_date_tt = swe.revjul(
                    next_eq if use_tt else next_eq_tt, swe.GREG_CAL)

            # Calculate moon's distance from node at new moon
            moon_pos = swe.calc_ut(nm_jd_ut, swe.MOON)[0][0]
            moon_node = swe.calc_ut(nm_jd_ut, swe.MEAN_NODE)[0][0]
            moon_to_node = abs(moon_pos - moon_node)
            if moon_to_node > 180:
                moon_to_node = 360 - moon_to_node

            data_point = {
                "last_eq_jd": last_eq,
                "next_eq_jd": next_eq,
                "nm_jd": nm_jd,
                "nm_jd_ut": nm_jd_ut,  # Always include UT version for reference
                "nm_date": f"{eq_str_ut[1]}/{eq_str_ut[2]}/{eq_str_ut[0]}",
                "nm_to_next_eq": next_eq - nm_jd,
                "last_eq_to_nm": nm_jd - last_eq,
                "sun_year": eq_str_ut[0],
                "lunar_days_in_month": days_in_month,
                # Use UT for day of week
                "day_of_week": swe.day_of_week(nm_jd_ut),
                "time_scale": "TT" if use_tt else "UT",
                "moon_to_node_deg": moon_to_node
            }

            # Add debug information if requested
            if debug:
                data_point.update({
                    "delta_t_days": swe.deltat(nm_jd_ut) / 86400.0 if use_tt else 0,
                    "last_eq_ut_date": f"{last_eq_date_ut[1]}/{last_eq_date_ut[2]}",
                    "last_eq_tt_date": f"{last_eq_date_tt[1]}/{last_eq_date_tt[2]}",
                    "next_eq_ut_date": f"{next_eq_date_ut[1]}/{next_eq_date_ut[2]}",
                    "next_eq_tt_date": f"{next_eq_date_tt[1]}/{next_eq_date_tt[2]}"
                })

            nm.append(data_point)

        jd_start = nm_end

    df = pd.DataFrame(nm)
    return df


def merge_astronomic_data(
        df,
        YEAR_START,
        use_tt=True,
        get_ecliptic_longitude=False):
    """Attach new-moon/equinox flags and Julian days to the daily frame."""
    total_years = int(np.ceil(df.shape[0] / 364))

    # Generate the lunar/equinox table and skip the pre-March rows
    nm_df = calculate_lunar_month_years(
        swe, YEAR_START, total_years, use_tt=use_tt)
    nm_df = add_lunar_cycles(nm_df)
    nm_df = nm_df.iloc[2:].reset_index(drop=True)

    # Anchor the synthetic daycount to the first astronomical JD
    base_jd = nm_df["nm_jd"].iloc[0]
    result = df.copy()
    result["jd_noon"] = (int(base_jd) + result["daycount"]).astype(float)

    # Sunrise-based days: Day starts at sunrise (~6:00 local time)
    # Jerusalem/Qumran sunrise averages ~4:00 UTC (winter) to ~3:00 UTC (summer)
    MIDDLE_EAST_SUNRISE_OFFSET = -(35.0 / 360.0 + 6.0 / 24.0)  # -0.347 days
    result["jd"] = result["jd_noon"] + MIDDLE_EAST_SUNRISE_OFFSET

    # Prepare the event lists we’ll merge as-of (by nearest JD)
    nm_events = (
        nm_df.loc[:, ["nm_jd"]]
        .dropna()
        .drop_duplicates()
        .sort_values("nm_jd")
        .reset_index(drop=True)
    )
    eq_events = (
        nm_df.loc[:, ["last_eq_jd"]]
        .dropna()
        .drop_duplicates()
        .sort_values("last_eq_jd")
        .rename(columns={"last_eq_jd": "eq_jd"})
        .reset_index(drop=True)
    )

    # Sort the daily rows on JD so merge_asof works, but hang onto
    # the original row order under the “index” column
    days = result.sort_values("jd").reset_index()

    # Attach the closest new moon within ±0.5 day (about 12 hours)
    days = pd.merge_asof(
        days,
        nm_events,
        left_on="jd",
        right_on="nm_jd",
        direction="nearest",
        tolerance=0.5,
    )
    days["has_nm"] = days["nm_jd"].notna()

    # Attach the closest equinox within the same tolerance
    days = pd.merge_asof(
        days,
        eq_events,
        left_on="jd",
        right_on="eq_jd",
        direction="nearest",
        tolerance=0.5,
    )
    days["has_eq"] = days["eq_jd"].notna()

    # Restore the original row order and return the enriched frame
    result = (
        days.sort_values("index")
        .drop(columns="index")
        .reset_index(drop=True)
    )

    result = count_in_between(result, "has_nm", "lunar_month_day")
    lunar_month_count = result['has_nm'].cumsum()
    result['lunar_month'] = ((lunar_month_count + 1) % 12) + 1

    result = count_in_between(result, "has_eq", "solar_day_in_year")
    solar_year_count = result['has_eq'].cumsum()
    result['solar_year'] = solar_year_count + YEAR_START - 1

    if get_ecliptic_longitude:
        # Initialize solar position columns (populated by enoch.merge_astronomic_data)
        result['sun_ecliptic_longitude'] = np.nan
        result['sun_distance_au'] = np.nan

        # Convert ecliptic longitude to signed range [-180, 180] for visualization
        result["sun_ecliptic_longitude_neg"] = np.where(
            result["sun_ecliptic_longitude"] <= 180,
            result["sun_ecliptic_longitude"],
            result["sun_ecliptic_longitude"] - 360
        )

    return result
