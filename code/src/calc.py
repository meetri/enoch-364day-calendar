import numpy as np
from scipy.signal import argrelextrema

# wednesday == 3
weekname_lookup = [
    "sunday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday"]

priestsQumran = [
    "Gamul",     # Week 1
    "Delaiah",   # Week 2
    "Maaziah",   # Week 3
    "Jehoiarib",  # Week 4
    "Jedaiah",   # Week 5
    "Harim",     # Week 6
    "Seorim",    # Week 7
    "Malchijah",  # Week 8
    "Mijamin",   # Week 9
    "Hakkoz",    # Week 10
    "Abijah",    # Week 11
    "Jeshua",    # Week 12
    "Shecaniah",  # Week 13
    "Eliashib",  # Week 14
    "Jakim",     # Week 15
    "Huppah",    # Week 16
    "Jeshebeab",  # Week 17
    "Bilgaha",   # Week 18
    "Immer",     # Week 19
    "Hezir",     # Week 20
    "Happizzez",  # Week 21
    "Pethahiah",  # Week 22
    "Jehezkel",  # Week 23
    "Jachin"     # Week 24
]

monthDays = [
    0,    # Month 1 | Portal 4 | Spring
    30,   # Month 2 | Portal 3
    60,   # Month 3 | Portal 2
    91,   # Month 4 | Portal 1 | Summer
    121,  # Month 5 | Portal -1
    151,  # Month 6 | Portal -2
    182,  # Month 7 | Portal -3
    212,  # Month 8 | Portal -4
    242,  # Month 9 | Portal -5
    273,  # Month 10 | Portal -6 | Winter
    303,  # Month 11 | Portal 6
    333   # Month 12 | Portal 5
]

# Enoch month number (1-12) and day using reverse iteration algorithm


def get_enoch_month_day(day_in_year):
    # Find the appropriate month by checking portal days in reverse
    for i in range(11, -1, -1):  # 11 down to 0 (reverse iteration)
        if day_in_year >= monthDays[i]:
            month = i + 1
            day = day_in_year - monthDays[i] + 1
            return month, day
    # Fallback for edge cases
    return 1, 1


def start_enoch_day_count(nm_df, start_idx):
    nm_df['cumulative_days'] = 0
    nm_df.loc[start_idx:,
              'total_days'] = nm_df.loc[start_idx:,
                                        'lunar_days_in_month'].cumsum()
    nm_df.loc[start_idx:,
              'enoch_day'] = nm_df.loc[start_idx:,
                                       'total_days'] % 364
    return nm_df[start_idx:]


def count_in_between(df, flagname_column, count_column_name):
    # Get boolean mask of flag positions
    flag_mask = df[flagname_column]

    # Create group identifier that changes at each True flag
    group_id = flag_mask.cumsum()

    # Group by this ID and create incremental counter within each group
    df[count_column_name] = df.groupby(group_id).cumcount() + 1
    return df


def calculate_local_minima(
        nm_df,
        tag="minor_lunar_cycle",
        nm_eq_label="last_eq_to_nm"):
    # Create copy of original data and initialize minor_lunar_cycle column
    result_df = nm_df.copy()
    result_df[tag] = False

    # Filter all rows where the lunar_month = 1 and closest to the vernal equinox.
    # (this will be used in defining the lunar cycle)
    # mask = nm_df["lunar_month"] == 1
    # month1_df = nm_df[mask].copy()
    month1_df = nm_df.copy()
    ilocs_min = argrelextrema(
        month1_df[nm_eq_label].values,
        np.less_equal)[0]  # [::9]

    # Get the actual indices of minima in the original DataFrame
    minima_indices = month1_df.iloc[ilocs_min].index[1:]

    # Overlay minor lunar cycle flags back onto full DataFrame
    result_df.loc[minima_indices, tag] = True
    return result_df
    # df = calculate_cycle_days_between(result_df, "minor_lunar_cycle")
    # return fill_cycle_total_days(df, "minor_lunar_cycle")
