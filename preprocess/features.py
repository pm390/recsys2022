import datetime
from itertools import compress
from math import sin, cos

import numpy as np
import pandas as pd


# Idee features:
# similarità media degli item nella sessione, similarità dei top 3 item visti più a lungo nella sessione.
# Embeddare in qualche modo le caratteristiche dell'item visto più a lungo/meno a lungo?

# Features su reloaded page + revisited item in the session. Mettere booleano + codice elemento?

# Pulire il dataset dai reload degli item??

def macro_features_generation(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    input_dataframe = remove_reloaded_items(input_dataframe)
    print('completed removal')
    input_dataframe = get_date_features(input_dataframe)
    print('completed date features')
    input_dataframe = get_session_length_features(input_dataframe)
    print('completed session length')
    input_dataframe = get_special_date_features(input_dataframe)
    print('completed special features')
    return input_dataframe


# Remove sequent items if the same item has a delta t of 30 seconds
def remove_reloaded_items(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    input_dataframe[['date', 'item_id']] = input_dataframe[['date', 'item_id']].apply(
        remove_items,
        axis=1,
        result_type="expand"
    )
    return input_dataframe


def remove_items(x):
    boolean_vector = [
        not ((x['item_id'][i] == x['item_id'][i + 1]) and ((x['date'][i + 1] - x['date'][i]).total_seconds() < 30)) for
        i in range(len(x['item_id']) - 1)]

    boolean_vector.append(True)

    filtered_items = list(compress(x['item_id'], boolean_vector))
    filtered_timestamps = list(compress(x['date'], boolean_vector))

    return filtered_timestamps, filtered_items


def get_date_features(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    date_feature_names = [
        'timedelta',
        'date_normalized'
    ]

    input_dataframe["date_0"] = pd.to_datetime(input_dataframe['date'].str[0])
    input_dataframe["date_hour_sin"] = np.sin(input_dataframe["date_0"].dt.hour * np.pi / 30)
    input_dataframe["date_hour_cos"] = np.cos(input_dataframe["date_0"].dt.hour * np.pi / 30)
    input_dataframe["date_day_sin"] = np.sin(input_dataframe["date_0"].dt.hour * np.pi / 15)
    input_dataframe["date_day_cos"] = np.cos(input_dataframe["date_0"].dt.hour * np.pi / 15)
    input_dataframe["date_month_sin"] = np.sin(input_dataframe["date_0"].dt.hour * np.pi / 6)
    input_dataframe["date_month_cos"] = np.cos(input_dataframe["date_0"].dt.hour * np.pi / 6)
    input_dataframe["date_year_2020"] = input_dataframe["date_0"].dt.year == 2020

    input_dataframe[date_feature_names] = input_dataframe[['date']].apply(
        process_timestamps,
        axis=1,
        result_type="expand"
    )
    return input_dataframe


def process_timestamps(x):
    # TODO: insert this in time series features
    x = x[0]
    times = [datetime.minute + datetime.second / 60 for datetime in x]
    times = [time - times[0] for time in times]
    timedelta = [times[index + 1] - times[index] for index in range(len(times) - 1)]
    timedelta.append(np.mean(timedelta) if len(timedelta) > 0 else -1)

    return (
        timedelta,
        times,
    )


def get_session_length_features(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    session_length_feature_names = ['length_of_session_seconds',
                                    'avg_time_spent_per_item_seconds',
                                    'variance_time_spent_per_item_seconds',
                                    'longest_seen_item',
                                    'shortest_seen_item',
                                    'n_seen_items',
                                    'user_went_afk']

    # compute length of sessions in seconds
    input_dataframe[session_length_feature_names] = input_dataframe[['date', 'item_id']].apply(
        compute_lengths,
        axis=1,
        result_type="expand"
    )
    return input_dataframe


def compute_lengths(x):
    session_length_seconds = (x['date'][-1] - x['date'][0]).total_seconds()

    n_seen_items = len(x['item_id'])

    time_deltas_between_items = np.array([(x['date'][i + 1] - x['date'][i]).total_seconds() for i in
                                          range(len(x['date']) - 1)]) if n_seen_items > 1 else np.array([0])

    avg_time_spent_on_item_seconds = session_length_seconds / (len(x['date']) - 1) if n_seen_items > 1 else 0

    variance_time_spent_on_item_seconds = np.var(time_deltas_between_items)

    user_went_afk = any(time_deltas_between_items / 60 > 30)

    if n_seen_items > 1:
        longest_seen_item = x['item_id'][
            np.argmax(time_deltas_between_items)]
        shortest_seen_item = x['item_id'][
            np.argmin(time_deltas_between_items)]
    else:
        longest_seen_item = x['item_id'][0]
        shortest_seen_item = longest_seen_item

    return (
        session_length_seconds,
        avg_time_spent_on_item_seconds,
        variance_time_spent_on_item_seconds,
        longest_seen_item,
        shortest_seen_item,
        n_seen_items,
        user_went_afk
    )


# TODO: Implement the special time features
def get_special_date_features(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    input_dataframe["date_0"] = pd.to_datetime(input_dataframe['date'].str[0])
    input_dataframe["is_weekend"] = (input_dataframe["date_0"].dt.day_of_week == 5) | (input_dataframe["date_0"].dt.day_of_week == 6)
    input_dataframe["is_hot_hour"] = (datetime.time(hour=21) > input_dataframe["date_0"].dt.time) & (input_dataframe["date_0"].dt.time > datetime.time(hour=18))
    input_dataframe["is_night"] = (datetime.time(hour=23) < input_dataframe["date_0"].dt.time) | (input_dataframe["date_0"].dt.time < datetime.time(hour=5))
    input_dataframe["is_christmas_time"] = input_dataframe["date_0"].dt.month == 12
    input_dataframe["is_black_friday"] = (input_dataframe["date_0"].dt.month == 11) & (27 <= input_dataframe["date_0"].dt.day) & (input_dataframe["date_0"].dt.day <= 30)
    return input_dataframe
