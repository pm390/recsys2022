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

def macro_features_generation(input_dataframe: pd.DataFrame, embeddings) -> pd.DataFrame:
    # input_dataframe = remove_reloaded_items(input_dataframe)

    input_dataframe = input_dataframe.sort_values(by='date').groupby(['session_id']).agg(list).reset_index()
    input_dataframe.sort_values(by="session_id", inplace=True)

    print('completed removal')
    # input_dataframe = get_date_features(input_dataframe)
    print('completed date features')
    # input_dataframe = get_session_length_features(input_dataframe)
    print('completed session length')
    # input_dataframe = get_special_date_features(input_dataframe)
    print('completed special features')

    input_dataframe = get_session_similarity(input_dataframe, embeddings)
    return input_dataframe


def get_session_similarity(input_dataframe: pd.DataFrame, embeddings) -> pd.DataFrame:
    session_similarity = ['session_similarity']

    # compute length of sessions in seconds
    input_dataframe['session_similarity'] = input_dataframe[['date', 'item_id']].apply(
        compute_similarity,
        embeddings=embeddings,
        axis=1,
        result_type="expand"
    )
    return input_dataframe


def compute_similarity(x, embeddings):
    normalized_embedding_matrix = embeddings[x['item_id']] / np.expand_dims(np.linalg.norm(embeddings[x['item_id']], axis=1), axis=1)
    s = np.linalg.svd(normalized_embedding_matrix, compute_uv=False)[0]
    return s




# Remove sequent items if the same item has a delta t of 30 seconds
def remove_reloaded_items(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    shifted_item_ids_per_group = input_dataframe.sort_values(['session_id', 'date']).groupby(['session_id'])[
        'item_id'].shift(-1).fillna(0).astype(int)
    shifted_datetime_per_group = input_dataframe.sort_values(['session_id', 'date']).groupby(['session_id'])[
        'date'].shift(-1)  # .fillna(0)
    consecutive_item_filter = (
                input_dataframe.sort_values(['session_id', 'date'])['item_id'] - shifted_item_ids_per_group).eq(0)
    time_delta_filter = (shifted_datetime_per_group - input_dataframe.sort_values(['session_id', 'date'])[
        'date']).dt.total_seconds() < 30
    duplication_filter = ~(consecutive_item_filter & time_delta_filter)

    filtered_dataframe = input_dataframe[duplication_filter]
    return filtered_dataframe
    # input_dataframe[['date', 'item_id']] = input_dataframe[['date', 'item_id']].apply(
    #     remove_items,
    #     axis=1,
    #     result_type="expand"
    # )
    # return input_dataframe


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

    input_dataframe["date_session_starting"] = pd.to_datetime(input_dataframe['date'].str[0])
    input_dataframe["date_session_ending"] = pd.to_datetime(input_dataframe['date'].str[-1])

    input_dataframe["date_hour_sin"] = np.sin(input_dataframe["date_session_starting"].dt.hour * np.pi / 30)
    input_dataframe["date_hour_cos"] = np.cos(input_dataframe["date_session_starting"].dt.hour * np.pi / 30)
    input_dataframe["date_day_sin"] = np.sin(input_dataframe["date_session_starting"].dt.hour * np.pi / 15)
    input_dataframe["date_day_cos"] = np.cos(input_dataframe["date_session_starting"].dt.hour * np.pi / 15)
    input_dataframe["date_month_sin"] = np.sin(input_dataframe["date_session_starting"].dt.hour * np.pi / 6)
    input_dataframe["date_month_cos"] = np.cos(input_dataframe["date_session_starting"].dt.hour * np.pi / 6)

    input_dataframe["date_hour_sin_ending"] = np.sin(input_dataframe["date_session_ending"].dt.hour * np.pi / 30)
    input_dataframe["date_hour_cos_ending"] = np.cos(input_dataframe["date_session_ending"].dt.hour * np.pi / 30)
    input_dataframe["date_day_sin_ending"] = np.sin(input_dataframe["date_session_ending"].dt.hour * np.pi / 15)
    input_dataframe["date_day_cos_ending"] = np.cos(input_dataframe["date_session_ending"].dt.hour * np.pi / 15)
    input_dataframe["date_month_sin_ending"] = np.sin(input_dataframe["date_session_ending"].dt.hour * np.pi / 6)
    input_dataframe["date_month_cos_ending"] = np.cos(input_dataframe["date_session_ending"].dt.hour * np.pi / 6)

    input_dataframe["date_year_2020"] = (input_dataframe["date_session_starting"].dt.year == 2020).astype(int)

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
                                    'n_unique',
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
    n_unique = len(set(x["item_id"]))

    time_deltas_between_items = np.array([(x['date'][i + 1] - x['date'][i]).total_seconds() for i in
                                          range(len(x['date']) - 1)]) if n_seen_items > 1 else np.array([0])

    avg_time_spent_on_item_seconds = session_length_seconds / (len(x['date']) - 1) if n_seen_items > 1 else 0

    variance_time_spent_on_item_seconds = np.var(time_deltas_between_items)

    user_went_afk = int(any(time_deltas_between_items / 60 > 30))

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
        n_unique,
        n_seen_items,
        user_went_afk
    )


# TODO: Implement the special time features
def get_special_date_features(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    input_dataframe["date_0"] = pd.to_datetime(input_dataframe['date'].str[0])

    input_dataframe["is_weekend"] = ((input_dataframe["date_0"].dt.day_of_week == 5) | (
            input_dataframe["date_0"].dt.day_of_week == 6)).astype(int)

    input_dataframe["is_hot_hour"] = ((datetime.time(hour=21) > input_dataframe["date_0"].dt.time) & (
            input_dataframe["date_0"].dt.time > datetime.time(hour=18))).astype(int)

    input_dataframe["is_night"] = ((datetime.time(hour=23) < input_dataframe["date_0"].dt.time) | (
            input_dataframe["date_0"].dt.time < datetime.time(hour=5))).astype(int)

    input_dataframe["is_christmas_time"] = (input_dataframe["date_0"].dt.month == 12).astype(int)

    input_dataframe["is_black_friday"] = ((input_dataframe["date_0"].dt.month == 11) & (
            27 <= input_dataframe["date_0"].dt.day) & (input_dataframe["date_0"].dt.day <= 30)).astype(int)

    return input_dataframe
