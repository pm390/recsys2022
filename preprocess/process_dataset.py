import pandas as pd
import os
import tensorflow as tf
import numpy as np
from preprocess.features import macro_features_generation

base_path = '../dataset'

original_data_path = os.path.join(base_path, 'original_data')
processed_data_path = os.path.join(base_path, 'processed_data')
train_session_mapped_path = os.path.join(base_path, 'train_sessions_mapped.csv')
train_purchases_mapped_path = os.path.join(base_path, 'train_purchases_mapped.csv')
embeddings_path = os.path.join(base_path, 'compressed_features.npy')

static_features = [
    'date_hour_sin',
    'date_hour_cos',
    'date_day_sin',
    'date_day_cos',
    'date_month_sin',
    'date_month_cos',
    'date_year_2020',
    # 'length_of_session_seconds',
    # 'avg_time_spent_per_item_seconds',
    # 'variance_time_spent_per_item_seconds',
    'n_seen_items',
    'user_went_afk',
    'is_weekend',
    'is_hot_hour',
    'is_night',
    'is_christmas_time',
    'is_black_friday',
]

item_related_features = [
    'timedelta'
]


def pre_process_dataset():

    #######
    # Create training session data
    #######

    train_sessions = pd.read_csv(train_session_mapped_path,
                                 parse_dates=['date'],
                                 infer_datetime_format=True
                                 )

    train_sessions = train_sessions.sort_values(by='date').groupby(['session_id']).agg(tuple).applymap(list).reset_index()

    train_sessions.sort_values(by="session_id", inplace=True)
    train_sessions = macro_features_generation(train_sessions)

    ########

    ########
    # Create training session target
    ########

    train_purchases = pd.read_csv(train_purchases_mapped_path, usecols=['session_id', 'item_id'])
    train_purchases.sort_values(by="session_id", inplace=True)
    y = train_purchases['item_id'].to_numpy()
    ########

    ########
    # padding sequences
    #######
    x_ids = tf.keras.preprocessing.sequence.pad_sequences(
        train_sessions['item_id'],
        padding='post'
    )
    #######

    #######
    # padding item related features
    #######

    # for item_related_feature in item_related_features:
    print('Padding {}'.format(item_related_features[0]))
    x_item_related = tf.keras.preprocessing.sequence.pad_sequences(
        train_sessions[item_related_features[0]],
        dtype='float16',
        maxlen=100,
        padding='post'
    )

    if len(item_related_features) == 1:
        x_item_related = np.expand_dims(x_item_related, axis=-1)

    ########
    embedding_weights = np.load(embeddings_path)

    y_features = embedding_weights[y]


    # returno questo e poi quelli sotto sono train set diversi in base alla rete usata
    x_ids_train, x_ids_test, x_static_train, x_static_test, x_item_related_train, x_item_related_test, y_train, y_test, y_features_train, y_features_test = train_test_split(
        x_ids,
        train_sessions[static_features].to_numpy(),
        x_item_related,
        y,
        y_features,
        test_size=0.2,
        random_state=1234
    )



    train_set = tf.data.Dataset.from_tensor_slices(
        (x_ids_train, (y_train, y_features_train))
    ).batch(512, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).shuffle(1563,
                                                                                         reshuffle_each_iteration=True)
    test_set = tf.data.Dataset.from_tensor_slices(
        (x_ids_test, (y_test, y_features_test))
    ).batch(512)

    double_output_ds = tf.data.Dataset.from_tensor_slices((y_train, y_train))
    input_ds = tf.data.Dataset.from_tensor_slices(x_ids_train)

    double_train_set = tf.data.Dataset.zip((input_ds, double_output_ds)).batch(512,
                                                                               num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE).shuffle(1563, reshuffle_each_iteration=True)

    train_set_featureless = tf.data.Dataset.from_tensor_slices((x_ids_train, y_train)).batch(512,
                                                                                             num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE).shuffle(1563, reshuffle_each_iteration=True)
    test_set_featureless = tf.data.Dataset.from_tensor_slices((x_ids_test, y_test)).batch(512)

    train_set_time = tf.data.Dataset.from_tensor_slices(
        ((x_ids_train, x_static_train), (y_train, y_features_train))
    ).batch(512, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).shuffle(1563,
                                                                                         reshuffle_each_iteration=True)
    test_set_time = tf.data.Dataset.from_tensor_slices(
        ((x_ids_test, x_static_test), (y_test, y_features_test))
    ).batch(512)

    train_set_complete = tf.data.Dataset.from_tensor_slices(
        ((x_ids_train, x_item_related_train, x_static_train), (y_train, y_features_train))
    ).batch(512, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).shuffle(1563,
                                                                                         reshuffle_each_iteration=True)
    test_set_complete = tf.data.Dataset.from_tensor_slices(
        ((x_ids_test, x_item_related_test, x_static_test), (y_test, y_features_test))
    ).batch(512)




