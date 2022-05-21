import pandas as pd
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess.features import macro_features_generation

base_path = '../dataset'

original_data = os.path.join(base_path, 'original_data')
processed_data = os.path.join(base_path, 'processed_data')
train_session_mapped = os.path.join(processed_data, 'train_sessions_mapped.csv')
train_purchases_mapped = os.path.join(processed_data, 'train_purchases_mapped.csv')
embeddings = os.path.join(processed_data, 'compressed_features.npy')

static_features_list = (
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
)

item_related_features = [
    'timedelta'
]


class Preprocessor:
    def __init__(self, original_data_path=original_data, processed_data_path=processed_data,
                 train_session_mapped_path=train_session_mapped, train_purchases_mapped_path=train_purchases_mapped,
                 embeddings_path=embeddings, static_features=static_features_list):
        self.original_data_path = original_data_path
        self.processed_data_path = processed_data_path
        self.train_session_mapped_path = train_session_mapped_path
        self.train_purchases_mapped_path = train_purchases_mapped_path
        self.embeddings_path = embeddings_path
        self.static_features = static_features

    def pre_process_dataset(self):
        train_session = self._get_train_session()
        # train_session = self._get_extra_features(train_session)  # dataframe di train completo con tutte le colonne
        y_item_id = self._get_target_data()  # returna gli item_id target delle singole sessioni
        y_features = self._get_features_target(y_item_id)

        # x_ids = self._pad_sequences(train_session)
        # x_item_related = self._pad_item_features(train_session)

        x_train, x_test, y_train, y_test, y_features_train, y_features_test = self._train_test_split(train_session,
                                                                                                     y_item_id,
                                                                                                     y_features)
        x_train['item_id'] = self._pad_sequences(x_train['item_id']).tolist()
        x_train[item_related_features[0]] = self._pad_sequences(x_train[item_related_features[0]]).tolist()


        # TODO: fallo anche per gli altri cazzo di padding
        # TODO:
        print('bllallbllabl')

    def _get_train_session(self):
        train_sessions = pd.read_csv(self.train_session_mapped_path,
                                     parse_dates=['date'],
                                     infer_datetime_format=True
                                     )
        train_sessions = train_sessions.sort_values(by='date').groupby(['session_id']).agg(list).reset_index()

        train_sessions.sort_values(by="session_id", inplace=True)
        return train_sessions

    @staticmethod
    def _get_extra_features(train_session):
        return macro_features_generation(train_session)

    def _get_target_data(self):
        train_purchases = pd.read_csv(self.train_purchases_mapped_path, usecols=['session_id', 'item_id'])
        train_purchases.sort_values(by="session_id", inplace=True)
        y = train_purchases['item_id'].to_numpy()
        return y

    # Returna un numpy array questa roba, quindi solo il numpy array associato al padding delle sequenze item_id
    @staticmethod
    def _pad_sequences(sequences):
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            padding='post'
        )
        return padded_sequences

    @staticmethod
    def _pad_item_features(sequences):
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            dtype='float16',
            maxlen=100,
            padding='post'
        )

        if len(item_related_features) == 1:
            padded_sequences = np.expand_dims(padded_sequences, axis=-1)

        return padded_sequences

    def _get_features_target(self, train_purchases):
        embedding_weights = np.load(self.embeddings_path)
        y_features = embedding_weights[train_purchases]
        return y_features

    @staticmethod
    def _train_test_split(train_sessions, y, y_features):
        return train_test_split(train_sessions, y, y_features, test_size=0.2, random_state=1234)

    def _get_dataset_type(self, x_ids_train, x_ids_test, y_train, y_test, y_features_train, y_features_test):
        raise NotImplementedError

#
# class PreprocessorSimple(Preprocessor):
#     def _get_dataset_type(self, x_ids_train, x_ids_test, y_train, y_test, y_features_train, y_features_test):
#         train_set = tf.data.Dataset.from_tensor_slices(
#             (x_ids_train, (y_train, y_features_train))
#         ).batch(512, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).shuffle(1563,
#                                                                                              reshuffle_each_iteration=True)
#         test_set = tf.data.Dataset.from_tensor_slices(
#             (x_ids_test, (y_test, y_features_test))
#         ).batch(512)
#
#         return train_set, test_set
#
#
# class PreprocessorDouble(Preprocessor):
#     def _get_dataset_type(self, x_ids_train, x_ids_test, y_train, y_test, y_features_train, y_features_test):
#         double_output_ds = tf.data.Dataset.from_tensor_slices((y_train, y_train))
#         input_ds = tf.data.Dataset.from_tensor_slices(x_ids_train)
#
#         double_train_set = tf.data.Dataset.zip((input_ds, double_output_ds)).batch(512,
#                                                                                    num_parallel_calls=tf.data.AUTOTUNE).prefetch(
#             tf.data.AUTOTUNE).shuffle(1563, reshuffle_each_iteration=True)
#
#         test_set = tf.data.Dataset.from_tensor_slices(
#             (x_ids_test, (y_test, y_features_test))
#         ).batch(512)
#
#         return double_train_set, test_set
#
#
# class PreprocessorTime(Preprocessor):
#     def _get_dataset_type(self, x_ids_train, x_ids_test, y_train, y_test, y_features_train, y_features_test,
#                           x_static_train, x_static_test):
#         train_set_time = tf.data.Dataset.from_tensor_slices(
#             ((x_ids_train, x_static_train), (y_train, y_features_train))
#         ).batch(512, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).shuffle(1563,
#                                                                                              reshuffle_each_iteration=True)
#         test_set_time = tf.data.Dataset.from_tensor_slices(
#             ((x_ids_test, x_static_test), (y_test, y_features_test))
#         ).batch(512)
#
#         return train_set_time, test_set_time
#
# class PreprocessorComplete(Preprocessor):
#     def _get_dataset_type(self, x_ids_train, x_ids_test, y_train, y_test, y_features_train, y_features_test):
#         train_set_complete = tf.data.Dataset.from_tensor_slices(
#             ((x_ids_train, x_item_related_train, x_static_train), (y_train, y_features_train))
#         ).batch(512, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).shuffle(1563,
#                                                                                              reshuffle_each_iteration=True)
#         test_set_complete = tf.data.Dataset.from_tensor_slices(
#             ((x_ids_test, x_item_related_test, x_static_test), (y_test, y_features_test))
#         ).batch(512)
