import os

import pandas as pd

from preprocess.features import macro_features_generation

base_path = '../dataset'

original_data = os.path.join(base_path, 'original_data')
processed_data = os.path.join(base_path, 'processed_data')

train_sessions = pd.read_csv(
    os.path.join(processed_data, 'train_sessions_mapped.csv'),
    parse_dates=['date'],
    infer_datetime_format=True
)
# train_sessions = train_sessions.sort_values(by='date').groupby(['session_id']).agg(tuple).applymap(list).reset_index()
# train_sessions.sort_values(by="session_id", inplace=True)
result = macro_features_generation(train_sessions)
# result = get_special_date_features(train_sessions)
print(result)
