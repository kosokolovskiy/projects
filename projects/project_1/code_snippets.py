IMPORTS = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pathlib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

import xgboost
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras import layers
'''

VISUALIZATION = '''
def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
    """
    Plots a timesteps  against values.
    Parameters
    ---------
    timesteps : array of timesteps
    values : array of values across time
    format : style of plot, default "."
    start : where to start the plot 
    end : where to end the plot 
    label : label to show on plot of values
    """
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)

def plot_loss_curves(history, metrics='mse'):
    """
    Return separate loss curves for training and validation metrics.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history[metrics]
    val_accuracy = history.history[f'val_{metrics}']

    epochs = range(len(history.history['loss']))

    fig, ax = plt.subplots(1, 2, figsize=(15, 4), dpi=200)

    ax[0].plot(epochs, loss, label='training_loss')
    ax[0].plot(epochs, val_loss, label='val_loss')
    ax[0].set_title('loss')
    ax[0].set_xlabel('epochs')
    ax[0].legend()

    ax[1].plot(epochs, accuracy, label=f'training_{metrics}')
    ax[1].plot(epochs, val_accuracy, label=f'val_{metrics}')
    ax[1].set_title(metrics)
    ax[1].set_xlabel('epochs')
    ax[1].legend()
'''

PREDICTIONS = '''
def evaluate_preds(y_true, y_pred):
    """
    Evaluates prediction accuracy by computing various error metrics between 
    the true labels and the predicted labels. The function casts inputs to float32,
    and calculates Mean Absolute Error (MAE), Mean Squared Error (MSE),
    Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
    
    Parameters:
    - y_true (Tensor): True labels. A 1D tensor of real values for the output.
    - y_pred (Tensor): Predicted labels. A 1D tensor of predicted values.
    
    Returns:
    - dict: A dictionary containing the computed metrics:
        - 'mae': Mean Absolute Error
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'mape': Mean Absolute Percentage Error
    """
    # Ensure inputs are float32
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate error metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)

    # Return metrics
    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy()}

'''

SLIDING_WINDOW = '''

# 1) Numpy Based

def get_labelled_windows(x, horizon=1):
    """
    Creates labels for windowed dataset.

    E.g. if horizon=1 (default)
    Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
    """
    return x[:, :-horizon], x[:, -horizon:]

def make_windows(x, window_size=7, horizon=1):
    """
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)

    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T 

    windowed_array = x[window_indexes]

    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels

# 2) Pandas Based

def make_windows_pandas(df, labels, features=None, window_size=7, horizon=1):
    """
    Transforms a DataFrame into a format suitable for windowed time series forecasting.
    It generates new features by shifting the specified columns backward according to 
    the window size and horizon, effectively creating lagged features. The function also 
    prepares target columns based on the specified labels.

    Parameters:
    - df (pd.DataFrame): The original DataFrame containing time series data.
    - labels (list of str): Column names in `df` that are to be used as targets.
    - features (list of str, optional): Column names to be used for generating features. 
      If None, all columns in `df` are used.
    - window_size (int, optional): The number of past observations to consider for each window.
    - horizon (int, optional): The number of steps ahead to forecast.

    Returns:
    - pd.DataFrame: A DataFrame containing the lagged features, with each column shifted 
      to create a window of observations.
    - pd.DataFrame: A DataFrame containing the target column(s) for the forecast horizon.
    """
    if features is None:
        features = []
    df_copy = df.copy()

    cols = features or df_copy.columns
    shifts = range(1, window_size + horizon)

    new_features = []
    for shift in shifts:
        for col in cols:
            df_copy[f'{col}_{shift}'] = df[col].shift(-shift)
            new_features.append(f'{col}_{shift}')

    new_features.extend(features)

    df_copy.dropna(inplace=True)

    df_copy.index = df_copy.index + pd.DateOffset(days=window_size + 2)

    target = pd.DataFrame()
    for label in labels:
        target[label] = df_copy[f'{label}_{window_size}']
        df_copy.drop(f'{label}_{window_size}', axis=1, inplace=True)

    df_copy.drop([col for col in df_copy.columns if col not in new_features], axis=1, inplace=True)

    return df_copy, target
'''


DIFFERENTS = '''
def make_train_test_splits(windows, labels, test_split=0.2):
    """
    Splits arrays or matrices of windows and labels into training and testing sets.
    
    This function is designed to split data into training and testing sets based on a specified
    proportion. It's particularly useful for time series data or when you have corresponding features
    (windows) and targets (labels) that need to be split in sync.

    Parameters:
    - windows (array-like or pd.DataFrame): The feature data to split. It should be indexed or ordered in the same way as `labels`.
    - labels (array-like or pd.DataFrame): The target data to split, corresponding to the `windows` data.
    - test_split (float, optional): The proportion of the dataset to include in the test split. Should be between 0.0 and 1.0. Defaults to 0.2.

    Returns:
    - train_windows (array-like or pd.DataFrame): The subset of `windows` used for training.
    - test_windows (array-like or pd.DataFrame): The subset of `windows` used for testing.
    - train_labels (array-like or pd.DataFrame): The subset of `labels` used for training.
    - test_labels (array-like or pd.DataFrame): The subset of `labels` used for testing.
    """
    split_size = int(len(windows) * (1-test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels

def add_features_to_df(df):
    df = df.copy()
    df['volume_shock_yes_no'] = (df['volume'].diff() / df['volume'].shift() * 100 > 10).astype(int)
    df['volume_shock_direction'] = (df['volume'].diff() / df['volume'].shift() > 0).astype(int)
    df['price_shock_yes_no'] = (df['close'].diff() / df['close'].shift() * 100 > 2).astype(int)
    df['price_shock_direction'] = (df['close'].diff() / df['close'].shift() > 0).astype(int)

    scaler = MinMaxScaler()
    df['scaled_volume'] = scaler.fit_transform(df.volume.to_numpy().reshape(-1, 1))
    df.drop('volume', axis=1, inplace=True)

    return df
'''

SCRAPPER = '''
    import requests
    from bs4 import BeautifulSoup

    def scrap_holidays(year):
        url = f'https://zerodha.com/z-connect/traders-zone/holidays/trading-holidays-{year}-nse-bse-mcx'

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        if table := soup.find('table'):
            holidays = []
            for row in table.find_all('tr')[1:]:
                columns = row.find_all('td')
                if len(columns) > 1:
                    date_with_year = f"{columns[2].get_text().strip()} 2015"
                    holiday_data = {
                        'serial_number': columns[0].get_text().strip(),
                        'holiday_name': columns[1].get_text().strip(),
                        'date': date_with_year
                    }
                    holidays.append(holiday_data)
            holidays_df = pd.DataFrame(holidays)
            holidays_df['date'] = pd.to_datetime(holidays_df['date'])
            return holidays_df
'''

AMAZON_S3 = '''
path_to_local_plots = ***
path_local_images = 'data/images/'
def upload_to_aws_png(file_name):
    upload_to_s3(f'{path_to_local_plots}{file_name}.png', f'projects/Stock_Price_analysis/plots/{file_name}.png')

def upload_plot(plot_name):
    plt.savefig(path_local_images + plot_name, dpi=300)
    upload_to_aws_png(plot_name)

path_to_local_csv = ***
path_local_csv = 'data/csv/'
def upload_to_aws_csv(file_name):
    upload_to_s3(f'{path_to_local_csv}{file_name}.csv', f'projects/Stock_Price_analysis/csv/{file_name}.csv')

def upload_csv(df, csv_name):
    df.to_csv(path_local_csv + csv_name + '.csv')
    upload_to_aws_csv(csv_name)
'''


MACHINE_LEARNING = '''
    def make_regression(x_train, y_train, x_test, y_test, model, model_name, verbose=True):

        model.fit(x_train, y_train)

        y_predict = model.predict(x_train)
        train_error = mean_squared_error(y_train, y_predict, squared=False)

        y_predict = model.predict(x_test)
        test_error = mean_squared_error(y_test, y_predict, squared=False
        )

        y_predict = model.predict(x_train)
        r2 = r2_score(y_train, y_predict)

        if verbose:
            print(f"----Model name = {model_name}-----")
            print(f"Train error = {train_error}")
            print(f"Test error = {test_error}")
            print(f"r2_score = {r2}")
            print("--------------------------------")

        trained_model = model

        return trained_model, y_predict, train_error, test_error, r2
}

'''

TRAIN_TEST_SPLIT = '''
split_size = int(0.8 * len(infy_df)) 

X_train, y_train = infy_df.index[:split_size], infy_df.close[:split_size]

X_test, y_test = infy_df.index[split_size:], infy_df.close[split_size:]

len(X_train), len(X_test), len(y_train), len(y_test)
'''

MAKE_WINDOWS_LABELS = '''
full_windows, full_labels = make_windows_pandas(infy_df, 
                                                labels=['close'], 
                                                features=['close'], 
                                                window_size=WINDOW_SIZE, 
                                                horizon=HORIZON
                                                )
len(full_windows), len(full_labels)

# Output: (3536, 3536)
'''

MODEL_1 = '''
tf.random.set_seed(42)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=10, 
                                            restore_best_weights=True
                                            )

model_1 = tf.keras.Sequential([
  layers.Dense(64, activation="relu"),
  layers.Dense(HORIZON, activation="linear")                   
], name="model_1_dense") 

model_1.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(0.0001),
                metrics=["mae"]) 

history_1 = model_1.fit(x=train_windows,
            y=train_labels, 
            epochs=200,
            verbose=1,
            batch_size=8,
            validation_data=(test_windows, test_labels), 
            callbacks=[callback],
            ) 
'''

MODEL_2 = '''
    tf.random.set_seed(42)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    model_2 = tf.keras.Sequential([
    layers.Lambda(lambda x: tf.expand_dims(x, axis=1)), 
    layers.Conv1D(filters=64, kernel_size=5, padding="causal", activation="relu"),
    layers.Dense(HORIZON)
    ], name="model_2_conv1D")

    model_2.compile(loss="mae",
                    optimizer=tf.keras.optimizers.Adam())

    model_2.fit(train_windows,
                train_labels,
                batch_size=64, 
                epochs=100,
                verbose=1,
                validation_data=(test_windows, test_labels),
                callbacks=[callback])
'''

MODEL_3 = '''
    regression_models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'XGBoost': XGBRegressor(n_estimators=1_000, max_depth=2, eta=1)}
'''

MODEL_3_TRAINING = '''
    pred_dict = {
        'model': [],
        'regression_model': [],
        'Train Error': [],
        'Test Error': [],
        'R2': []}

        for model_name in regression_models.keys():

            trained_model, y_predict, train_error, test_error, r2 = make_regression(train_windows, 
                                                                                    np.ravel(train_labels), 
                                                                                    test_windows, np.ravel(test_labels), 
                                                                                    regression_models[model_name], 
                                                                                    model_name=model_name, 
                                                                                    verbose=True)

            pred_dict['model'].append(trained_model)
            pred_dict["regression_model"].append(model_name)
            pred_dict["Train Error"].append(train_error)
            pred_dict["Test Error"].append(test_error)
            pred_dict["R2"].append(r2)
'''

FEATURE_IMPORTANCE_WITHOUT = '''
    model = pred_dict['model'][2]

    fig, ax = plt.subplots(figsize=(15, 15)) 
    xgboost.plot_importance(model, ax=ax)
    ax.set_title('Feature Importance', fontsize=14) 
    ax.set_xlabel('F score', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)  
    ax.grid('on', which='major', linestyle='-', linewidth='0.5', color='gray') 
    ax.tick_params(axis='both', which='major', labelsize=10)
'''

NEW_DF_WITH_FEATURES = '''
    infy_df_features = infy_df.copy()
    infy_df_added_features = add_features_to_df(infy_df_features)
'''

MODEL_4 = '''
    full_windows_with_features, full_labels_with_features = make_windows_pandas(infy_df_added_features, 
                                                                                labels=['close'], 
                                                                                features=list(infy_df_added_features.columns),
                                                                                window_size=WINDOW_SIZE, 
                                                                                horizon=HORIZON)
    train_windows_feat, test_windows_feat, train_labels_feat, test_labels_feat = make_train_test_splits(full_windows_with_features, 
                                                                                                        full_labels_with_features, 
                                                                                                        test_split=.2)
    tf.random.set_seed(42)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    model_4 = tf.keras.Sequential([
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(HORIZON, activation="linear")                   
    ], name="model_4_dense") 

    model_4.compile(loss="mae",
                    optimizer=tf.keras.optimizers.Adam(0.0001),
                    metrics=["mae"]) 

    history_4 = model_4.fit(x=train_windows_feat,
                y=train_labels_feat, 
                epochs=200,
                verbose=1,
                batch_size=8,
                validation_data=(test_windows_feat, test_labels_feat), 
                callbacks=[callback]) 
'''

MODEL_5 = '''
    regression_models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'XGBoost': XGBRegressor(n_estimators=1_000, max_depth=2, eta=1)}
'''


MODEL_5_TRAINING = '''
    pred_dict = {
        'model': [],
        'regression_model': [],
        'Train Error': [],
        'Test Error': [],
        'R2': []}

        for model_name in regression_models.keys():

            trained_model, y_predict, train_error, test_error, r2 = make_regression(train_windows, 
                                                                                    np.ravel(train_labels), 
                                                                                    test_windows, np.ravel(test_labels), 
                                                                                    regression_models[model_name], 
                                                                                    model_name=model_name, 
                                                                                    verbose=True)

            pred_dict['model'].append(trained_model)
            pred_dict["regression_model"].append(model_name)
            pred_dict["Train Error"].append(train_error)
            pred_dict["Test Error"].append(test_error)
            pred_dict["R2"].append(r2)
'''

FEATURE_IMPORTANCE_WITH = '''
    model = pred_dict['model'][2]

    fig, ax = plt.subplots(figsize=(15, 15)) 
    xgboost.plot_importance(model, ax=ax)
    ax.set_title('Feature Importance', fontsize=14) 
    ax.set_xlabel('F score', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)  
    ax.grid('on', which='major', linestyle='-', linewidth='0.5', color='gray') 
    ax.tick_params(axis='both', which='major', labelsize=10)
'''
