from flask import Flask, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from math import ceil
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, LSTM

# from flask import Flask, request, jsonify
# import yfinance as yf
# from datetime import datetime, timedelta
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import io
# import base64
# from math import ceil
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import neighbors
# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense, LSTM

app = Flask(__name__)

def plot_to_img():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_bytes = buf.getvalue()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    buf.close()
    plt.clf()
    return encoded

def get_stock_data(ticker):
    end_date = datetime.today() - timedelta(days=0)
    start_date = end_date - timedelta(days=365 * 10)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    df1 = yf.download(ticker, start=start_date_str, end=end_date_str)
    return df1


def prepare_data(df1,ticker):
    # Plot the data
    df1['Date'] = pd.to_datetime(df1.index)
    csv_filename = f"{ticker}.csv"
    df1.to_csv(csv_filename)
    df = pd.read_csv(csv_filename)
    df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
    df.index = df['Date']

    # Drop rows where the index is NaT
    df = df.dropna(subset=['Open'])
    # Convert 'Open' column to numeric, forcing errors to NaN
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    # Drop any rows where 'Open' is NaN after conversion
    df = df.dropna(subset=['Open'])

    # Drop rows where the index is NaT
    df = df.dropna(subset=['Close'])
    # Convert 'Close' column to numeric, forcing errors to NaN
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    # Drop any rows where 'Close' is NaN after conversion
    df = df.dropna(subset=['Close'])

    # Delete the CSV file after processing
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    return df

def ensemble_lstm_fnn_prediction(df):
    # Get LSTM predictions
    lstm_rmse, lstm_img, lstm_pred = lstm_prediction(df)
    # Get FNN predictions
    fnn_rmse, fnn_img, fnn_pred = fnn_prediction(df)

    # Calculate weighted average of predictions based on inverse RMSE
    lstm_weight = 1 / lstm_rmse if lstm_rmse != 0 else 0.5
    fnn_weight = 1 / fnn_rmse if fnn_rmse != 0 else 0.5
    total_weight = lstm_weight + fnn_weight
    ensemble_pred = (lstm_pred * lstm_weight + fnn_pred * fnn_weight) / total_weight

    # Combine RMSE as weighted average
    ensemble_rmse = (lstm_rmse * lstm_weight + fnn_rmse * fnn_weight) / total_weight

    # For plotting, average the predictions from both models on the validation set
    shape = df.shape[0]
    df_new = df[['Close']]
    train = df_new[:ceil(shape * 0.75)]
    valid = df_new[ceil(shape * 0.75):]

    # Get LSTM and FNN predictions on validation set
    # Reuse code from lstm_prediction and fnn_prediction but only get predictions array
    # For simplicity, we will call the functions and extract predictions from valid['Predictions']

    # Get LSTM predictions on valid set
    _, _, _ = lstm_prediction(df)
    lstm_valid_preds = valid['Predictions'].values if 'Predictions' in valid else np.array([])

    # Get FNN predictions on valid set
    _, _, _ = fnn_prediction(df)
    fnn_valid_preds = valid['Predictions'].values if 'Predictions' in valid else np.array([])

    # If predictions are empty, fallback to ensemble_pred for plotting
    if len(lstm_valid_preds) == 0 or len(fnn_valid_preds) == 0:
        ensemble_valid_preds = np.array([ensemble_pred] * len(valid))
    else:
        ensemble_valid_preds = (lstm_valid_preds + fnn_valid_preds) / 2

    plt.figure(figsize=(10,6))
    plt.plot(train['Close'], label='Training Data')
    plt.plot(valid['Close'], label='Actual Data')
    plt.plot(valid.index, ensemble_valid_preds, label='Ensemble Predicted Data')
    plt.legend()
    plt.title('Ensemble LSTM + FNN Prediction')
    img = plot_to_img()

    return ensemble_rmse, img, ensemble_pred

@app.route('/predict')
def predict():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error': 'Please provide a stock ticker in the URL as ?ticker=XXXX'}), 400

    df = get_stock_data(ticker)
    if df.empty:
        return jsonify({'error': f'No data found for ticker {ticker}'}), 404
    try:
        df = prepare_data(df,ticker)
    except KeyError as e:
        return jsonify({'error': str(e)}), 400

    results = {}

    # Moving Average Prediction
    ma_rmse, ma_img, ma_pred = moving_avg_prediction(df)
    ma_confidence = max(0, 100 - ma_rmse * 10)
    results['Moving Average'] = {'rmse': ma_rmse, 'img': ma_img,  'price_pred': ma_pred}

    # KNN Prediction
    knn_rmse, knn_img, knn_pred = k_nearest_neighbours_predict(df)
    knn_confidence = max(0, 100 - knn_rmse * 10)
    results['K-Nearest Neighbors'] = {'rmse': knn_rmse, 'img': knn_img,  'price_pred': knn_pred}

    # LSTM Prediction
    lstm_rmse, lstm_img, lstm_pred = lstm_prediction(df)
    lstm_confidence = max(0, 100 - lstm_rmse * 10)
    results['LSTM'] = {'rmse': lstm_rmse, 'img': lstm_img,  'price_pred': lstm_pred}

    # ANN Prediction
    ann_rmse, ann_img, ann_pred = ann_prediction(df)
    ann_confidence = max(0, 100 - ann_rmse * 10)
    results['ANN'] = {'rmse': ann_rmse, 'img': ann_img, 'price_pred': ann_pred}

    # FNN Prediction
    fnn_rmse, fnn_img, fnn_pred = fnn_prediction(df)
    fnn_confidence = max(0, 100 - fnn_rmse * 10)
    results['FNN'] = {'rmse': fnn_rmse, 'img': fnn_img,  'price_pred': fnn_pred}

    # Ensemble LSTM + ANN Prediction
    ensemble_rmse, ensemble_img, ensemble_pred = ensemble_lstm_ann_prediction(df)
    ensemble_confidence = max(0, 100 - ensemble_rmse * 10)
    results['Ensemble LSTM+ANN'] = {'rmse': ensemble_rmse, 'img': ensemble_img, 'price_pred': ensemble_pred}

    # Ensemble LSTM + FNN Prediction
    ensemble_fnn_rmse, ensemble_fnn_img, ensemble_fnn_pred = ensemble_lstm_fnn_prediction(df)
    ensemble_fnn_confidence = max(0, 100 - ensemble_fnn_rmse * 10)
    results['Ensemble LSTM+FNN'] = {'rmse': ensemble_fnn_rmse, 'img': ensemble_fnn_img, 'price_pred': ensemble_fnn_pred}

    # Find best model by RMSE
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    best_name = best_model[0]
    best_rmse = best_model[1]['rmse']
    best_img = best_model[1]['img']
    best_price_pred = best_model[1]['price_pred']
    best_confidence = (100-best_rmse)

    import json
    import numpy as np

    def convert(o):
        if isinstance(o, np.float32):
            return float(o)
        raise TypeError

    current_price = None
    if not df.empty:
        current_price = float(df['Close'][-1])

    response = {
        'ticker': ticker,
        'best_model': best_name,
        'best_rmse': best_rmse,
        'prediction_plot': best_img,
        'best_price_pred': float(best_price_pred) if best_price_pred is not None else None,
        'best_confidence': float(best_confidence) if best_confidence is not None else None,
        'current_price': current_price
    }

    return jsonify(response)

def moving_avg_prediction(df):
    shape = df.shape[0]
    df_new = df[['Close']]
    train_set = df_new.iloc[:ceil(shape * 0.75)]
    valid_set = df_new.iloc[ceil(shape * 0.75):]
    preds = []
    for i in range(0, valid_set.shape[0]):
        a = train_set['Close'][len(train_set) - valid_set.shape[0] + i:].sum() + sum(preds)
        b = a / (valid_set.shape[0])
        preds.append(b)
    rms = np.sqrt(np.mean(np.power((np.array(valid_set['Close']) - preds), 2)))
    valid_set['Predictions'] = preds
    plt.figure(figsize=(10,6))
    plt.plot(train_set['Close'], label='Training Data')
    plt.plot(valid_set['Close'], label='Actual Data')
    plt.plot(valid_set['Predictions'], label='Predicted Data')
    plt.legend()
    plt.title('Moving Average Prediction')
    img = plot_to_img()
    price_pred = preds[-1] if preds else None
    return rms, img, price_pred

def k_nearest_neighbours_predict(df):
    shape = df.shape[0]
    df_new = df[['Close']]
    train_set = df_new.iloc[:ceil(shape * 0.75)]
    valid_set = df_new.iloc[ceil(shape * 0.75):]
    train = train_set.reset_index()
    valid = valid_set.reset_index()
    x_train = train['Date'].map(datetime.toordinal)
    y_train = train[['Close']]
    x_valid = valid['Date'].map(datetime.toordinal)
    y_valid = valid[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(np.array(x_train).reshape(-1, 1))
    x_valid_scaled = scaler.transform(np.array(x_valid).reshape(-1, 1))
    knn = neighbors.KNeighborsRegressor()
    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
    model = GridSearchCV(knn, params, cv=5)
    model.fit(x_train_scaled, y_train)
    preds = model.predict(x_valid_scaled)
    rms = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))
    valid_set['Predictions'] = preds
    plt.figure(figsize=(10,6))
    plt.plot(train_set['Close'], label='Training Data')
    plt.plot(valid_set['Close'], label='Actual Data')
    plt.plot(valid_set['Predictions'], label='Predicted Data')
    plt.legend()
    plt.title('K-Nearest Neighbors Prediction')
    img = plot_to_img()
    price_pred = preds[-1] if len(preds) > 0 else None
    return rms, img, price_pred

def lstm_prediction(df):
    from tensorflow.keras.callbacks import EarlyStopping
    shape = df.shape[0]
    df_new = df[['Close']]
    dataset = df_new.values
    train = df_new[:ceil(shape * 0.75)]
    valid = df_new[ceil(shape * 0.75):]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []
    for i in range(40, len(train)):
        x_train.append(scaled_data[i-40:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=5)
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])
    inputs = df_new[len(df_new) - len(valid) - 40:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(40, inputs.shape[0]):
        X_test.append(inputs[i-40:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
    valid['Predictions'] = closing_price
    plt.figure(figsize=(10,6))
    plt.plot(train['Close'], label='Training Data')
    plt.plot(valid['Close'], label='Actual Data')
    plt.plot(valid['Predictions'], label='Predicted Data')
    plt.legend()
    plt.title('LSTM Prediction')
    img = plot_to_img()
    price_pred = closing_price[-1][0] if len(closing_price) > 0 else None
    return rms, img, price_pred

def ann_prediction(df):
    from tensorflow.keras.callbacks import EarlyStopping
    shape = df.shape[0]
    df_new = df[['Close']]
    train = df_new[:ceil(shape * 0.75)]
    valid = df_new[ceil(shape * 0.75):]
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_new)
    x_train, y_train = [], []
    for i in range(40, len(train)):
        x_train.append(scaled_data[i-40:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=5)
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])
    inputs = df_new[len(df_new) - len(valid) - 40:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(40, inputs.shape[0]):
        X_test.append(inputs[i-40:i, 0])
    X_test = np.array(X_test)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    rms = np.sqrt(np.mean(np.power((valid['Close'] - predictions.flatten()), 2)))
    valid['Predictions'] = predictions
    plt.figure(figsize=(10,6))
    plt.plot(train['Close'], label='Training Data')
    plt.plot(valid['Close'], label='Actual Data')
    plt.plot(valid['Predictions'], label='Predicted Data')
    plt.legend()
    plt.title('ANN Prediction')
    img = plot_to_img()
    price_pred = predictions[-1][0] if len(predictions) > 0 else None
    return rms, img, price_pred

def fnn_prediction(df):
    from tensorflow.keras.callbacks import EarlyStopping
    shape = df.shape[0]
    df_new = df[['Close']]
    dataset = df_new.values
    train = df_new[:ceil(shape * 0.75)]
    valid = df_new[ceil(shape * 0.75):]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []
    for i in range(40, len(train)):
        x_train.append(scaled_data[i-40:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=5)
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])
    inputs = df_new[len(df_new) - len(valid) - 40:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(40, inputs.shape[0]):
        X_test.append(inputs[i-40:i, 0])
    X_test = np.array(X_test)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    rms = np.sqrt(np.mean(np.power((valid['Close'] - predictions.flatten()), 2)))
    valid['Predictions'] = predictions
    plt.figure(figsize=(10,6))
    plt.plot(train['Close'], label='Training Data')
    plt.plot(valid['Close'], label='Actual Data')
    plt.plot(valid['Predictions'], label='Predicted Data')
    plt.legend()
    plt.title('FNN Prediction')
    img = plot_to_img()
    price_pred = predictions[-1][0] if len(predictions) > 0 else None
    return rms, img, price_pred

def ensemble_lstm_ann_prediction(df):
    # Get LSTM predictions
    lstm_rmse, lstm_img, lstm_pred = lstm_prediction(df)
    # Get ANN predictions
    ann_rmse, ann_img, ann_pred = ann_prediction(df)

    # Calculate weighted average of predictions based on inverse RMSE
    lstm_weight = 1 / lstm_rmse if lstm_rmse != 0 else 0.5
    ann_weight = 1 / ann_rmse if ann_rmse != 0 else 0.5
    total_weight = lstm_weight + ann_weight
    ensemble_pred = (lstm_pred * lstm_weight + ann_pred * ann_weight) / total_weight

    # Combine RMSE as weighted average
    ensemble_rmse = (lstm_rmse * lstm_weight + ann_rmse * ann_weight) / total_weight

    # For plotting, average the predictions from both models on the validation set
    shape = df.shape[0]
    df_new = df[['Close']]
    train = df_new[:ceil(shape * 0.75)]
    valid = df_new[ceil(shape * 0.75):]

    # Get LSTM and ANN predictions on validation set
    # Reuse code from lstm_prediction and ann_prediction but only get predictions array
    # For simplicity, we will call the functions and extract predictions from valid['Predictions']

    # Get LSTM predictions on valid set
    _, _, _ = lstm_prediction(df)
    lstm_valid_preds = valid['Predictions'].values if 'Predictions' in valid else np.array([])

    # Get ANN predictions on valid set
    _, _, _ = ann_prediction(df)
    ann_valid_preds = valid['Predictions'].values if 'Predictions' in valid else np.array([])

    # If predictions are empty, fallback to ensemble_pred for plotting
    if len(lstm_valid_preds) == 0 or len(ann_valid_preds) == 0:
        ensemble_valid_preds = np.array([ensemble_pred] * len(valid))
    else:
        ensemble_valid_preds = (lstm_valid_preds + ann_valid_preds) / 2

    plt.figure(figsize=(10,6))
    plt.plot(train['Close'], label='Training Data')
    plt.plot(valid['Close'], label='Actual Data')
    plt.plot(valid.index, ensemble_valid_preds, label='Ensemble Predicted Data')
    plt.legend()
    plt.title('Ensemble LSTM + ANN Prediction')
    img = plot_to_img()

    return ensemble_rmse, img, ensemble_pred



if __name__ == '__main__':
    app.run(debug=True)
