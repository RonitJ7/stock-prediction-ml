from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pandas as pd

def add_original_features(df,df_new):
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)


def add_avg_price(df, df_new):
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)

    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']


def add_avg_volume(df, df_new):
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)

    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']


def add_std_price(df, df_new):
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)
    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)

    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']

def add_std_volume(df, df_new):
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)

    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']

def add_return_feature(df,df_new):
    df_new['return_1'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df_new['return_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
    df_new['return_30'] = (df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)
    df_new['return_365'] = (df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)
    df_new['moving_avg_5'] = df_new['return_5'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_30'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_365'].rolling(252).mean().shift(1)

def generate_features(df):
    df_new = pd.DataFrame()
    add_original_features(df, df_new)
    add_avg_price(df, df_new)
    add_avg_volume(df, df_new)
    add_std_price(df, df_new)
    add_std_volume(df, df_new)
    add_return_feature(df, df_new)
    
    df_new['close'] = df['Close']
    # Drop rows with NaN values that were created by shifting and rolling operations
    df_new.dropna(inplace=True,axis = 0)
    
    return df_new

def performLR(data):
    start_train = '1990-01-01'
    end_train = '2023-12-31'
    start_test = '2024-01-01'
    end_test = '2024-12-31'
    data_train = data.loc[start_train:end_train]
    data_test = data.loc[start_test:end_test]
    X_train = data_train.drop('close', axis=1).values
    Y_train = data_train['close'].values
    X_test = data_test.drop('close', axis=1).values
    Y_test = data_test['close'].values
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    lr = SGDRegressor(max_iter = 5000,penalty = None, learning_rate = 'invscaling', eta0 = 0.01, random_state = 42)
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(lr,param_grid = {"alpha": [1e-4,3e-3,1e-3,3e-2]},cv = tscv,scoring = 'r2')
    grid_search.fit(X_scaled_train, Y_train)
    print("Best parameters found: ", grid_search.best_params_)
    lr_search = grid_search.best_estimator_
    predictions = lr_search.predict(X_scaled_test)
    print("R2 score on test set: ", r2_score(Y_test,predictions))
    pred_csv = pd.DataFrame({"date": data.loc['2024-01-01': '2024-12-31'].index, "predicted_close": predictions, "actual_close": Y_test})
    pred_csv.to_csv('predictions.csv', index=False)

def main():
    #first let's obtain the data and add the features
    data_raw = pd.read_csv('nasdaq_1990_2024.csv', index_col='Price', parse_dates=True)
    # features to consider: 
    # avg close price over the past 3 timeframes, and ratios of above. 
    # also use volume traded: avg volume over timeframes and ratios of such again. 
    # volatility of stock: Std Dev of above values
    data = generate_features(data_raw)
    #Now we will perform linear regression on the data
    performLR(data)

if __name__ == "__main__":
    main()
