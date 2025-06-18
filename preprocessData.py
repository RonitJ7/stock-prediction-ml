from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler

def preprocess(data):    
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
    # scaler = MinMaxScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    return X_train, X_test , X_scaled_train,X_scaled_test,Y_train,Y_test
