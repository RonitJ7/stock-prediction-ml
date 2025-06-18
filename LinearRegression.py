from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score
import pandas as pd

def performLR(X_scaled_train, X_scaled_test, Y_train, Y_test, data):
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