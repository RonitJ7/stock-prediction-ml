from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV , TimeSeriesSplit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def decisionTree(X_train,X_test,Y_train,Y_test):
    param_grid = {
        'max_depth' : [20,30,50] ,
        'min_samples_split' : [2,5,10] , 
        'min_samples_leaf' : [1,3,5]
    }
    dt = DecisionTreeRegressor(random_state = 42)
    tscv = TimeSeriesSplit(n_splits = 3)
    grid_search = GridSearchCV(dt,param_grid,cv = tscv, scoring = "r2" , n_jobs = -1)
    grid_search.fit(X_train,Y_train)
    print(grid_search.best_params_)
    dt_best = grid_search.best_estimator_
    predictions_dt = dt_best.predict(X_test)
    print(f"R^2: {r2_score(Y_test,predictions_dt) : .3f}")
    print(X_train.shape, X_test.shape)
    print(Y_train.shape, Y_test.shape)
    print("Train R²:", dt_best.score(X_train, Y_train))
    print("Test R²:", dt_best.score(X_test, Y_test))
    plt.plot(Y_test, label="Actual")
    plt.plot(predictions_dt, label="Predicted")
    plt.legend()
    plt.title("Decision Tree Predictions vs Actual")
    plt.show()

