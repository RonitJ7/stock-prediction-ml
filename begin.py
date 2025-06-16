from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pandas as pd
from LinearRegression import performLR
from getFeatures import generate_features

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
