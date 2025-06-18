import pandas as pd
from LinearRegression import performLR
from getFeatures import generate_features
from preprocessData import preprocess
from DecisionTree import decisionTree
def main():
    #first let's obtain the data and add the features
    data_raw = pd.read_csv('nasdaq_1990_2024.csv', index_col='Price', parse_dates=True)
    # features to consider: 
    # avg close price over the past 3 timeframes, and ratios of above. 
    # also use volume traded: avg volume over timeframes and ratios of such again. 
    # volatility of stock: Std Dev of above values
    data = generate_features(data_raw)
    x_train,x_test,x_scaled_train , x_scaled_test , y_train , y_test = preprocess(data)
    #Now we will perform linear regression on the data
    # performLR(x_scaled_train,x_scaled_test,y_train,y_test,data)
    decisionTree(x_train,x_test, y_train,y_test)



if __name__ == "__main__":
    main()
