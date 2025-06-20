import torch
import numpy as np
import sklearn.metrics as sm
import pandas as pd

def NNpredict(X_train ,y_train ,  X_scaled_train , X_scaled_test , y_test,data):
    X_train_torch = torch.from_numpy(X_scaled_train.astype(np.float32))
    X_test_torch = torch.from_numpy(X_scaled_test.astype(np.float32))
    y_train = y_train.reshape(y_train.shape[0],1)
    y_train_torch = torch.from_numpy(y_train.astype(np.float32))
    torch.manual_seed(42)
    #just use a simple neural network which has one hidden layer with 32 nodes, followed by a ReLU function. 
    model = torch.nn.Sequential(torch.nn.Linear(X_train.shape[1],32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.3)
    for epoch in range(1000):
        model.train()  # set model to training mode
        predictions = model(X_train_torch)
        loss = loss_function(predictions, y_train_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Loss in epoch {epoch} is {loss.item():.4f}")
    predictions = model(X_test_torch).detach().numpy()[:,0]
    print(f"MSE(NN): {sm.mean_squared_error(y_test,predictions)}")
    print(f"MAE(NN): {sm.mean_absolute_error(y_test,predictions)}")
    print(f"R^2 score (NN): {sm.r2_score(y_test,predictions)}")
    pred_csv = pd.DataFrame({"date": data.loc['2024-01-01': '2024-12-31'].index, "predicted_close": predictions, "actual_close": y_test})
    pred_csv.to_csv('NNpredictions.csv', index=False)
