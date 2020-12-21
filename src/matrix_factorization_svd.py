from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np


# load the bostom house data
data = load_boston(return_X_y=False)
print(data)

# extract the input data matrix and the targets
A = data.data
b = data.target

# append a columns of 1s (these are the biases)
A = np.column_stack([np.ones(A.shape[0]), A])

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(A, b, test_size=0.50, random_state=42)

# calculate the economy SVD for the data matrix A
U,S,Vt = np.linalg.svd(X_train, full_matrices=False)

# solve Ax = b for the best possible approximate solution in terms of least squares
x_hat = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y_train

# perform train and test inference
train_predictions = X_train @ x_hat
test_predictions = X_test @ x_hat

# compute train and test MSE
train_mse = np.mean((train_predictions - y_train)**2)
test_mse = np.mean((test_predictions - y_test)**2)

print("Train Mean Squared Error:", train_mse)
print("Test Mean Squared Error:", test_mse)