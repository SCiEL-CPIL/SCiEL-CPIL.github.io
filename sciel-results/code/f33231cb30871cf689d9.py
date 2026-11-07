import numpy as np

# Precompute centers and width
centers = np.array([
    [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
    [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
    [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]
])
sigma = 0.6

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x, y are arrays of length N
    # params is array of length 10
    
    # Compute squared distances to each center
    # X: (N, 1), Y: (N, 1)
    # C: (9, 2)
    
    # Efficient computation
    # dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    
    x_col = x[:, np.newaxis]
    y_col = y[:, np.newaxis]
    cx = centers[:, 0][np.newaxis, :]
    cy = centers[:, 1][np.newaxis, :]
    
    dist_sq = (x_col - cx)**2 + (y_col - cy)**2
    
    # Gaussian basis values
    # shape (N, 9)
    basis = np.exp(-dist_sq / (2 * sigma**2))
    
    # Add constant term as the 10th basis function
    # We can append a column of 1s
    # Or just add params[9] * 1.0
    
    # Prediction = basis @ params[:9] + params[9]
    pred = basis @ params[:9] + params[9]
    
    return pred
