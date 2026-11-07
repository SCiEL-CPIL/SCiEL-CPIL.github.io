import numpy as np
import time

_START_TIME = time.time()

def equation(x, y, params):
    # Deadline guard: safety margin for the 150s limit
    if time.time() - _START_TIME > 145:
        return np.zeros_like(x)

    # Structural Change: Radial Basis Functions (RBFs)
    # This allows the model to capture local features and complex non-linearities
    # better than a global polynomial/trig basis, provided the centers are well-placed.
    
    # 10 Centers distributed in [-1, 1] x [-1, 1]
    # Using a grid plus some diagonal points for better coverage
    centers = np.array([
        [-0.7, -0.7],
        [ 0.0, -0.7],
        [ 0.7, -0.7],
        [-0.7,  0.0],
        [ 0.0,  0.0],
        [ 0.7,  0.0],
        [-0.7,  0.7],
        [ 0.0,  0.7],
        [ 0.7,  0.7],
        [ 0.35, 0.35]
    ])
    
    # Width (sigma) controls the spread of each RBF
    # A smaller sigma makes the model more local/flexible
    sigma = 0.5
    
    # Calculate squared distances from each point (x, y) to each center
    dx = x[:, np.newaxis] - centers[:, 0]
    dy = y[:, np.newaxis] - centers[:, 1]
    dist_sq = dx**2 + dy**2
    
    # Evaluate RBFs: exp(-dist_sq / (2 * sigma^2))
    rbf = np.exp(-dist_sq / (2 * sigma**2))
    
    # Linear combination of RBFs with learned parameters
    result = rbf @ params
    
    return result
