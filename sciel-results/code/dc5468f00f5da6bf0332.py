import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params[0:5] = RBF weights at 5 centers
    # params[5:10] = polynomial coefficients: 1, x, y, x^2, y^2
    
    # RBF centers: origin and 4 diagonal points
    centers = np.array([
        [ 0.0,  0.0],
        [-0.7, -0.7],
        [ 0.7,  0.7],
        [-0.7,  0.7],
        [ 0.7, -0.7]
    ])
    
    sigma = 0.65
    
    # Compute RBF contributions
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]  # n x 5
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]  # n x 5
    dist_sq = dx**2 + dy**2  # n x 5
    
    rbf_values = np.exp(-dist_sq / (2.0 * sigma**2))  # n x 5
    
    # Polynomial basis
    poly = np.column_stack([
        np.ones_like(x),
        x,
        y,
        x**2,
        y**2
    ])  # n x 5
    
    # Combine: RBF part + polynomial part
    rbf_part = rbf_values @ params[0:5]       # n
    poly_part = poly @ params[5:10]            # n
    
    return rbf_part + poly_part
