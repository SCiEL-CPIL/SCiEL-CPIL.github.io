import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using an orthonormal Legendre polynomial basis model.
    The model is a linear combination of 10 orthonormal 2D Legendre polynomial basis functions.
    Orthonormality improves the conditioning of the optimization problem for BFGS.
    """
    # Compute 1D orthonormal Legendre polynomials up to degree 3
    # Normalization factors: sqrt((2n+1)/2)
    
    # P0 = 1, norm factor sqrt(1/2)
    phi0_x = np.ones_like(x) * np.sqrt(0.5)
    phi1_x = x * np.sqrt(1.5)
    phi2_x = 0.5 * (3 * x**2 - 1) * np.sqrt(2.5)
    phi3_x = 0.5 * (5 * x**3 - 3 * x) * np.sqrt(3.5)
    
    phi0_y = np.ones_like(y) * np.sqrt(0.5)
    phi1_y = y * np.sqrt(1.5)
    phi2_y = 0.5 * (3 * y**2 - 1) * np.sqrt(2.5)
    phi3_y = 0.5 * (5 * y**3 - 3 * y) * np.sqrt(3.5)
    
    # Construct 2D orthonormal basis functions (products of 1D orthonormal Legendre polynomials)
    # Total 10 terms covering degrees 0 to 3
    b0 = phi0_x * phi0_y
    b1 = phi1_x * phi0_y
    b2 = phi0_x * phi1_y
    b3 = phi2_x * phi0_y
    b4 = phi0_x * phi2_y
    b5 = phi1_x * phi1_y
    b6 = phi3_x * phi0_y
    b7 = phi0_x * phi3_y
    b8 = phi2_x * phi1_y
    b9 = phi1_x * phi2_y
    
    # Stack into matrix of shape (N, 10)
    B = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Prediction is B @ params
    pred = B.dot(params)
    
    return pred
