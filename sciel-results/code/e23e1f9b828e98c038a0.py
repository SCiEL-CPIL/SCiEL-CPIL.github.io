"""
Chebyshev polynomial basis with 10 parameters.
This maps the input domain [0.1, 6] to [-1, 1] and uses the first 10
Chebyshev polynomials T_0(t) to T_9(t) as a basis.
This is a structural change from Fourier series, providing a polynomial
approximation that is optimal in the minimax sense for smooth functions
on a finite interval, potentially capturing non-periodic trends better
than fixed-frequency sinusoids.
"""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial model with 10 parameters.
    params[0..9]: coefficients for T_0(t) to T_9(t)
    
    1. Map x from [0.1, 6] to t in [-1, 1].
       t = (2 * (x - x_min) - (x_max - x_min)) / (x_max - x_min)
       t = (2x - 1.2 - 5.9) / 5.9 = (2x - 7.1) / 5.9
       
       Wait, let's re-calculate carefully.
       Standard mapping [a, b] to [-1, 1]:
       t = (2 * (x - a) - (b - a)) / (b - a)
       a = 0.1, b = 6.
       b - a = 5.9
       t = (2 * (x - 0.1) - 5.9) / 5.9
       t = (2x - 0.2 - 5.9) / 5.9
       t = (2x - 6.1) / 5.9
       
       Let's check endpoints:
       x = 0.1 -> t = (0.2 - 6.1) / 5.9 = -5.9 / 5.9 = -1. Correct.
       x = 6.0 -> t = (12.0 - 6.1) / 5.9 = 5.9 / 5.9 = 1. Correct.
       
    2. Compute Chebyshev polynomials T_k(t) for k=0..9.
       T_0(t) = 1
       T_1(t) = t
       T_k(t) = 2t * T_{k-1}(t) - T_{k-2}(t)
       
    3. Prediction is the dot product of params and the basis matrix.
    """
    # Map x to t in [-1, 1]
    x_min = 0.1
    x_max = 6.0
    range_x = x_max - x_min
    
    t = (2.0 * (x - x_min) - range_x) / range_x
    
    # Number of parameters / basis functions
    n_params = 10
    
    # Initialize basis matrix
    # basis[i, k] is T_k(t_i)
    basis = np.empty((x.shape[0], n_params), dtype=np.float64)
    
    # T_0(t) = 1
    basis[:, 0] = 1.0
    
    # T_1(t) = t
    if n_params > 1:
        basis[:, 1] = t
    
    # Recurrence relation for k >= 2
    for k in range(2, n_params):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    # Prediction
    prediction = basis @ params
    
    return prediction
