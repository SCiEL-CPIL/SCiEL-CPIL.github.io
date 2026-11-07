"""
Strategy: Adaptive Hybrid Basis via Internal Grid Search.
Instead of relying on a fixed, potentially redundant basis or a complex evolutionary search
that might be slow or unstable, we use the time budget to perform a robust grid search over
a small set of distinct, well-conditioned basis families. We evaluate each candidate basis
using a fast linear least-squares fit (since all bases are linear in parameters) on the
training data, select the one with the lowest training error, and then use that basis for
prediction. This allows the model to adapt its functional form (polynomial, trigonometric,
RBF) to the specific structure of the hidden target function.

This approach directly addresses the open question of whether structural diversity (the 'diff'
operator in the evolutionary context) improves performance. By explicitly testing diverse
basis families, we isolate the effect of basis choice on generalization.
"""
import numpy as np
import time

def _basis_polynomial(x, y):
    """
    Polynomial basis up to degree 3.
    1, x, y, x^2, y^2, xy, x^3, y^3, x^2y, xy^2
    """
    return np.column_stack([
        np.ones_like(x),
        x,
        y,
        x**2,
        y**2,
        x * y,
        x**3,
        y**3,
        x**2 * y,
        x * y**2
    ])

def _basis_trigonometric(x, y):
    """
    Trigonometric basis.
    1, cos(x), sin(x), cos(y), sin(y), cos(x)sin(y), sin(x)cos(y), cos(2x), sin(2x), cos(2y)
    """
    return np.column_stack([
        np.ones_like(x),
        np.cos(x),
        np.sin(x),
        np.cos(y),
        np.sin(y),
        np.cos(x) * np.sin(y),
        np.sin(x) * np.cos(y),
        np.cos(2 * x),
        np.sin(2 * x),
        np.cos(2 * y)
    ])

def _basis_rbf(x, y):
    """
    Radial Basis Function basis (Gaussian kernels) on a 3x3 grid + bias.
    Centers: (-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)
    Gamma = 1.0
    """
    centers = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],  [0, 0],  [0, 1],
        [1, -1],  [1, 0],  [1, 1]
    ])
    
    dx = x[:, None] - centers[None, :, 0]
    dy = y[:, None] - centers[None, :, 1]
    dist_sq = dx**2 + dy**2
    
    kernels = np.exp(-1.0 * dist_sq)
    
    # Add a bias term as the 10th column
    bias = np.ones((len(x), 1))
    return np.hstack([kernels, bias])

def _basis_hybrid(x, y):
    """
    Hybrid basis: mix of low-order polynomial and trigonometric.
    1, x, y, x^2, y^2, xy, cos(x), sin(x), cos(y), sin(y)
    """
    return np.column_stack([
        np.ones_like(x),
        x,
        y,
        x**2,
        y**2,
        x * y,
        np.cos(x),
        np.sin(x),
        np.cos(y),
        np.sin(y)
    ])

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    x_flat = np.ravel(x)
    y_flat = np.ravel(y)
    
    # List of candidate basis functions
    # Each function takes x, y and returns a matrix of shape (N, 10)
    bases = [
        _basis_polynomial,
        _basis_trigonometric,
        _basis_rbf,
        _basis_hybrid
    ]
    
    best_basis_func = _basis_polynomial
    best_error = np.inf
    
    # Evaluate each basis using fast linear least squares (LSTSQ)
    # This is valid because all bases are linear in params.
    # BFGS will converge to the same local minimum (global for linear LSQ)
    # if the problem is well-posed.
    
    for basis_func in bases:
        if time.time() > deadline:
            break
            
        try:
            A = basis_func(x_flat, y_flat)
            # Solve A @ p = y_flat for p
            # Use lstsq for robustness
            p_hat, residuals, rank, s = np.linalg.lstsq(A, y_flat, rcond=None)
            error = np.mean((A @ p_hat - y_flat)**2)
            
            if error < best_error:
                best_error = error
                best_basis_func = basis_func
        except Exception:
            continue
    
    # If no basis was evaluated successfully (should not happen), fallback to polynomial
    # But we must return a prediction using the 'params' provided by the evaluator.
    # The evaluator will fit 'params' using BFGS on the training data with the chosen structure.
    # So we just need to return the prediction for the given 'params' using the best structure.
    
    A_test = best_basis_func(x_flat, y_flat)
    pred = A_test @ params
    
    return pred
