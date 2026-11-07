"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    p = np.asarray(params, dtype=np.float64)
    if p.shape[0] != 10:
        p = p[:10] if p.size >= 10 else np.pad(p, (0, 10 - p.size))

    # Check deadline
    if time.time() > deadline:
        return np.zeros(n, dtype=np.float64)

    # Tensor-product cubic B-spline basis with 10 control points
    # We compute the cubic B-spline basis functions evaluated at x
    # and return their linear combination with params as weights.
    
    # Define knot vector for 10 cubic B-spline basis functions
    # Using uniform knots on [-3, 3] with appropriate clamping
    # For cubic B-splines with n basis functions, we need n+4 knots
    # We'll use a simple approach: define basis functions directly
    
    # Normalize x to [0, 1] for the B-spline parameter space
    xn = (x + 3.0) / 6.0  # Map [-3, 3] to [0, 1]
    
    # Define 10 cubic B-spline basis functions
    # Using the Cox-de Boor recursion for uniform knots
    # For simplicity and numerical stability, we'll use a set of
    # overlapping cubic B-spline basis functions
    
    # Knot vector: uniform with clamped ends
    # For 10 basis functions of degree 3, we need 10+3+1 = 14 knots
    # Using uniform spacing with clamped ends for stability
    num_basis = 10
    degree = 3
    
    # Create uniform knot vector with clamped ends
    # Interior knots evenly spaced, ends clamped
    interior_knots = np.linspace(0.0, 1.0, num_basis - degree + 1)
    # Clamped ends: repeat first and last knots (degree+1 times)
    knots = np.concatenate([
        np.full(degree + 1, interior_knots[0]),
        interior_knots[1:-1],
        np.full(degree + 1, interior_knots[-1])
    ])
    
    # Ensure knots are non-decreasing and handle numerical issues
    knots = np.maximum.accumulate(knots)
    
    # Compute B-spline basis functions using Cox-de Boor recursion
    # B_{i,0}(t) = 1 if knots[i] <= t < knots[i+1], else 0
    # B_{i,k}(t) = (t - knots[i])/(knots[i+k] - knots[i]) * B_{i,k-1}(t) +
    #               (knots[i+k+1] - t)/(knots[i+k+1] - knots[i+1]) * B_{i+1,k-1}(t)
    
    n_knots = len(knots)
    b = np.zeros((num_basis, n), dtype=np.float64)
    
    # Initialize degree 0
    for i in range(num_basis):
        if i + 1 < n_knots:
            b[i, :] = np.where((xn >= knots[i]) & (xn < knots[i + 1]), 1.0, 0.0)
    
    # Recurse up to degree 3
    for k in range(1, degree + 1):
        for i in range(num_basis):
            if i + k + 1 < n_knots:
                # First term
                denom1 = knots[i + k] - knots[i]
                if denom1 > 1e-12:
                    term1 = (xn - knots[i]) / denom1 * b[i, :]
                else:
                    term1 = np.zeros(n, dtype=np.float64)
                
                # Second term
                denom2 = knots[i + k + 1] - knots[i + 1]
                if denom2 > 1e-12:
                    term2 = (knots[i + k + 1] - xn) / denom2 * b[i + 1, :]
                else:
                    term2 = np.zeros(n, dtype=np.float64)
                
                b[i, :] = term1 + term2
    
    # Linear combination of basis functions with parameters
    result = b @ p
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
