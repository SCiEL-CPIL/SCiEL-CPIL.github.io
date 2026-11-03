"""
Corrected program for the uncertainty inequality task.

The previous program had a critical mathematical error in the construction of the
P^2 (momentum squared) matrix in the Hermite basis. The off-diagonal elements of
P^2 have opposite signs to those of X^2. This correction allows for a valid
minimization of the uncertainty bound.

Strategy:
1. Construct X^2 and P^2 matrices correctly.
2. Define the objective function as the uncertainty product bound derived from these matrices.
3. Use Differential Evolution for global optimization to avoid local minima.
4. Enforce the g(0) = 0 constraint by expressing c12 in terms of c0, c4, c8.
5. Return the optimal coefficients [c0, c4, c8].
"""

import time
import numpy as np
from scipy.optimize import differential_evolution

DEADLINE_MARGIN = 10.0
MAX_RUN_TIME = 200.0

def construct():
    """
    Returns a numpy array of shape (3,) with coefficients [c0, c4, c8].
    The coefficient of H12 is implicitly determined by the constraint g(0) = 0.
    """
    t0 = time.time()
    
    # Basis indices
    basis_idx = [0, 4, 8, 12]
    size = 13  # n=0..12
    
    # Initialize matrices
    X2 = np.zeros((size, size))
    P2 = np.zeros((size, size))
    
    # Build X^2 and P^2
    for n in range(size):
        # Diagonal elements
        X2[n, n] = n + 0.5
        P2[n, n] = n + 0.5
        
        # Off-diagonal elements
        # X^2_{n, n-2} = 0.5 * sqrt(n(n-1))
        # P^2_{n, n-2} = -0.5 * sqrt(n(n-1))
        
        if n >= 2:
            coeff = 0.5 * np.sqrt(n * (n - 1))
            X2[n, n-2] = coeff
            X2[n-2, n] = coeff
            
            P2[n, n-2] = -coeff
            P2[n-2, n] = -coeff
            
        if n < size - 2:
            coeff = 0.5 * np.sqrt((n + 1) * (n + 2))
            X2[n, n+2] = coeff
            X2[n+2, n] = coeff
            
            P2[n, n+2] = -coeff
            P2[n+2, n] = -coeff

    # Extract submatrices for basis {0, 4, 8, 12}
    X2_sub = np.zeros((4, 4))
    P2_sub = np.zeros((4, 4))
    for i, ni in enumerate(basis_idx):
        for j, nj in enumerate(basis_idx):
            X2_sub[i, j] = X2[ni, nj]
            P2_sub[i, j] = P2[ni, nj]
    
    # Compute psi_n(0) for n=0,4,8,12
    # psi_0(0) = pi^(-1/4)
    # psi_n(0) = -sqrt((n-1)/n) * psi_{n-2}(0) for even n
    psi_vals = np.zeros(size)
    psi_vals[0] = np.pi ** (-0.25)
    for n in range(2, size, 2):
        psi_vals[n] = -np.sqrt((n-1)/n) * psi_vals[n-2]
        
    h0_vec = np.array([psi_vals[0], psi_vals[4], psi_vals[8], psi_vals[12]])
    
    def get_c12(c0, c4, c8):
        # c0*psi0(0) + c4*psi4(0) + c8*psi8(0) + c12*psi12(0) = 0
        return -(c0 * h0_vec[0] + c4 * h0_vec[1] + c8 * h0_vec[2]) / h0_vec[3]
    
    def objective(c_free):
        c0, c4, c8 = c_free
        c12 = get_c12(c0, c4, c8)
        c = np.array([c0, c4, c8, c12])
        
        norm_sq = np.dot(c, c)
        if norm_sq < 1e-12:
            return np.inf
            
        x2_val = np.dot(c, np.dot(X2_sub, c))
        p2_val = np.dot(c, np.dot(P2_sub, c))
        
        # Ensure positive for sqrt
        if x2_val < 1e-12 or p2_val < 1e-12:
            return np.inf
            
        return np.sqrt(x2_val * p2_val) / norm_sq

    # Seed value
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = objective(c_seed)
    best_c = c_seed.copy()
    
    # Bounds for Differential Evolution
    # Based on seed magnitude, allow reasonable exploration
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
    
    # Run Differential Evolution
    try:
        if time.time() - t0 < MAX_RUN_TIME:
            de_res = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-8,
                polish=True,
                time_limit=max(1, MAX_RUN_TIME - (time.time() - t0) - DEADLINE_MARGIN)
            )
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_c = de_res.x.copy()
    except Exception as e:
        pass
    
    # If no improvement or error, return seed
    return np.array(best_c)

if __name__ == "__main__":
    print(construct())
