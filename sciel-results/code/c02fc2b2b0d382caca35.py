"""Rewritten program: fit a shared degree-five polynomial to unresolved tendency."""
# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = 145.0  # safety margin before 150s wall-clock kill
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _check_deadline():
    return _elapsed() > DEADLINE


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model using Chebyshev basis for numerical stability.
    
    Returns 6 coefficients in ASCENDING powers of x.
    """
    training_x = np.asarray(training_x, dtype=np.float64).ravel()
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = training_x.size
    if n == 0:
        return np.zeros(6)
    
    # Ensure at least as many points as we need for degree 5
    if n < 6:
        # Fall back to least degree possible
        deg = max(1, n - 1)
        deg = min(deg, 5)
        coeff = np.zeros(6)
        try:
            c = np.polynomial.polynomial.polyfit(training_x, training_subgrid_tendency, deg)
            coeff[:deg+1] = c
        except Exception:
            coeff[0] = np.mean(training_subgrid_tendency)
        return coeff
    
    # Strategy: Fit in Chebyshev basis on normalized domain for superior conditioning
    # 1. Scale x to z in [-1, 1] using min-max normalization
    # 2. Fit Chebyshev series C(z) = sum c_k T_k(z)
    # 3. Convert to monomial basis in z
    # 4. Transform from z-space to x-space
    
    x_min = np.min(training_x)
    x_max = np.max(training_x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # Degenerate case: all x values are the same
        return np.array([np.mean(training_subgrid_tendency)] + [0.0]*5)
    
    # Map x to z in [-1, 1]
    # z = 2*(x - x_min)/(x_max - x_min) - 1
    # x = x_min + (z + 1)*(x_max - x_min)/2
    z = 2.0 * (training_x - x_min) / x_range - 1.0
    
    # Fit Chebyshev series of degree 5
    # Try multiple regularization strengths
    alphas = [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    
    best_coeff = None
    best_score = np.inf
    
    for alpha in alphas:
        if _check_deadline():
            break
        
        try:
            if alpha == 0.0:
                # Pure Chebyshev fit
                c_cheb = np.polynomial.chebyshev.chebfit(z, training_subgrid_tendency, 5)
            else:
                # Regularized Chebyshev fit using lstsq with Tikhonov
                # Build Chebyshev Vandermonde
                V_cheb = np.polynomial.chebyshev.chebvander(z, 5)
                A = V_cheb.T @ V_cheb + alpha * np.eye(6)
                b = V_cheb.T @ training_subgrid_tendency
                c_cheb = np.linalg.solve(A, b)
            
            # Convert Chebyshev coefficients to monomial coefficients in z
            # c_z = cheb2poly(c_cheb)
            c_z = np.polynomial.chebyshev.cheb2poly(c_cheb)
            c_z = np.zeros(6)
            c_z[:len(c_z)] = np.zeros(6)  # ensure length 6
            c_z = np.polynomial.chebyshev.cheb2poly(c_cheb)
            if len(c_z) < 6:
                c_z_full = np.zeros(6)
                c_z_full[:len(c_z)] = c_z
                c_z = c_z_full
            else:
                c_z = c_z[:6]
            
            # Now transform from z-space to x-space
            # z = (2/(x_max - x_min)) * x - (2*x_min/(x_max - x_min)) - 1
            # Let a = 2/(x_max - x_min), b = -(2*x_min/(x_max - x_min)) - 1
            # z = a*x + b
            # y = sum(c_z_i * z^i) = sum(c_z_i * (a*x + b)^i)
            # Expand using binomial theorem with exact recurrence
            
            a_val = 2.0 / x_range
            b_val = -(2.0 * x_min / x_range) - 1.0
            
            # Compute transformation matrix using recurrence to avoid cancellation
            # T[j, i] = coefficient of x^j in (a*x + b)^i
            # Use recurrence: (a*x + b)^0 = 1
            # (a*x + b)^i = (a*x + b) * (a*x + b)^(i-1)
            T = np.zeros((6, 6))
            T[0, 0] = 1.0  # (a*x+b)^0 = 1
            
            for i in range(1, 6):
                # (a*x + b)^i = a*x * (a*x+b)^(i-1) + b * (a*x+b)^(i-1)
                # If (a*x+b)^(i-1) = sum_{j=0}^{i-1} T[j, i-1] * x^j
                # Then a*x * (a*x+b)^(i-1) = sum_{j=0}^{i-1} a * T[j, i-1] * x^(j+1)
                # And b * (a*x+b)^(i-1) = sum_{j=0}^{i-1} b * T[j, i-1] * x^j
                for j in range(i):
                    T[j, i] = b * T[j, i-1]
                    if j + 1 <= i:
                        T[j+1, i] += a * T[j, i-1]
            
            c_original = T @ c_z
            
            # Compute training MSE as a proxy
            pred = np.polynomial.polynomial.polyval(training_x, c_original)
            mse = np.mean((pred - training_subgrid_tendency) ** 2)
            
            # Penalize high-degree coefficients to prevent overfitting
            coeff_penalty = np.sum(np.abs(c_original[3:]) ** 2) * 1e-3
            score = mse + coeff_penalty
            
            if score < best_score:
                best_score = score
                best_coeff = c_original.copy()
                
        except Exception:
            continue
    
    if best_coeff is None:
        # Fallback: simple linear fit
        try:
            c = np.polynomial.polynomial.polyfit(training_x, training_subgrid_tendency, 1)
            best_coeff = np.zeros(6)
            best_coeff[:2] = c
        except Exception:
            best_coeff = np.zeros(6)
            best_coeff[0] = np.mean(training_subgrid_tendency)
    
    # Final check: ensure all coefficients are finite
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)
        best_coeff[0] = np.mean(training_subgrid_tendency)
    
    return best_coeff
# EVOLVE-BLOCK-END
