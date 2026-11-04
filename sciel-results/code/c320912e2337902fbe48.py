"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses tridiagonal matrix eigenvalue parameterization for smooth optimization.
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute transformation matrix from standard basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)

    # c = A_T_inv @ p where p is standard-basis coeffs (ascending)
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute squared Laguerre norm of P(x) = prod(x - r_i).
        """
        # Build polynomial coefficients in standard basis (ascending)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        # Convert to Laguerre basis
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def roots_from_tridiagonal(params):
        """
        Convert tridiagonal matrix parameters to eigenvalues (roots).
        params: [d0, d1, ..., d11, b1, b2, ..., b11]
        where d_i are diagonal entries, b_i are off-diagonal entries.
        Returns sorted eigenvalues (positive).
        """
        N = 12
        diag = params[:N]
        offdiag = params[N:2*N-1]
        
        # Build tridiagonal matrix
        T = np.zeros((N, N))
        for i in range(N):
            T[i, i] = diag[i]
        for i in range(N-1):
            T[i, i+1] = offdiag[i]
            T[i+1, i] = offdiag[i]
        
        # Compute eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(T)
        except np.linalg.LinAlgError:
            return None
        
        # Ensure positive by shifting if needed
        if np.any(eigvals <= 0):
            shift = -np.min(eigvals) + 0.1
            eigvals = eigvals + shift
        
        return np.sort(eigvals)

    def tridiagonal_from_roots(roots):
        """
        Find tridiagonal matrix parameters that have the given roots as eigenvalues.
        Uses the fact that for a symmetric tridiagonal matrix with distinct eigenvalues,
        we can use the inverse eigenvalue problem.
        Here we use a simple approach: start with a diagonal matrix and add small off-diagonals.
        """
        N = 12
        roots = np.sort(roots)
        
        # Initialize with diagonal entries close to roots
        diag = roots.copy()
        
        # Small off-diagonal entries to make it non-diagonal
        offdiag = np.ones(N-1) * 0.1
        
        return np.concatenate([diag, offdiag])

    def cost_func(params):
        """
        Minimize upper bound on C4 ~ 1/||P||^2 using tridiagonal parameterization.
        """
        roots = roots_from_tridiagonal(params)
        
        if roots is None or np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    def cost_with_grad(params):
        """
        Cost and numerical gradient for tridiagonal parameterization.
        Uses central differences for gradient computation.
        """
        roots = roots_from_tridiagonal(params)
        
        if roots is None or np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10, np.zeros_like(params)
        
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10, np.zeros_like(params)
        
        cost = 1.0 / norm_sq
        
        # Numerical gradient using central differences
        grad = np.zeros_like(params)
        h = 1e-6
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += h
            params_minus[i] -= h
            
            cost_plus = cost_func(params_plus)
            cost_minus = cost_func(params_minus)
            
            grad[i] = (cost_plus - cost_minus) / (2 * h)
        
        return cost, grad

    # Optimization Strategy:
    # 1. Global search over tridiagonal matrix parameters using Differential Evolution.
    # 2. Final high-precision polish with L-BFGS-B using numerical gradient.
    
    # Initial seed: geometric sequence converted to tridiagonal parameters
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_params = tridiagonal_from_roots(seed_roots)
    
    # Bounds for tridiagonal parameters
    # Diagonal entries: allow range that covers reasonable root values
    diag_bounds = [(-10.0, 50.0)] * 12
    # Off-diagonal entries: small to moderate values
    offdiag_bounds = [(-5.0, 5.0)] * 11
    bounds = diag_bounds + offdiag_bounds
    
    best_roots = None
    best_val = 1e10
    best_params = None

    # Run Differential Evolution with multiple restarts
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed in seeds:
        if not time_ok():
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=80, 
                popsize=15, 
                tol=1e-8, 
                polish=False,
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
                best_roots = roots_from_tridiagonal(best_params)
                
        except Exception as e:
            pass

    # Fallback / Initial guess
    seed_val = cost_func(seed_params)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_params = seed_params
        best_roots = seed_roots

    # Final high-precision polish with L-BFGS-B using numerical gradient
    if best_params is not None and time_ok():
        try:
            polish_result = minimize(
                cost_with_grad,
                best_params,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_params = polish_result.x
                best_roots = roots_from_tridiagonal(best_params)
                
        except Exception as e:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
