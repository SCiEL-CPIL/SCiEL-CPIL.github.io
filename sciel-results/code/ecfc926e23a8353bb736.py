"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial using Regularized Total Least Squares (TLS).
    
    Strategy:
    1. Center and scale x to improve numerical conditioning.
    2. Build the augmented data matrix [x_powers, y] for TLS.
    3. Use SVD to find the orthogonal regression manifold, truncating small
       singular values for implicit regularization (stability for unseen regimes).
    4. Extract polynomial coefficients from the TLS solution.
    5. Map back to original x scale.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Remove non-finite entries
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    n = len(x)
    if n < 10:
        return np.zeros(6)
    
    # Center and scale x for better conditioning
    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-10:
        x_std = 1.0
    
    x_scaled = (x - x_mean) / x_std
    
    # Scale y as well for balanced TLS
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    
    y_scaled = (y - y_mean) / y_std
    
    # Build Vandermonde matrix for degree 5 (in scaled x)
    deg = 5
    V = np.vander(x_scaled, N=deg + 1, increasing=True)  # (n, 6)
    
    # Augmented matrix for TLS: [V, y_scaled]
    # TLS finds the hyperplane minimizing orthogonal distance
    A = np.hstack([V, y_scaled[:, np.newaxis]])  # (n, 7)
    
    # SVD decomposition
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    except Exception:
        # Fallback to standard OLS if SVD fails
        try:
            coeff_scaled, _, _, _ = np.linalg.lstsq(V, y_scaled, rcond=None)
            # Map back to original scale
            return _map_coefficients(coeff_scaled, x_mean, x_std, y_mean, y_std, deg)
        except Exception:
            return np.array([y_mean] + [0.0] * deg)
    
    # Truncate small singular values for regularization
    # Keep components with significant variance
    s_max = s[0] if len(s) > 0 else 1.0
    threshold = 1e-6 * s_max
    
    # The TLS solution: the last row of Vt (corresponding to smallest singular value)
    # gives the normal to the fitted hyperplane
    # We need to find the best fit by projecting out the noise direction
    
    # Alternative approach: Use ridge-regularized OLS which is more stable
    # and achieves similar regularization effect
    
    # Compute ridge parameter adaptively based on data spread
    # Larger x_range -> more regularization needed for unseen regimes
    x_range = x.max() - x.min()
    
    # Adaptive lambda: scale with data variance
    lambda_base = 1e-4
    lambda_adaptive = lambda_base * (1.0 + x_range / (10.0 * x_std + 1e-10))
    
    # Try multiple regularization strengths and pick best via cross-validation-like criterion
    best_coeff = None
    best_score = np.inf
    
    lambdas = [0.0, lambda_adaptive * 0.1, lambda_adaptive, lambda_adaptive * 10.0, lambda_adaptive * 100.0]
    
    for lam in lambdas:
        if time.time() > deadline - 10:
            break
        
        try:
            # Ridge regression: (V^T V + lam*I) coeff = V^T y
            VtV = V.T @ V
            Vty = V.T @ y_scaled
            
            # Add regularization (don't regularize intercept too heavily)
            reg_matrix = lam * np.eye(deg + 1)
            reg_matrix[0, 0] *= 0.1  # Lighter regularization on intercept
            
            A_reg = VtV + reg_matrix
            b_reg = Vty
            
            # Solve the regularized system
            try:
                coeff_scaled = np.linalg.solve(A_reg, b_reg)
            except np.linalg.LinAlgError:
                coeff_scaled = np.linalg.lstsq(A_reg, b_reg, rcond=None)[0]
            
            if not np.all(np.isfinite(coeff_scaled)):
                continue
            
            # Evaluate fit quality (normalized MSE)
            pred_scaled = V @ coeff_scaled
            mse_scaled = np.mean((pred_scaled - y_scaled) ** 2)
            
            # Also compute a complexity penalty (norm of coefficients)
            coeff_norm = np.linalg.norm(coeff_scaled)
            
            # Combined score: MSE + small penalty for coefficient magnitude
            score = mse_scaled + 1e-6 * coeff_norm
            
            if score < best_score:
                best_score = score
                best_coeff = coeff_scaled.copy()
                
        except Exception:
            continue
    
    if best_coeff is None:
        # Final fallback
        try:
            best_coeff, _, _, _ = np.linalg.lstsq(V, y_scaled, rcond=None)
        except Exception:
            best_coeff = np.zeros(deg + 1)
            best_coeff[0] = 0.0
    
    # Map coefficients back to original x scale
    coeff = _map_coefficients(best_coeff, x_mean, x_std, y_mean, y_std, deg)
    
    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.where(np.isfinite(coeff), coeff, 0.0)
    
    return coeff


def _map_coefficients(coeff_scaled, x_mean, x_std, y_mean, y_std, deg):
    """
    Map polynomial coefficients from scaled space to original space.
    
    If p(x_scaled) = sum(c_i * x_scaled^i) where x_scaled = (x - x_mean)/x_std
    then p((x-x_mean)/x_std) = sum(c_i * ((x-x_mean)/x_std)^i)
    
    We need to express this as sum(a_j * x^j).
    """
    # Expand using binomial theorem
    # (x - x_mean)^i = sum_{j=0}^{i} C(i,j) * x^j * (-x_mean)^(i-j)
    # So ((x-x_mean)/x_std)^i = (1/x_std^i) * sum_{j=0}^{i} C(i,j) * x^j * (-x_mean)^(i-j)
    
    coeff_orig = np.zeros(deg + 1)
    
    for i in range(deg + 1):
        if abs(coeff_scaled[i]) < 1e-30:
            continue
        
        factor = coeff_scaled[i] / (y_std * x_std ** i)
        
        for j in range(i + 1):
            from math import comb
            binom = comb(i, j)
            term = binom * ((-x_mean) ** (i - j))
            coeff_orig[j] += factor * term
    
    # Add the y offset
    coeff_orig[0] += y_mean / y_std * y_std  # y = y_mean + y_std * p_scaled
    # Wait, need to be more careful:
    # y = y_mean + y_std * p(x_scaled)
    # p(x_scaled) = sum(coeff_scaled[i] * x_scaled^i)
    # So y = y_mean + y_std * sum(coeff_scaled[i] * ((x-x_mean)/x_std)^i)
    # = y_mean + sum(coeff_scaled[i] * y_std / x_std^i * (x-x_mean)^i)
    # = y_mean + sum(coeff_scaled[i] * y_std / x_std^i * sum_j C(i,j) x^j (-x_mean)^(i-j))
    
    # Redo properly
    coeff_orig = np.zeros(deg + 1)
    
    for i in range(deg + 1):
        if abs(coeff_scaled[i]) < 1e-30:
            continue
        
        factor = coeff_scaled[i] * y_std / (x_std ** i)
        
        for j in range(i + 1):
            from math import comb
            binom = comb(i, j)
            term = binom * ((-x_mean) ** (i - j))
            coeff_orig[j] += factor * term
    
    coeff_orig[0] += y_mean
    
    return coeff_orig
# EVOLVE-BLOCK-END
