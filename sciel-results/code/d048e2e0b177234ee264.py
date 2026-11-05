"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model.
    
    Args:
        training_x: Flattened array of slow-state values
        training_subgrid_tendency: Flattened array of unresolved tendencies
    
    Returns:
        Array of 6 finite real polynomial coefficients in ascending powers of x
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, with safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove any NaN/Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n = len(x)
    if n < 6:
        # Not enough data for degree-5 fit
        return np.zeros(6, dtype=np.float64)
    
    # Center and scale x to improve numerical stability and regularization effectiveness
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
        
    z = (x - mu) / sigma
    
    # Build Vandermonde matrix for scaled variable z
    powers = np.arange(6)
    V_z = z[:, None] ** powers  # shape (n, 6)
    
    # Check elapsed time
    if time.time() - start_time > DEADLINE - 5:
        # Fallback to linear fit on original x
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.array([coeff[0], coeff[1], 0, 0, 0, 0], dtype=np.float64)
        except:
            return np.zeros(6, dtype=np.float64)
    
    # Ridge regression on scaled variable with a wide range of regularization
    # Large lambda effectively reduces the degree of the polynomial
    best_coeff_z = np.zeros(6)
    best_mse = np.inf
    
    # Grid search for lambda. Include large values to allow for lower effective degrees.
    # The range is logarithmic from small to very large.
    lambdas = np.logspace(-10, 10, 21) # 1e-10 to 1e10
    
    # Precompute VtV and Vty for speed
    VtV_z = V_z.T @ V_z
    Vty_z = V_z.T @ y
    I = np.eye(6)
    
    for lam in lambdas:
        if time.time() - start_time > DEADLINE - 2:
            break
        try:
            A = VtV_z + lam * I
            coeff_z = np.linalg.solve(A, Vty_z)
            
            if not np.all(np.isfinite(coeff_z)):
                continue
            
            # Calculate MSE on training data to select best lambda
            # Note: This is biased towards low lambda (overfitting). 
            # However, the task score is on *validation* data.
            # Without validation data during training, we rely on the fact that
            # the true relationship is likely smooth and not extremely high-frequency.
            # A common heuristic is to use a moderate lambda or cross-validation.
            # Given the previous failure of complex fits, we should favor stronger regularization.
            
            # Let's evaluate a proxy: if the coefficients are huge, it's likely overfitting.
            # We can also just pick a fixed moderate lambda if cross-val is too slow.
            # Let's try to use a simple validation split if n is large enough, otherwise use a heuristic.
            
            if n > 100:
                # Simple cross-validation (2-fold) for lambda selection
                # To save time, we'll just pick the lambda that minimizes training MSE 
                # but with a penalty on coefficient magnitude? No, that's circular.
                # Let's stick to the idea that the previous program failed because it 
                # allowed too much complexity. We will bias the selection towards larger lambdas.
                
                # Heuristic: Prefer lambdas that result in smaller L2 norm of coefficients
                # while keeping MSE reasonable.
                mse = np.mean((V_z @ coeff_z - y) ** 2)
                l2_norm = np.linalg.norm(coeff_z)
                
                # Score function: minimize MSE + penalty * L2_norm^2
                # This is essentially what ridge does, but we are selecting lambda.
                # Let's just iterate and keep the one with best "generalization" proxy.
                # A good proxy for generalization in polynomial fits is the curvature or magnitude of high-order terms.
                
                # Let's simply collect a few candidates and pick the one with smallest high-order coefficients?
                # No, let's use the actual training MSE but with a strong prior on simplicity.
                
                # Actually, the safest bet given the previous drop in score is to use a 
                # fixed, reasonably strong regularization that shrinks high-order terms.
                # Let's try a few specific strong lambdas.
                
                pass # We will handle selection below
            else:
                pass
                
        except Exception:
            continue
            
    # Revised Strategy: 
    # Since we don't have validation data, and overfitting was the likely cause of the score drop,
    # we should select the lambda that minimizes the magnitude of the high-order coefficients 
    # or simply use a strong fixed regularization.
    
    # Let's evaluate a set of specific lambdas and pick the one that minimizes 
    # a combination of training error and model complexity (L2 norm of coeffs).
    # This is a form of model selection.
    
    candidate_lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    best_score = np.inf
    best_coeff_z = np.zeros(6)
    
    for lam in candidate_lambdas:
        if time.time() - start_time > DEADLINE - 2:
            break
        try:
            A = VtV_z + lam * I
            coeff_z = np.linalg.solve(A, Vty_z)
            
            if not np.all(np.isfinite(coeff_z)):
                continue
            
            mse = np.mean((V_z @ coeff_z - y) ** 2)
            complexity = np.sum(coeff_z**2)
            
            # We want to balance fit and simplicity.
            # AIC-like score: n * log(MSE) + k * log(n)
            # Or simply: MSE + alpha * complexity
            # Let's use a score that penalizes high-order coefficients more heavily.
            # Weighted complexity: sum(w_i * coeff_i^2) where w_i increases with i.
            weights = np.array([1, 10, 100, 1000, 10000, 100000])
            weighted_complexity = np.sum(weights * coeff_z**2)
            
            score = mse + 0.1 * weighted_complexity # Tune the 0.1 factor?
            
            if score < best_score:
                best_score = score
                best_coeff_z = coeff_z
        except Exception:
            continue
    
    if not np.all(np.isfinite(best_coeff_z)):
        # Fallback to simple linear fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.array([coeff[0], coeff[1], 0, 0, 0, 0], dtype=np.float64)
        except:
            return np.zeros(6, dtype=np.float64)
            
    # Transform coefficients from z-basis to x-basis
    # y = sum_{i=0}^5 a_i z^i
    # z = (x - mu) / sigma
    # z^i = (x - mu)^i / sigma^i
    # (x - mu)^i = sum_{j=0}^i binom(i, j) x^j (-mu)^{i-j}
    # So y = sum_{i=0}^5 a_i / sigma^i * sum_{j=0}^i binom(i, j) x^j (-mu)^{i-j}
    # c_j = sum_{i=j}^5 a_i / sigma^i * binom(i, j) * (-mu)^{i-j}
    
    c = np.zeros(6)
    for i in range(6):
        if i == 0:
            continue
        ai = best_coeff_z[i]
        if ai == 0:
            continue
        term_coeff = ai / (sigma ** i)
        # Expand (x - mu)^i
        for j in range(i + 1):
            binom_val = 1.0
            # Calculate binom(i, j)
            if j == 0 or j == i:
                binom_val = 1.0
            else:
                # Use math.comb if available, or manual calculation
                # Python 3.8+ has math.comb
                import math
                binom_val = float(math.comb(i, j))
            
            mu_power = ((-mu) ** (i - j))
            c[j] += term_coeff * binom_val * mu_power
            
    # Handle i=0 term separately for clarity
    c[0] += best_coeff_z[0]
    
    # Ensure finite
    if not np.all(np.isfinite(c)):
        return np.zeros(6, dtype=np.float64)
        
    return c.astype(np.float64)
# EVOLVE-BLOCK-END
