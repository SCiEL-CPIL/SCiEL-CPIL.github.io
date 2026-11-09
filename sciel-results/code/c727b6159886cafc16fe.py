# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Uses a parametric Gaussian ansatz P(x,y) ∝ exp(-αx² - βy² - γxy) to reduce
    the 100-dimensional search to 3 parameters, then optimizes via gradient ascent
    on the entropy ratio D/M.
    """
    start_time = time.time()
    deadline = 620.0  # safety margin before 630s kill
    
    n = 10
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    
    # Meshgrid: P[y, x] indexing
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y
    
    # Projected values for each slope
    val_neg1 = X - Y       # x - y  (D = H_{-1})
    val_0 = X             # x      (H_0)
    val_1 = X + Y         # x + y  (H_1)
    val_2 = X + 2*Y       # x + 2y (H_2)
    
    def entropy_from_values(values, probs):
        """Compute Shannon entropy in bits from values and joint probs."""
        # Pool probabilities by unique values
        unique_vals, inverse = np.unique(values, return_inverse=True)
        pooled = np.zeros(len(unique_vals), dtype=np.float64)
        np.add.at(pooled, inverse, probs)
        # Ignore zero probabilities
        mask = pooled > 1e-15
        if not np.any(mask):
            return 0.0
        p = pooled[mask]
        p = p / p.sum()
        return -np.sum(p * np.log2(p))
    
    def compute_ratio(params):
        """Compute D/M ratio for given Gaussian parameters."""
        alpha, beta, gamma = params
        # Build log-probabilities
        log_p = -alpha * X**2 - beta * Y**2 - gamma * X * Y
        # Shift for numerical stability
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        total = p.sum()
        if total < 1e-300:
            return 0.0
        p = p / total
        
        D = entropy_from_values(val_neg1, p)
        H0 = entropy_from_values(val_0, p)
        H1 = entropy_from_values(val_1, p)
        H2 = entropy_from_values(val_2, p)
        M = max(H0, H1, H2)
        if M < 1e-9:
            return 0.0
        return D / M
    
    def neg_ratio(params):
        return -compute_ratio(params)
    
    # Initial guesses: try several starting points
    initial_guesses = [
        [0.0, 0.0, 0.0],      # uniform
        [0.1, 0.1, -0.1],     # slight correlation
        [0.05, 0.05, -0.05],
        [0.2, 0.2, -0.2],
        [0.1, 0.1, 0.1],
        [0.0, 0.0, 0.05],
        [0.05, 0.0, -0.05],
        [0.0, 0.05, -0.05],
    ]
    
    best_ratio = -np.inf
    best_params = None
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    # Evaluate initial guesses
    for guess in initial_guesses:
        if time.time() - start_time > deadline:
            break
        r = compute_ratio(guess)
        if r > best_ratio:
            best_ratio = r
            best_params = guess.copy()
            alpha, beta, gamma = guess
            log_p = -alpha * X**2 - beta * Y**2 - gamma * X * Y
            log_p -= np.max(log_p)
            p = np.exp(log_p)
            best_P = p / p.sum()
    
    # Use scipy minimize from the best initial point
    if best_params is not None:
        # Try multiple optimization methods
        for method in ['Nelder-Mead', 'Powell', 'BFGS']:
            if time.time() - start_time > deadline:
                break
            try:
                result = minimize(neg_ratio, best_params, method=method,
                                  options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                if time.time() - start_time > deadline:
                    break
                r = -result.fun
                if r > best_ratio:
                    best_ratio = r
                    best_params = result.x.copy()
                    alpha, beta, gamma = best_params
                    log_p = -alpha * X**2 - beta * Y**2 - gamma * X * Y
                    log_p -= np.max(log_p)
                    p = np.exp(log_p)
                    best_P = p / p.sum()
            except Exception:
                pass
        
        # Also try from other good starting points
        for guess in initial_guesses:
            if time.time() - start_time > deadline:
                break
            if np.allclose(guess, best_params, atol=1e-6):
                continue
            try:
                result = minimize(neg_ratio, guess, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
                if time.time() - start_time > deadline:
                    break
                r = -result.fun
                if r > best_ratio:
                    best_ratio = r
                    best_params = result.x.copy()
                    alpha, beta, gamma = best_params
                    log_p = -alpha * X**2 - beta * Y**2 - gamma * X * Y
                    log_p -= np.max(log_p)
                    p = np.exp(log_p)
                    best_P = p / p.sum()
            except Exception:
                pass
    
    # Final validation and normalization
    P = best_P.copy()
    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / total
    
    # Ensure no NaN or inf
    if not np.all(np.isfinite(P)):
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    return P


# EVOLVE-BLOCK-END
