# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model.
    
    Uses ridge regression with multiple regularization strengths, selecting
    the best based on training performance with a bias-variance tradeoff.
    """
    start_time = time.time()
    DEADLINE = 110.0  # seconds, safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std
    
    # Build design matrix for degree 5 polynomial
    # Columns: 1, x, x^2, x^3, x^4, x^5
    X = np.ones((n, 6))
    for i in range(1, 6):
        X[:, i] = x_scaled ** i
    
    # Try multiple regularization strengths
    # Use a range of lambda values
    lambdas = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]
    
    best_coeff = None
    best_score = np.inf
    
    # Compute y statistics for normalization
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-30:
        y_var = 1.0
    
    for lam in lambdas:
        if time.time() - start_time > DEADLINE:
            break
        
        # Ridge regression: (X^T X + lambda*I) c = X^T y
        # Use diagonal loading for the identity matrix
        XTX = X.T @ X
        XTy = X.T @ y
        
        # Add regularization (don't regularize the constant term as heavily)
        reg = np.eye(6) * lam
        reg[0, 0] = lam * 0.1  # Lighter regularization on intercept
        
        try:
            A = XTX + reg
            c_scaled = np.linalg.solve(A, XTy)
        except np.linalg.LinAlgError:
            continue
        
        # Convert back to original x scale
        # c_scaled fits: y = sum(c_scaled[i] * (x - x_mean)/x_std)^i
        # We need coefficients for y = sum(c_orig[i] * x^i)
        # Use polynomial expansion
        c_orig = _convert_coefficients(c_scaled, x_mean, x_std)
        
        if not np.all(np.isfinite(c_orig)):
            continue
        
        # Compute training MSE
        pred = np.zeros(n)
        for i in range(5, -1, -1):
            pred = pred * x + c_orig[i]
        mse = np.mean((pred - y) ** 2)
        
        # Use a simple selection criterion: MSE + penalty for complexity
        # Higher degree terms should be penalized
        complexity_penalty = lam * 0.01
        score = mse + complexity_penalty
        
        if score < best_score:
            best_score = score
            best_coeff = c_orig
    
    if best_coeff is None:
        # Fallback: least squares with no regularization
        try:
            c_scaled = np.linalg.lstsq(X, y, rcond=None)[0]
            best_coeff = _convert_coefficients(c_scaled, x_mean, x_std)
        except:
            best_coeff = np.zeros(6)
    
    # Ensure all coefficients are finite
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)
    
    return best_coeff


def _convert_coefficients(c_scaled, x_mean, x_std):
    """
    Convert coefficients from scaled variable t = (x - x_mean)/x_std
    to original variable x.
    
    If y = sum(c_scaled[i] * t^i) where t = (x - x_mean)/x_std,
    find c_orig such that y = sum(c_orig[i] * x^i).
    
    Uses Taylor expansion: t = (x - x_mean)/x_std = x/x_std - x_mean/x_std
    """
    # c_scaled[i] * t^i = c_scaled[i] * (x/x_std - x_mean/x_std)^i
    # Let a = 1/x_std, b = -x_mean/x_std
    # t^i = (a*x + b)^i = sum_{j=0}^{i} binom(i,j) * (a*x)^j * b^(i-j)
    #     = sum_{j=0}^{i} binom(i,j) * a^j * b^(i-j) * x^j
    
    a = 1.0 / x_std
    b = -x_mean / x_std
    
    c_orig = np.zeros(6)
    for i in range(6):
        for j in range(i + 1):
            # binom(i, j)
            binom = _binom(i, j)
            c_orig[j] += c_scaled[i] * binom * (a ** j) * (b ** (i - j))
    
    return c_orig


def _binom(n, k):
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result *= (n - i)
        result //= (i + 1)
    return result
# EVOLVE-BLOCK-END
