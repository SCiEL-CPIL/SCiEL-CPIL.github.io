import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Uses Ridge Regression on scaled inputs for stability and generalization.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Flatten just in case
    x = x.flatten()
    y = y.flatten()
    
    N = len(x)
    if N == 0:
        return np.zeros(6)
    
    # 1. Scaling
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    if sigma_x < 1e-12:
        sigma_x = 1.0
        
    x_scaled = (x - mu_x) / sigma_x
    
    # 2. Build Vandermonde matrix for degree 5
    # V[:, j] = x_scaled^j
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # 3. Ridge Regression
    # Minimize ||V a - y||^2 + lambda ||a||^2
    # (V^T V + lambda I) a = V^T y
    
    # Choose lambda. 
    # The scale of V^T V is roughly N. 
    # We want a small penalty.
    # Let's try a few lambdas quickly if time permits, or just pick a robust one.
    # Given the fast execution, we can do a small grid search on lambda using 
    # a subset of data or just pick a standard small value.
    # Let's use a small value relative to the diagonal of V^T V.
    
    VtV = V.T @ V
    diag_mean = np.mean(np.diag(VtV))
    
    # Regularization strength relative to the data term
    # Typically lambda is in [1e-4, 1e-2] * diag_mean for mild regularization
    # Let's try lambda = 1e-3 * diag_mean
    
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2]
    
    # We will select the best lambda by minimizing the training error 
    # with a small penalty? No, that's circular.
    # Better: Just pick a small lambda. Overfitting on 6 params with N>100 is low.
    # The main risk is extrapolation to F=10. 
    # Ridge helps by shrinking coefficients.
    
    # Let's use lambda = 1e-3 * diag_mean
    lam = 1e-3 * diag_mean
    
    A = VtV + lam * np.eye(6)
    b = V.T @ y
    
    try:
        a = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        a, _, _, _ = np.linalg.lstsq(V, y, rcond=None)

    # Check time
    if time.time() - start_time > DEADLINE:
         # Return best effort
         pass

    # 4. Transform coefficients from scaled domain to original domain
    # We have P(u) = sum_{k=0}^5 a_k u^k where u = (x - mu) / sigma
    # We want Q(x) = sum_{j=0}^5 c_j x^j such that Q(x) = P((x-mu)/sigma)
    
    # Expansion:
    # u^k = ((x - mu) / sigma)^k = sigma^{-k} (x - mu)^k
    # (x - mu)^k = sum_{j=0}^k binom(k, j) x^j (-mu)^{k-j}
    
    # c_j = sum_{k=j}^5 a_k * sigma^{-k} * binom(k, j) * (-mu)^{k-j}
    
    c = np.zeros(6)
    for k in range(6):
        for j in range(k + 1):
            # binom(k, j)
            from math import comb
            coeff_factor = comb(k, j) * ((-mu_x) ** (k - j)) / (sigma_x ** k)
            c[j] += a[k] * coeff_factor
            
    # Ensure finite
    if not np.all(np.isfinite(c)):
        # Fallback to simple OLS on original x if something went wrong
        V_orig = np.vander(x, N=6, increasing=True)
        try:
             c, _, _, _ = np.linalg.lstsq(V_orig, y, rcond=None)
        except:
             c = np.zeros(6)
             c[0] = np.mean(y)
             
    return c
