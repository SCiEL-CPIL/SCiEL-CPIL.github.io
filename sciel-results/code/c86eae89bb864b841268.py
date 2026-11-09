"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = 110.0  # 10s buffer before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = x.shape[0]
    if n == 0:
        return np.zeros(6, dtype=np.float64)
        
    if n != y.shape[0]:
        m = min(n, y.shape[0])
        x = x[:m]
        y = y[:m]
        n = m
        if n == 0:
            return np.zeros(6, dtype=np.float64)

    # Handle degenerate case
    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    if x_std < 1e-14:
        c = np.zeros(6, dtype=np.float64)
        c[0] = float(np.mean(y))
        return c

    # Standardize x for numerical stability
    u = (x - x_mean) / x_std
    
    # Standardize y as well to keep coefficients in a reasonable range
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std < 1e-14:
        y_std = 1.0
    y_hat = (y - y_mean) / y_std

    # Build Vandermonde matrix for u
    V = np.vander(u, N=6, increasing=True)
    
    # We want to find coefficients w in the standardized basis:
    # P(x) = y_mean + y_std * sum_{j=0}^5 w_j * u^j
    # where u = (x - x_mean) / x_std
    
    # We will use ridge regression on the standardized basis.
    # The intercept term (w_0) should not be regularized.
    
    # Try a small grid of regularization parameters
    # Since we don't have a separate validation set provided in the inputs 
    # (only training_x, training_subgrid_tendency), we rely on a small holdout 
    # from the training data if n is large enough, otherwise just pick a 
    # conservative regularization.
    
    # Given the "unseen forcing regime" in validation, some regularization 
    # is likely beneficial to prevent overfitting to training noise.
    
    # Let's do a simple 5-fold cross-validation or a holdout to select lambda.
    # But to be safe on time, we'll just try a few lambdas and pick the best 
    # on a 10% holdout if n > 100, otherwise use a fixed small lambda.
    
    use_holdout = n >= 200
    if use_holdout:
        # Shuffle indices
        indices = np.random.permutation(n)
        split_idx = int(0.9 * n)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        V_train = V[train_idx]
        y_train = y_hat[train_idx]
        V_val = V[val_idx]
        y_val = y_hat[val_idx]
    else:
        V_train = V
        y_train = y_hat
        V_val = None
        y_val = None

    # Regularization parameters to try
    # Note: We regularize the coefficients of the polynomial in u.
    # A larger lambda penalizes higher-order terms more if we weight them,
    # but standard ridge penalizes all non-intercept terms equally.
    # To specifically penalize higher-order terms (which are more prone to overfitting
    # and oscillation), we could use a weighted ridge.
    # Let's stick to standard ridge first, but maybe scale lambda by the magnitude
    # of the Gram matrix.
    
    lambdas = [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    best_w = None
    best_err = np.inf
    
    for lam in lambdas:
        if time.time() - t0 > deadline:
            break
            
        if lam == 0.0:
            try:
                # Unregularized least squares
                w, residuals, rank, sv = np.linalg.lstsq(V_train, y_train, rcond=None)
            except np.linalg.LinAlgError:
                continue
        else:
            # Ridge regression
            # (V^T V + lambda * I) w = V^T y
            # But we don't regularize the intercept (index 0).
            A = V_train.T @ V_train
            b = V_train.T @ y_train
            
            # Create regularization matrix
            reg = np.diag(np.ones(6))
            reg[0, 0] = 0.0  # No regularization on intercept
            
            # Add regularization
            A_reg = A + lam * reg
            
            try:
                w = np.linalg.solve(A_reg, b)
            except np.linalg.LinAlgError:
                # Fallback to lstsq if singular (should be rare with ridge)
                try:
                    w, *_ = np.linalg.lstsq(V_train, y_train, rcond=None)
                except np.linalg.LinAlgError:
                    continue

        # Evaluate error
        if V_val is not None:
            pred_val = V_val @ w
            err = float(np.mean((pred_val - y_val) ** 2))
        else:
            # If no holdout, just use training error (risk of overfitting)
            pred_train = V_train @ w
            err = float(np.mean((pred_train - y_train) ** 2))
            
        if err < best_err:
            best_err = err
            best_w = w

    if best_w is None:
        # Fallback: constant polynomial
        best_w = np.zeros(6, dtype=np.float64)
        best_w[0] = 1.0  # Since y_hat has mean 0, w_0 should be ~0, but let's be safe

    # Convert coefficients from standardized basis (u) to raw basis (x)
    # We have: P(x) = y_mean + y_std * sum_{j=0}^5 w_j * u^j
    # where u = (x - x_mean) / x_std
    # So: P(x) = y_mean + y_std * sum_{j=0}^5 w_j * ((x - x_mean) / x_std)^j
    #         = y_mean + sum_{j=0}^5 (y_std * w_j / x_std^j) * (x - x_mean)^j
    
    # Let a_j = y_std * w_j / x_std^j
    # Then P(x) = y_mean + sum_{j=0}^5 a_j * (x - x_mean)^j
    
    # We need to expand (x - x_mean)^j into powers of x:
    # (x - x_mean)^j = sum_{k=0}^j C(j, k) * x^k * (-x_mean)^(j-k)
    
    # So P(x) = y_mean + sum_{j=0}^5 a_j * sum_{k=0}^j C(j, k) * x^k * (-x_mean)^(j-k)
    #         = y_mean + sum_{k=0}^5 x^k * [ sum_{j=k}^5 a_j * C(j, k) * (-x_mean)^(j-k) ]
    
    # The coefficient of x^k in the final polynomial is:
    # c_k = sum_{j=k}^5 a_j * C(j, k) * (-x_mean)^(j-k)
    # And c_0 also has the y_mean added?
    # Wait, P(x) = y_mean + sum_{k=0}^5 c_k x^k
    # So the final coefficient vector C returned by construct should be:
    # C[0] = y_mean + c_0
    # C[k] = c_k for k > 0
    
    a = np.zeros(6, dtype=np.float64)
    for j in range(6):
        a[j] = y_std * best_w[j] / (x_std ** j)
        
    C = np.zeros(6, dtype=np.float64)
    for k in range(6):
        c_k = 0.0
        for j in range(k, 6):
            c_k += a[j] * comb(j, k) * ((-x_mean) ** (j - k))
        C[k] = c_k
        
    C[0] += y_mean
    
    # Final check for finite values
    if not np.all(np.isfinite(C)):
        # Fallback to linear fit if something went wrong
        try:
            coeff_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            C = np.zeros(6, dtype=np.float64)
            C[:2] = coeff_lin
        except Exception:
            C = np.zeros(6, dtype=np.float64)
            C[0] = np.mean(y)
            
    return C

# EVOLVE-BLOCK-END
