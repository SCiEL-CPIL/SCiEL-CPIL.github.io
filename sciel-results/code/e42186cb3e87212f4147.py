"""
Improved program for unresolved-tendency closure task.
Uses scaled monomial fitting with adaptive Tikhonov regularization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    Uses scaled monomial basis with Tikhonov regularization and validation-based lambda selection.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Determine scaling parameters: mean and standard deviation
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    # Scale x to u
    u = (x - x_mean) / x_std

    def convert_scaled_to_original(c_u, x_mean, x_std):
        """
        Convert coefficients from scaled basis u = (x - mean)/std to original basis x.
        P(x) = sum_{k=0}^5 c_u[k] * ((x - mean)/std)^k
             = sum_{k=0}^5 c_u[k] / std^k * (x - mean)^k
             = sum_{k=0}^5 c_u[k] / std^k * sum_{j=0}^k binom(k, j) x^j (-mean)^(k-j)
        c_x[j] = sum_{k=j}^5 c_u[k] / std^k * binom(k, j) * (-mean)^(k-j)
        """
        c_x = np.zeros(6)
        for k in range(6):
            if abs(c_u[k]) < 1e-15:
                continue
            coeff_k = c_u[k] / (x_std ** k)
            for j in range(k + 1):
                c_x[j] += coeff_k * comb(k, j) * ((-x_mean) ** (k - j))
        return c_x

    def evaluate_monomial(c_x, x_data):
        """Evaluate polynomial sum c_j * x^j at x_data."""
        result = np.zeros_like(x_data)
        for j in range(6):
            result += c_x[j] * (x_data ** j)
        return result

    def fit_regularized(u_data, y_data, lam, weight_vec):
        """
        Fit scaled coefficients c_u minimizing ||V_u c_u - y||^2 + lam * sum(w_i c_u[i]^2).
        V_u is the Vandermonde matrix for u.
        """
        V = np.vander(u_data, N=6, increasing=True)
        # Normal equations: (V^T V + lam * W) c = V^T y
        # W is diagonal with weight_vec
        VtV = V.T @ V
        W = np.diag(weight_vec)
        A = VtV + lam * W
        b = V.T @ y_data
        
        try:
            c_u = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
            
        if not np.all(np.isfinite(c_u)):
            return None
        return c_u

    # Weight vector for regularization: penalize higher order terms more
    # This helps stabilize the fit and reduce oscillations
    weight_vec = np.array([1.0, 1.0, 2.0, 4.0, 8.0, 16.0])

    # Select lambda using validation subset
    # If n is small, just use a moderate lambda
    best_c_x = None
    best_val_mse = np.inf

    # Define lambda candidates
    lambdas = np.logspace(-8, 4, 20)

    # Check if we have enough data for CV
    use_cv = n > 50 and (time.time() - start_time) < (DEADLINE - 20.0)

    if use_cv:
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        val_size = max(10, n // 10)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        u_train = u[train_idx]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]

        for lam in lambdas:
            if time.time() > DEADLINE:
                break
                
            c_u_train = fit_regularized(u_train, y_train, lam, weight_vec)
            if c_u_train is None:
                continue
                
            c_x_candidate = convert_scaled_to_original(c_u_train, x_mean, x_std)
            if not np.all(np.isfinite(c_x_candidate)):
                continue
                
            pred_val = evaluate_monomial(c_x_candidate, x_val)
            mse_val = np.mean((pred_val - y_val) ** 2)
            
            if mse_val < best_val_mse:
                best_val_mse = mse_val
                # Refit on all data with this lambda to get final coefficients
                c_u_full = fit_regularized(u, y, lam, weight_vec)
                if c_u_full is not None:
                    c_x_full = convert_scaled_to_original(c_u_full, x_mean, x_std)
                    if np.all(np.isfinite(c_x_full)):
                        best_c_x = c_x_full

    # Fallback if CV didn't find a good solution or wasn't used
    if best_c_x is None:
        # Try a few fixed lambdas on all data
        for lam in [1e-6, 1e-4, 1e-2, 1.0, 10.0]:
            if time.time() > DEADLINE:
                break
            c_u = fit_regularized(u, y, lam, weight_vec)
            if c_u is not None:
                c_x = convert_scaled_to_original(c_u, x_mean, x_std)
                if np.all(np.isfinite(c_x)):
                    # Check training MSE as a heuristic
                    pred = evaluate_monomial(c_x, x)
                    mse = np.mean((pred - y) ** 2)
                    if best_c_x is None or mse < best_val_mse:
                        best_val_mse = mse
                        best_c_x = c_x

    # Final fallback: simple linear fit
    if best_c_x is None:
        try:
            # Fit linear in original x
            coeffs = np.polyfit(x, y, 1)
            c_x = np.zeros(6)
            c_x[1] = coeffs[0]
            c_x[0] = coeffs[1]
            best_c_x = c_x
        except Exception:
            best_c_x = np.zeros(6)

    return best_c_x
# EVOLVE-BLOCK-END
