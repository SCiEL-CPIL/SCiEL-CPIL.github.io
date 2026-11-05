"""Climate dynamics: unresolved-tendency closure via robust polynomial regression."""
import time
import numpy as np
from scipy import stats

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    
    Strategy:
    1. Standardize x to improve numerical conditioning.
    2. Use Chebyshev basis for stable polynomial evaluation.
    3. Robust regression (IRLS) to handle outliers from extreme fast-scale states.
    4. Regularization (Ridge) to prevent overfitting of high-order terms.
    5. Convert back to monomial coefficients in ascending powers of x.
    """
    t0 = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    if n < 6:
        # Not enough data; return zeros
        return np.zeros(6)
    
    # Handle edge cases
    if np.allclose(x, 0):
        return np.zeros(6)
    
    # Standardize x for better conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # Build Chebyshev design matrix (degree 0-5)
    # T_k(x) = cos(k * arccos(x)) for x in [-1, 1]
    # For x outside [-1,1], use the recursive definition
    def chebyshev_matrix(x_vals, deg=5):
        n_pts = len(x_vals)
        T = np.zeros((n_pts, deg + 1))
        T[:, 0] = 1.0
        T[:, 1] = x_vals
        for k in range(2, deg + 1):
            T[:, k] = 2 * x_vals * T[:, 1] - T[:, 0]
            T[:, 0], T[:, 1] = T[:, 1], T[:, k]
        return T
    
    # Clip x_norm to [-1, 1] for Chebyshev stability, but use recursive form
    # which works for any real x
    T = chebyshev_matrix(x_norm, 5)
    
    # Robust IRLS regression with Ridge regularization
    def irls_robust_fit(T_mat, y_vals, max_iter=20, tol=1e-6, ridge_lambda=0.01):
        """IRLS robust regression with ridge penalty."""
        n_pts, n_params = T_mat.shape
        
        # Initial weights: uniform
        w = np.ones(n_pts)
        
        # Ridge matrix
        ridge = ridge_lambda * np.eye(n_params)
        ridge[0, 0] = 0.0  # Don't penalize intercept
        
        prev_coeff = None
        
        for it in range(max_iter):
            # Weighted normal equations
            W = np.diag(w)
            A = T_mat.T @ (W @ T_mat) + ridge
            b = T_mat.T @ (w * y_vals)
            
            try:
                coeff = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Compute residuals
            resid = y_vals - T_mat @ coeff
            
            # Update weights using Huber loss
            sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
            if sigma < 1e-12:
                sigma = 1.0
            delta = 1.5 * sigma  # Huber threshold
            abs_resid = np.abs(resid)
            w = np.where(abs_resid <= delta, 1.0, delta / abs_resid)
            w = np.maximum(w, 1e-6)  # Avoid zero weights
            
            # Check convergence
            if prev_coeff is not None:
                if np.max(np.abs(coeff - prev_coeff)) < tol:
                    break
            prev_coeff = coeff
        
        return coeff
    
    # Fit with moderate regularization
    coeff_cheb = irls_robust_fit(T, y, max_iter=15, tol=1e-5, ridge_lambda=0.01)
    
    elapsed = time.time() - t0
    if elapsed > deadline:
        # Convert and return immediately
        monomial_coeff = chebyshev_to_monomial(coeff_cheb, x_mean, x_std)
        return monomial_coeff
    
    # Convert Chebyshev coefficients to monomial coefficients in x
    # T_k(x_norm) where x_norm = (x - x_mean) / x_std
    # We need: sum_{k=0}^{5} c_k * T_k((x - x_mean)/x_std) = sum_{i=0}^{5} a_i * x^i
    #
    # First, express T_k(z) in monomials of z, then substitute z = (x - x_mean)/x_std
    # and expand.
    
    # Chebyshev polynomials in monomial form (up to degree 5):
    # T_0(z) = 1
    # T_1(z) = z
    # T_2(z) = 2z^2 - 1
    # T_3(z) = 4z^3 - 3z
    # T_4(z) = 8z^4 - 8z^2 + 1
    # T_5(z) = 16z^5 - 20z^3 + 5z
    
    # Build transformation matrix from Chebyshev to monomials of z
    # M[i, k] = coefficient of z^i in T_k(z)
    M_cheb_to_z = np.zeros((6, 6))
    M_cheb_to_z[0, 0] = 1
    M_cheb_to_z[1, 1] = 1
    M_cheb_to_z[0, 2] = -1;  M_cheb_to_z[2, 2] = 2
    M_cheb_to_z[1, 3] = -3;  M_cheb_to_z[3, 3] = 4
    M_cheb_to_z[0, 4] = 1;   M_cheb_to_z[2, 4] = -8;  M_cheb_to_z[4, 4] = 8
    M_cheb_to_z[1, 5] = 5;   M_cheb_to_z[3, 5] = -20; M_cheb_to_z[5, 5] = 16
    
    # Coefficients in powers of z: d_i = sum_k c_k * M[i, k]
    d = M_cheb_to_z @ coeff_cheb
    
    # Now z = (x - x_mean) / x_std = (1/x_std) * x - (x_mean/x_std)
    # Let a = 1/x_std, b = -x_mean/x_std
    # z = a*x + b
    # z^i = sum_{j=0}^{i} binom(i, j) * a^j * b^(i-j) * x^j
    
    a = 1.0 / x_std
    b = -x_mean / x_std
    
    # Convert from powers of z to powers of x
    # monomial_coeff[j] = sum_{i=j}^{5} d[i] * coeff_of_x^j_in_z^i
    monomial_coeff = np.zeros(6)
    for i in range(6):
        for j in range(i + 1):
            # Coefficient of x^j in (a*x + b)^i is binom(i, j) * a^j * b^(i-j)
            from math import comb
            c_j = comb(i, j) * (a ** j) * (b ** (i - j))
            monomial_coeff[j] += d[i] * c_j
    
    # Verify: check that the polynomial fits the training data reasonably
    # (quick sanity check)
    pred = np.zeros(n)
    for i in range(6):
        pred += monomial_coeff[i] * (x ** i)
    mse = np.mean((pred - y) ** 2)
    var_y = np.var(y)
    if var_y > 1e-12:
        nrmse = mse / var_y
        # If something went terribly wrong, fall back to simple linear fit
        if nrmse > 10.0 or not np.all(np.isfinite(monomial_coeff)):
            coeff_linear = np.polynomial.polynomial.polyfit(x, y, 1)
            monomial_coeff = np.zeros(6)
            monomial_coeff[:2] = coeff_linear
    
    # Final check for finite values
    if not np.all(np.isfinite(monomial_coeff)):
        monomial_coeff = np.zeros(6)
    
    return monomial_coeff


def chebyshev_to_monomial(coeff_cheb, x_mean, x_std):
    """Convert Chebyshev coefficients (in normalized x) to monomial coefficients in x."""
    # Chebyshev polynomials in monomial form (up to degree 5)
    M_cheb_to_z = np.zeros((6, 6))
    M_cheb_to_z[0, 0] = 1
    M_cheb_to_z[1, 1] = 1
    M_cheb_to_z[0, 2] = -1;  M_cheb_to_z[2, 2] = 2
    M_cheb_to_z[1, 3] = -3;  M_cheb_to_z[3, 3] = 4
    M_cheb_to_z[0, 4] = 1;   M_cheb_to_z[2, 4] = -8;  M_cheb_to_z[4, 4] = 8
    M_cheb_to_z[1, 5] = 5;   M_cheb_to_z[3, 5] = -20; M_cheb_to_z[5, 5] = 16
    
    d = M_cheb_to_z @ coeff_cheb
    
    a = 1.0 / x_std
    b = -x_mean / x_std
    
    from math import comb
    monomial_coeff = np.zeros(6)
    for i in range(6):
        for j in range(i + 1):
            c_j = comb(i, j) * (a ** j) * (b ** (i - j))
            monomial_coeff[j] += d[i] * c_j
    
    return monomial_coeff
