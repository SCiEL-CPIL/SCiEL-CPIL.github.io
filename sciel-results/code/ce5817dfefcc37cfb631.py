"""Structural change: exploit antisymmetry of fast-slow coupling by fitting odd-only polynomial,
combined with adaptive regularization and robust cross-validation."""
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    t0 = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n == 0:
        return np.zeros(6)

    if n < 3:
        c = np.zeros(6)
        c[0] = y.mean()
        return c

    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min

    if x_range < 1e-12:
        c = np.zeros(6)
        c[0] = y.mean()
        return c

    # Center x at 0 to exploit antisymmetry: fit y ≈ a1*x + a3*x^3 + a5*x^5
    x_c = x - 0.5 * (x_min + x_max)
    x_s = x_c / (0.5 * x_range + 1e-300)  # scale to roughly [-1, 1]

    # Strategy 1: Odd-only polynomial (exploits antisymmetry)
    # y ≈ c1*x + c3*x^3 + c5*x^5 in original x space
    # In scaled space: y ≈ a1*x_s + a3*x_s^3 + a5*x_s^5
    # Then convert back to original x coefficients

    def fit_odd_polynomial(x_s, y, n_samples):
        """Fit odd-only polynomial y ≈ a1*x_s + a3*x_s^3 + a5*x^5 with ridge regularization."""
        # Build odd Vandermonde: columns are x_s, x_s^3, x_s^5
        V = np.column_stack([x_s, x_s**3, x_s**5])
        
        best_lam_mse = np.inf
        best_coeff = None
        
        # Try a range of ridge parameters
        lams = [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
        
        for lam in lams:
            try:
                if lam == 0.0:
                    # Check condition number
                    A = V.T @ V
                    if np.linalg.cond(A) > 1e14:
                        continue
                    coeff = np.linalg.solve(A, V.T @ y)
                else:
                    A = V.T @ V + lam * np.eye(3)
                    coeff = np.linalg.solve(A, V.T @ y)
                
                if not np.all(np.isfinite(coeff)):
                    continue
                
                pred = V @ coeff
                mse = np.mean((pred - y) ** 2)
                
                if np.isfinite(mse) and mse < best_lam_mse:
                    best_lam_mse = mse
                    best_coeff = coeff
            except Exception:
                continue
        
        return best_coeff, best_lam_mse

    # Strategy 2: Full polynomial with strong regularization
    def fit_full_polynomial(x_s, y, n_samples, max_deg=5):
        """Fit full polynomial with ridge regularization."""
        best_lam_mse = np.inf
        best_coeff = None
        
        for deg in range(0, max_deg + 1):
            d = deg + 1
            if n_samples < d:
                continue
            
            V = np.polynomial.polynomial.polyvander(x_s, deg)
            
            lams = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]
            for lam in lams:
                try:
                    if lam == 0.0:
                        A = V.T @ V
                        if np.linalg.cond(A) > 1e14:
                            continue
                        coeff = np.linalg.solve(A, V.T @ y)
                    else:
                        A = V.T @ V + lam * np.eye(d)
                        coeff = np.linalg.solve(A, V.T @ y)
                    
                    if not np.all(np.isfinite(coeff)):
                        continue
                    
                    pred = V @ coeff
                    mse = np.mean((pred - y) ** 2)
                    
                    if np.isfinite(mse) and mse < best_lam_mse:
                        best_lam_mse = mse
                        best_coeff = coeff
                except Exception:
                    continue
        
        return best_coeff, best_lam_mse

    # Cross-validation for model selection
    k = min(5, max(2, n // 500))
    if k < 2:
        k = 2
    rng = np.random.default_rng(42)
    fold_ids = rng.permutation(n) % k

    # Evaluate odd-only model
    odd_cv_mse = np.zeros(k)
    for f in range(k):
        if time.time() > deadline:
            break
        val_mask = (fold_ids == f)
        tr_mask = ~val_mask
        if np.sum(tr_mask) < 3:
            odd_cv_mse[f] = np.inf
            continue
        
        coeff_tr, _ = fit_odd_polynomial(x_s[tr_mask], y[tr_mask], np.sum(tr_mask))
        if coeff_tr is not None:
            V_val = np.column_stack([x_s[val_mask], x_s[val_mask]**3, x_s[val_mask]**5])
            pred = V_val @ coeff_tr
            odd_cv_mse[f] = np.mean((pred - y[val_mask]) ** 2)
        else:
            odd_cv_mse[f] = np.inf

    odd_cv_mean = np.mean(odd_cv_mse)
    if not np.isfinite(odd_cv_mean):
        odd_cv_mean = np.inf

    # Evaluate full polynomial model
    full_cv_mse = np.zeros(k)
    for f in range(k):
        if time.time() > deadline:
            break
        val_mask = (fold_ids == f)
        tr_mask = ~val_mask
        if np.sum(tr_mask) < 6:
            full_cv_mse[f] = np.inf
            continue
        
        coeff_tr, _ = fit_full_polynomial(x_s[tr_mask], y[tr_mask], np.sum(tr_mask))
        if coeff_tr is not None:
            deg_tr = len(coeff_tr) - 1
            V_val = np.polynomial.polynomial.polyvander(x_s[val_mask], deg_tr)
            pred = V_val @ coeff_tr
            full_cv_mse[f] = np.mean((pred - y[val_mask]) ** 2)
        else:
            full_cv_mse[f] = np.inf

    full_cv_mean = np.mean(full_cv_mse)
    if not np.isfinite(full_cv_mean):
        full_cv_mean = np.inf

    # Choose better model
    use_odd = (odd_cv_mean <= full_cv_mean)

    # Refit on all data
    if use_odd:
        coeff_s, _ = fit_odd_polynomial(x_s, y, n)
        if coeff_s is None:
            coeff_s = np.zeros(3)
        
        # Convert from scaled odd basis to original x basis
        # y ≈ a1*x_s + a3*x_s^3 + a5*x_s^5
        # x_s = (x - x_center) / x_half_range
        # where x_center = 0.5*(x_min+x_max), x_half_range = 0.5*x_range
        x_center = 0.5 * (x_min + x_max)
        x_half_range = 0.5 * x_range + 1e-300
        
        # Expand: x_s = (x - x_center) / x_half_range
        # (x - x_center)^j / x_half_range^j
        # (x - x_center)^j = sum_{i=0}^{j} C(j,i) * x^i * (-x_center)^(j-i)
        
        coeff_orig = np.zeros(6)
        for j, aj in zip([1, 3, 5], coeff_s):
            if abs(aj) < 1e-30:
                continue
            scale = aj / (x_half_range ** j)
            for i in range(j + 1):
                binom = _binom(j, i)
                term = binom * ((-x_center) ** (j - i))
                coeff_orig[i] += scale * term
        
        if not np.all(np.isfinite(coeff_orig)):
            coeff_orig = np.zeros(6)
            coeff_orig[0] = y.mean()
        
        return coeff_orig
    else:
        coeff_s, _ = fit_full_polynomial(x_s, y, n)
        if coeff_s is None:
            coeff_s = np.zeros(6)
        
        # Convert from scaled full basis to original x basis
        deg = len(coeff_s) - 1
        x_center = 0.5 * (x_min + x_max)
        x_half_range = 0.5 * x_range + 1e-300
        
        coeff_orig = np.zeros(6)
        for j in range(deg + 1):
            if abs(coeff_s[j]) < 1e-30:
                continue
            scale = coeff_s[j] / (x_half_range ** j)
            for i in range(j + 1):
                binom = _binom(j, i)
                term = binom * ((-x_center) ** (j - i))
                coeff_orig[i] += scale * term
        
        if not np.all(np.isfinite(coeff_orig)):
            coeff_orig = np.zeros(6)
            coeff_orig[0] = y.mean()
        
        return coeff_orig


def _binom(n, k):
    """Compute binomial coefficient C(n, k) using integer arithmetic."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result *= (n - i)
        result //= (i + 1)
    return result
