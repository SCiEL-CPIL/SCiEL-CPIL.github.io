import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = time.time() + 115.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    x = x[:n]
    y = y[:n]

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n == 0:
        return np.zeros(6, dtype=np.float64)

    if n < 6:
        deg = max(0, n - 1)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            full_coeff = np.zeros(6, dtype=np.float64)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6, dtype=np.float64)

    # Scale x to [-1, 1] for better conditioning
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.array([y.mean(), 0, 0, 0, 0, 0], dtype=np.float64)

    x_center = (x_min + x_max) / 2.0
    x_half_range = x_range / 2.0
    x_scaled = (x - x_center) / x_half_range

    # --- Step 1: Denoise with GPR ---
    # Subsample for GPR if n is large to keep O(n^2) manageable
    n_max_gpr = 500
    if n > n_max_gpr:
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(n, n_max_gpr, replace=False)
        x_gpr = x_scaled[idx_sub]
        y_gpr = y[idx_sub]
    else:
        x_gpr = x_scaled
        y_gpr = y
        
    n_gpr = x_gpr.size
    
    # Hyperparameter search for GPR
    y_var = np.var(y)
    if y_var < 1e-12:
        y_var = 1.0

    l_candidates = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    sigma_n2_factors = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    
    best_loglik = -np.inf
    best_l = 0.2
    best_sigma_n2 = y_var * 0.1
    
    for l in l_candidates:
        if time.time() > deadline - 15:
            break
        for sn2_fac in sigma_n2_factors:
            sigma_n2 = y_var * sn2_fac
            
            # Build Kernel Matrix
            diff = x_gpr[:, None] - x_gpr[None, :]
            K = np.exp(-diff**2 / (2 * l**2))
            K += sigma_n2 * np.eye(n_gpr)
            
            try:
                # Cholesky decomposition for stability and log-det
                L = np.linalg.cholesky(K)
                # Solve L w = y
                w = np.linalg.solve(L, y_gpr)
                # Solve L^T v = w => v = K^{-1} y
                v = np.linalg.solve(L.T, w)
                
                # Log determinant is 2 * sum(log(diag(L)))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                
                # Log likelihood
                loglik = -0.5 * (np.dot(y_gpr, v) + log_det + n_gpr * np.log(2.0 * np.pi))
                
                if loglik > best_loglik:
                    best_loglik = loglik
                    best_l = l
                    best_sigma_n2 = sigma_n2
            except np.linalg.LinAlgError:
                continue

    # Use best hyperparameters to get denoised y
    l = best_l
    sigma_n2 = best_sigma_n2
    
    diff = x_gpr[:, None] - x_gpr[None, :]
    K = np.exp(-diff**2 / (2 * l**2))
    K += sigma_n2 * np.eye(n_gpr)
    
    try:
        w = np.linalg.solve(K, y_gpr)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        try:
            w, _, _, _ = np.linalg.lstsq(K, y_gpr, rcond=None)
        except Exception:
            w = None

    if w is not None:
        # Predict means for all x_scaled
        diff_star = x_scaled[:, None] - x_gpr[None, :]
        K_star = np.exp(-diff_star**2 / (2 * l**2))
        y_denoised = K_star @ w
    else:
        y_denoised = y.copy()

    # --- Step 2: Fit Polynomial to Denoised Data using Total Least Squares ---
    # Build Vandermonde matrix for monomial basis in scaled variable z
    # We want to fit y = sum(c_i * z^i)
    # TLS minimizes ||A * c - y||^2 + ||c||^2 * lambda (regularized)
    # Or more directly, SVD-based TLS
    
    # For polynomial fitting, standard LS is usually sufficient if data is clean.
    # However, TLS can be more robust if there is noise in x.
    # Given we denoised y with GPR, the residual noise is likely lower.
    # Let's try a regularized least squares fit on the monomial basis in scaled space.
    
    # Vandermonde matrix V where V[:, i] = x_scaled^i
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # Regularized least squares: (V^T V + lambda * I) c = V^T y
    # Choose lambda based on noise level
    # Estimate noise variance from GPR residuals? Or just use a small fixed value.
    # Since GPR denoises, we can use a smaller lambda.
    
    # Let's try a few lambda values and pick the one with best cross-validation?
    # Or just a reasonable default.
    # The previous score was 1.044. We want to improve it.
    # Maybe the Chebyshev basis was better conditioned?
    # Let's stick to Chebyshev basis for stability, then convert.
    
    # Build Chebyshev Vandermonde matrix (degree 0 to 5)
    T = np.zeros((n, 6), dtype=np.float64)
    T[:, 0] = 1.0
    T[:, 1] = x_scaled
    for k in range(2, 6):
        T[:, k] = 2.0 * x_scaled * T[:, k-1] - T[:, k-2]

    # Fit using Least Squares with small regularization
    TtT = T.T @ T
    Tty = T.T @ y_denoised
    
    # Adaptive regularization
    # Trace of TtT is roughly n * sum(diag(TtT)/n)
    # Let's use a relative regularization
    trace_TtT = np.trace(TtT)
    if trace_TtT < 1e-12:
        lam = 1e-8
    else:
        lam = 1e-6 * trace_TtT / 6.0
        
    A = TtT + lam * np.eye(6)
    
    try:
        cheb_coeff = np.linalg.solve(A, Tty)
    except np.linalg.LinAlgError:
        try:
            cheb_coeff, _, _, _ = np.linalg.lstsq(A, Tty, rcond=None)
        except Exception:
            return np.zeros(6, dtype=np.float64)

    if not np.all(np.isfinite(cheb_coeff)):
        return np.zeros(6, dtype=np.float64)

    # Convert Chebyshev coeffs to Monomial coeffs in scaled variable z
    # T_0 = 1
    # T_1 = z
    # T_2 = 2z^2 - 1
    # T_3 = 4z^3 - 3z
    # T_4 = 8z^4 - 8z^2 + 1
    # T_5 = 16z^5 - 20z^3 + 5z
    
    a0, a1, a2, a3, a4, a5 = cheb_coeff
    
    b0 = a0 - a2 + a4
    b1 = a1 - 3*a3 + 5*a5
    b2 = 2*a2 - 8*a4
    b3 = 4*a3 - 20*a5
    b4 = 8*a4
    b5 = 16*a5
    
    b = np.array([b0, b1, b2, b3, b4, b5], dtype=np.float64)
    
    # Convert from monomial coeffs in z to monomial coeffs in x
    # z = (x - c) / s
    # x = s * z + c
    # We have P(z) = sum(b_i * z^i)
    # We want Q(x) = P((x-c)/s) = sum(b_i * ((x-c)/s)^i)
    # Q(x) = sum(b_i * s^{-i} * (x-c)^i)
    # (x-c)^i = sum_{k=0}^i binom(i, k) x^k (-c)^{i-k}
    
    c = x_center
    s = x_half_range
    
    coeff_orig = np.zeros(6, dtype=np.float64)
    
    for i in range(6):
        bi = b[i]
        if abs(bi) < 1e-15:
            continue
        inv_s_pow = 1.0 / (s ** i)
        for k in range(i + 1):
            # binom(i, k) * x^k * (-c)^(i-k)
            # coeff of x^k in (x-c)^i is binom(i, k) * (-c)^(i-k)
            binom_coeff = 1.0
            for j in range(1, k+1):
                binom_coeff *= (i - j + 1) / j
            power_c = (-c) ** (i - k)
            coeff_orig[k] += bi * (binom_coeff * power_c) * inv_s_pow
            
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6, dtype=np.float64)

    return coeff_orig.astype(np.float64)
