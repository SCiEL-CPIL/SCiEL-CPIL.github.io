import numpy as np
import time
from math import comb

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

    # Scale x to [-1, 1] for GPR and to [0, 2pi] for Fourier
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.array([y.mean(), 0, 0, 0, 0, 0], dtype=np.float64)

    x_center = (x_min + x_max) / 2.0
    x_half_range = x_range / 2.0
    x_scaled = (x - x_center) / x_half_range # Range [-1, 1]
    
    # Map to [0, 2*pi] for Fourier
    x_fourier = np.pi * (x_scaled + 1.0) # Range [0, 2*pi]

    # --- Step 1: Denoise with GPR (Ensemble of top 3) ---
    n_max_gpr = 400
    if n > n_max_gpr:
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(n, n_max_gpr, replace=False)
        x_gpr = x_scaled[idx_sub]
        y_gpr = y[idx_sub]
    else:
        x_gpr = x_scaled
        y_gpr = y
        
    n_gpr = x_gpr.size
    
    y_var = np.var(y)
    if y_var < 1e-12:
        y_var = 1.0

    l_candidates = [0.1, 0.2, 0.3, 0.5, 0.8]
    sigma_n2_factors = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    # Store top 3 (loglik, l, sigma_n2)
    candidates = []
    
    for l in l_candidates:
        if time.time() > deadline - 10:
            break
        for sn2_fac in sigma_n2_factors:
            sigma_n2 = y_var * sn2_fac
            
            diff = x_gpr[:, None] - x_gpr[None, :]
            K = np.exp(-diff**2 / (2 * l**2))
            K += sigma_n2 * np.eye(n_gpr)
            
            try:
                L = np.linalg.cholesky(K)
                w = np.linalg.solve(L, y_gpr)
                v = np.linalg.solve(L.T, w)
                
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                loglik = -0.5 * (np.dot(y_gpr, v) + log_det + n_gpr * np.log(2.0 * np.pi))
                
                candidates.append((loglik, l, sigma_n2))
            except np.linalg.LinAlgError:
                continue

    if not candidates:
        # Fallback if all GPRs failed
        y_denoised = y
    else:
        # Sort by loglik descending
        candidates.sort(key=lambda item: item[0], reverse=True)
        
        # Take top 3
        top_k = min(3, len(candidates))
        ensemble_weights = np.zeros(n)
        
        # We can weight by softmax of logliks or just simple average
        # Simple average is robust
        for i in range(top_k):
            _, l, sigma_n2 = candidates[i]
            
            diff = x_gpr[:, None] - x_gpr[None, :]
            K = np.exp(-diff**2 / (2 * l**2))
            K += sigma_n2 * np.eye(n_gpr)
            
            try:
                w = np.linalg.solve(K, y_gpr)
                # Predict for all x_scaled
                diff_star = x_scaled[:, None] - x_gpr[None, :]
                K_star = np.exp(-diff_star**2 / (2 * l**2))
                y_pred = K_star @ w
                ensemble_weights += y_pred
            except np.linalg.LinAlgError:
                try:
                    w, _, _, _ = np.linalg.lstsq(K, y_gpr, rcond=None)
                    diff_star = x_scaled[:, None] - x_gpr[None, :]
                    K_star = np.exp(-diff_star**2 / (2 * l**2))
                    y_pred = K_star @ w
                    ensemble_weights += y_pred
                except Exception:
                    continue
        
        # Average the ensemble
        # Note: if some failed, we should divide by actual count
        # For simplicity, assume top 3 usually succeed. 
        # Let's track count
        count = 0
        ensemble_sum = np.zeros(n)
        for i in range(top_k):
            _, l, sigma_n2 = candidates[i]
            diff = x_gpr[:, None] - x_gpr[None, :]
            K = np.exp(-diff**2 / (2 * l**2))
            K += sigma_n2 * np.eye(n_gpr)
            try:
                w = np.linalg.solve(K, y_gpr)
                diff_star = x_scaled[:, None] - x_gpr[None, :]
                K_star = np.exp(-diff_star**2 / (2 * l**2))
                y_pred = K_star @ w
                ensemble_sum += y_pred
                count += 1
            except Exception:
                pass
        
        if count > 0:
            y_denoised = ensemble_sum / count
        else:
            y_denoised = y

    # --- Step 2: Fit Fourier Series to Denoised Data ---
    # Use x_fourier in [0, 2*pi]
    # Basis: 1, cos(x), sin(x), cos(2x), sin(2x), ... up to k=5
    # Total coeffs: 1 + 2*5 = 11
    fourier_order = 5
    num_fourier_coeffs = 1 + 2 * fourier_order
    
    F = np.zeros((n, num_fourier_coeffs))
    F[:, 0] = 1.0
    for k in range(1, fourier_order + 1):
        F[:, 2*k - 1] = np.cos(k * x_fourier)
        F[:, 2*k] = np.sin(k * x_fourier)
    
    # Least squares fit for Fourier coefficients
    FtF = F.T @ F
    Fty = F.T @ y_denoised
    
    # Regularize slightly for stability
    lam = 1e-8 * np.trace(FtF) / num_fourier_coeffs
    A = FtF + lam * np.eye(num_fourier_coeffs)
    
    try:
        fourier_coeff = np.linalg.solve(A, Fty)
    except np.linalg.LinAlgError:
        try:
            fourier_coeff, _, _, _ = np.linalg.lstsq(A, Fty, rcond=None)
        except Exception:
            # If Fourier fit fails, fall back to fitting poly directly to y_denoised
            y_for_poly = y_denoised
        else:
            # Generate smooth targets from Fourier fit
            y_for_poly = F @ fourier_coeff
            if not np.all(np.isfinite(y_for_poly)):
                y_for_poly = y_denoised

    # --- Step 3: Fit Degree-5 Monomial Polynomial to Smooth Targets ---
    # Use x_scaled in [-1, 1] for the polynomial fit to keep numbers well-scaled
    # Build Vandermonde matrix for monomial basis
    P = np.zeros((n, 6), dtype=np.float64)
    for i in range(6):
        P[:, i] = x_scaled ** i
        
    PtP = P.T @ P
    Pty = P.T @ y_for_poly
    
    # Regularization
    lam = 1e-8 * np.trace(PtP) / 6.0
    A_poly = PtP + lam * np.eye(6)
    
    try:
        poly_coeff_scaled = np.linalg.solve(A_poly, Pty)
    except np.linalg.LinAlgError:
        try:
            poly_coeff_scaled, _, _, _ = np.linalg.lstsq(A_poly, Pty, rcond=None)
        except Exception:
            return np.zeros(6, dtype=np.float64)

    if not np.all(np.isfinite(poly_coeff_scaled)):
        return np.zeros(6, dtype=np.float64)

    # Convert from scaled variable z = (x - c) / s to original x
    # z = (x - c) / s  =>  x = s*z + c
    # We have coeffs for z: sum b_i z^i
    # We want coeffs for x: sum a_j x^j
    # b_0 (s x / s + c/s)^0 ... wait.
    # z = (1/s) x - (c/s)
    # Let alpha = 1/s, beta = -c/s
    # z = alpha * x + beta
    # z^k = sum_{j=0}^k binom(k, j) (alpha x)^j beta^{k-j}
    # sum_{k=0}^5 b_k z^k = sum_{k=0}^5 b_k sum_{j=0}^k binom(k, j) alpha^j x^j beta^{k-j}
    # = sum_{j=0}^5 x^j [ sum_{k=j}^5 b_k binom(k, j) alpha^j beta^{k-j} ]
    
    alpha = 1.0 / x_half_range
    beta = -x_center / x_half_range
    
    coeff_orig = np.zeros(6, dtype=np.float64)
    
    for j in range(6):
        coeff_orig[j] = 0.0
        for k in range(j, 6):
            coeff_orig[j] += poly_coeff_scaled[k] * comb(k, j) * (alpha ** j) * (beta ** (k - j))
            
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6, dtype=np.float64)

    return coeff_orig.astype(np.float64)
