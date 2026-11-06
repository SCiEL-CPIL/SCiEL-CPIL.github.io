# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # --- Step 1: Nadaraya-Watson Smoothing on y given x ---
    # This denoises the target variable to capture the underlying smooth dynamics.
    # We choose a bandwidth h based on the data scale.
    
    # Robust scale estimate
    median_x = np.median(x)
    mad = np.median(np.abs(x - median_x))
    if mad < 1e-10:
        scale_x = np.std(x)
        if scale_x < 1e-10:
            scale_x = 1.0
    else:
        # Convert MAD to std equivalent (assuming normality approx)
        scale_x = mad / 0.6745

    # Grid of bandwidths to try (relative to scale_x)
    # We want to smooth enough to remove noise but not lose signal.
    # Typical bandwidths for NW smoothing are in the range of 0.1 to 1.0 * scale
    h_rels = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
    
    # We need to select the best bandwidth. 
    # Since we don't have a separate validation set explicitly provided for this step 
    # (the validation is done externally on unseen trajectories), we can use 
    # a simple heuristic or a small internal cross-validation.
    # However, cross-validation for NW is O(N^2) per bandwidth, which might be slow for large N.
    # Let's estimate N. If N is large (e.g., > 10k), we subsample for CV.
    
    # For efficiency, let's just pick a reasonable bandwidth first, or do a fast CV.
    # Given the time limit, let's try a small CV on a subsample if N is large.
    
    max_cv_samples = 2000
    if n > max_cv_samples:
        idx_cv = np.random.choice(n, max_cv_samples, replace=False)
        x_cv = x[idx_cv]
        y_cv = y[idx_cv]
        n_cv = max_cv_samples
    else:
        x_cv = x
        y_cv = y
        n_cv = n

    best_h = scale_x * 0.5
    best_cv_mse = np.inf

    # Evaluate a few bandwidths on the CV set
    # Sort x_cv for faster evaluation? NW is O(N^2) if done naively.
    # With N=2000, N^2 = 4M operations per bandwidth. 9 bandwidths -> 36M ops. Fast enough.
    
    for h_rel in h_rels:
        h = h_rel * scale_x
        # Compute NW smoothed y
        # y_smooth[i] = sum_j K((x_i - x_j)/h) * y_j / sum_j K((x_i - x_j)/h)
        # Use Gaussian kernel K(u) = exp(-u^2 / 2)
        
        # Vectorized computation for efficiency
        # diff = x_cv[:, None] - x_cv[None, :]
        # This creates an NxN matrix. For N=2000, that's 4M floats = 32MB. Acceptable.
        
        diff = x_cv[:, None] - x_cv[None, :]
        u = diff / h
        K = np.exp(-0.5 * u * u)
        
        # Avoid division by zero if all weights are 0 (unlikely with Gaussian)
        weights_sum = np.sum(K, axis=1)
        weights_sum[weights_sum == 0] = 1e-10
        
        y_smooth = np.sum(K * y_cv[None, :], axis=1) / weights_sum
        
        # Calculate MSE
        mse = np.mean((y_smooth - y_cv) ** 2)
        
        if mse < best_cv_mse:
            best_cv_mse = mse
            best_h = h

    # Apply the best bandwidth to the FULL dataset to get smoothed y
    # If n is very large, we might need to chunk this to avoid memory issues.
    # But typically training data for such tasks is manageable (e.g., < 100k).
    # Let's chunk if n > 50000 to be safe.
    
    y_smoothed = np.zeros_like(y)
    chunk_size = 10000
    
    for i in range(0, n, chunk_size):
        j = min(i + chunk_size, n)
        x_chunk = x[i:j]
        # diff against ALL x
        diff = x_chunk[:, None] - x[None, :]
        u = diff / best_h
        K = np.exp(-0.5 * u * u)
        
        weights_sum = np.sum(K, axis=1)
        weights_sum[weights_sum == 0] = 1e-10
        
        y_smoothed[i:j] = np.sum(K * y[None, :], axis=1) / weights_sum

    # Check time
    if time.time() > deadline - 10:
        # Fallback: return simple mean or zero if we ran out of time
        # Better to return a simple LS fit on raw data if time is tight
        try:
            x_mean = np.mean(x)
            x_std = np.std(x)
            if x_std < 1e-10: x_std = 1.0
            x_norm = (x - x_mean) / x_std
            V = np.vander(x_norm, N=6, increasing=True)
            coeff = np.linalg.lstsq(V, y, rcond=None)[0]
            # Transform back
            coeff_orig = _transform_coeffs(coeff, x_mean, x_std)
            return coeff_orig
        except:
            return np.zeros(6)

    # --- Step 2: Fit Degree-5 Polynomial to Smoothed y ---
    # Now we fit a global polynomial to the denoised signal.
    # This should be more stable and less overfitting-prone.
    
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    V = np.vander(x_norm, N=6, increasing=True)

    # We can use standard LS or Ridge. 
    # Since y is smoothed, LS might be sufficient, but Ridge adds robustness.
    # Let's use a small fixed lambda or a quick selection.
    # Given the smoothing already removed high-freq noise, a small lambda is likely best.
    
    # Let's do a quick lambda selection on a holdout set to be safe.
    n_holdout = max(10, min(1000, n // 10))
    indices = np.random.permutation(n)
    holdout_idx = indices[:n_holdout]
    train_idx = indices[n_holdout:]
    
    V_train = V[train_idx]
    y_train = y_smoothed[train_idx]
    V_hold = V[holdout_idx]
    y_hold = y_smoothed[holdout_idx]
    
    VtV = V_train.T @ V_train
    Vty = V_train.T @ y_train
    
    # Small range of lambdas since data is smoothed
    max_lambda = np.trace(VtV) / 6.0
    if max_lambda == 0:
        max_lambda = 1.0
    
    lambdas = np.logspace(-5, 2, 50) * max_lambda
    
    best_mse = np.inf
    best_lambda = 0.0
    
    for lam in lambdas:
        A = VtV + lam * np.eye(6)
        try:
            coeff_norm = np.linalg.solve(A, Vty)
        except np.linalg.LinAlgError:
            continue
        
        pred_hold = V_hold @ coeff_norm
        mse = np.mean((pred_hold - y_hold) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_lambda = lam
            
        if time.time() > deadline - 5:
            break

    # Refit on full data with best lambda
    VtV_full = V.T @ V
    Vty_full = V.T @ y_smoothed
    A_full = VtV_full + best_lambda * np.eye(6)
    
    try:
        coeff_norm_full = np.linalg.solve(A_full, Vty_full)
        if not np.all(np.isfinite(coeff_norm_full)):
            coeff_norm_full = np.linalg.lstsq(V, y_smoothed, rcond=None)[0]
    except:
        coeff_norm_full = np.linalg.lstsq(V, y_smoothed, rcond=None)[0]

    # Transform coefficients back to original x basis
    coeff_orig = _transform_coeffs(coeff_norm_full, x_mean, x_std)

    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.zeros(6)

    return coeff_orig


def _transform_coeffs(coeff_norm, x_mean, x_std):
    """
    Transform coefficients from normalized basis z = (x - mean)/std
    to original basis x.
    P(x) = sum c_i z^i = sum c_i (a*x + b)^i
    where a = 1/std, b = -mean/std.
    (a*x + b)^k = sum_{j=0}^k binom(k, j) a^j b^{k-j} x^j
    So coeff_orig[j] = sum_{k=j}^5 c_k * binom(k, j) * a^j * b^{k-j}
    """
    coeff_orig = np.zeros(6)
    a = 1.0 / x_std
    b = -x_mean / x_std
    
    for k in range(6):
        c_k = coeff_norm[k]
        if c_k == 0:
            continue
        for j in range(k + 1):
            binom_k_j = float(math.comb(k, j))
            coeff_orig[j] += c_k * binom_k_j * (a ** j) * (b ** (k - j))
            
    return coeff_orig
# EVOLVE-BLOCK-END
