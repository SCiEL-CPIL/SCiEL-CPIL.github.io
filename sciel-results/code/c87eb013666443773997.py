import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    x = np.asarray(training_x, dtype=float).flatten()
    t = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # If n is very large, subsample for kernel regression to ensure speed
    max_samples_for_smoothing = 5000
    if n > max_samples_for_smoothing:
        idx = np.random.choice(n, max_samples_for_smoothing, replace=False)
        x_smooth = x[idx]
        t_smooth = t[idx]
    else:
        x_smooth = x
        t_smooth = t
        
    # Shuffle to ensure random distribution for CV folds
    # (np.random.choice already shuffles, but if n <= max, we use original order which might be time-series)
    # Time-series data has autocorrelation. Random subsampling breaks this, but if we don't subsample, we should shuffle.
    if n <= max_samples_for_smoothing:
        shuffle_idx = np.random.permutation(n)
        x_smooth = x[shuffle_idx]
        t_smooth = t[shuffle_idx]

    n_smooth = len(x_smooth)
    
    # Estimate bandwidth for Nadaraya-Watson
    std_x = np.std(x_smooth)
    if std_x < 1e-10:
        return np.zeros(6)
    
    h_base = 1.06 * std_x * (n_smooth ** (-0.2))
    
    # Try a small grid of bandwidths around Silverman's estimate
    bandwidth_factors = [0.5, 0.7, 1.0, 1.3, 1.5]
    
    def _compute_denoised(h_val):
        """Compute Nadaraya-Watson denoised targets for given bandwidth."""
        t_den = np.zeros_like(t_smooth)
        chunk_size = 1000
        for i in range(0, n_smooth, chunk_size):
            end = min(i + chunk_size, n_smooth)
            x_chunk = x_smooth[i:end]
            diff = x_chunk[:, np.newaxis] - x_smooth[np.newaxis, :]
            weights = np.exp(-0.5 * (diff / h_val) ** 2)
            w_sum = np.sum(weights, axis=1)
            wt_sum = np.sum(weights * t_smooth[np.newaxis, :], axis=1)
            t_den[i:end] = wt_sum / np.where(w_sum > 1e-10, w_sum, 1.0)
        return t_den
    
    def _fit_polynomial_huber(x_data, t_data):
        """Fit degree-5 polynomial with Huber-loss reweighting and return coefficients in original x space."""
        mu = np.mean(x_data)
        sigma = np.std(x_data)
        if sigma < 1e-10:
            sigma = 1.0
        x_scaled = (x_data - mu) / sigma
        
        # Build Vandermonde matrix for degree-5 polynomial in scaled space
        V = np.vander(x_scaled, 6, increasing=True)
        
        # Initial least-squares fit
        try:
            coeff_scaled, _, _, _ = np.linalg.lstsq(V, t_data, rcond=None)
        except Exception:
            try:
                coeff_scaled, _, _, _ = np.linalg.lstsq(V[:, :4], t_data, rcond=None)
                temp = np.zeros(6)
                temp[:4] = coeff_scaled
                coeff_scaled = temp
            except:
                return None
        
        # Iterative reweighting with Huber loss
        for _ in range(5):
            if time.time() > deadline - 3.0:
                break
            # Compute residuals
            pred = V @ coeff_scaled
            resid = t_data - pred
            
            # Huber weights: w = 1 if |r| <= delta, else delta/|r|
            # Use MAD-based scale estimate for robustness
            abs_resid = np.abs(resid)
            mad = np.median(abs_resid)
            if mad < 1e-10:
                mad = np.std(resid)
            if mad < 1e-10:
                mad = 1e-10
            # delta = 1.4826 * MAD (consistent estimator of std for Gaussian)
            delta = 1.4826 * mad
            
            weights = np.where(abs_resid <= delta, 1.0, delta / abs_resid)
            # Clip weights to avoid extreme values
            weights = np.clip(weights, 0.01, 1.0)
            
            # Weighted least squares
            W_sqrt = np.sqrt(weights)
            V_w = V * W_sqrt[:, np.newaxis]
            t_w = t_data * W_sqrt
            try:
                coeff_scaled, _, _, _ = np.linalg.lstsq(V_w, t_w, rcond=None)
            except Exception:
                break
        
        # Transform from scaled x to original x
        c_u = coeff_scaled / (sigma ** np.arange(6))
        coeff_original = np.zeros(6)
        for i in range(6):
            for j in range(i+1):
                if j == 0 or j == i:
                    binom_val = 1.0
                else:
                    num = 1
                    for k in range(j):
                        num *= (i - k)
                    den = 1
                    for k in range(j):
                        den *= (k + 1)
                    binom_val = num / den
                coeff_original[j] += c_u[i] * binom_val * ((-mu) ** (i - j))
        return coeff_original

    best_coeff = None
    best_cv_score = np.inf
    best_h = h_base
    
    # 3-fold Cross-Validation for bandwidth selection
    # This is more robust than a single 80/20 split
    n_smooth = len(x_smooth)
    fold_size = n_smooth // 3
    indices = np.arange(n_smooth)
    
    for factor in bandwidth_factors:
        if time.time() > deadline - 10.0:
            break
            
        h_val = h_base * factor
        t_denoised = _compute_denoised(h_val)
        
        cv_errors = []
        for fold in range(3):
            if time.time() > deadline - 5.0:
                break
                
            # Define train and val indices
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < 2 else n_smooth
            
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            x_train = x_smooth[train_idx]
            t_train = t_denoised[train_idx]
            x_val = x_smooth[val_idx]
            t_val = t_denoised[val_idx]
            
            # Fit polynomial on train subset with Huber reweighting
            coeff_fit = _fit_polynomial_huber(x_train, t_train)
            if coeff_fit is None:
                continue
            
            # Evaluate MSE on validation set
            pred_val = np.zeros_like(x_val)
            for i in range(6):
                pred_val += coeff_fit[i] * x_val ** i
            mse_val = np.mean((pred_val - t_val) ** 2)
            cv_errors.append(mse_val)
            
        if cv_errors:
            avg_mse = np.mean(cv_errors)
            if avg_mse < best_cv_score:
                best_cv_score = avg_mse
                best_h = h_val
                
    # Refit on all data with the best bandwidth
    if time.time() < deadline - 5.0:
        t_denoised_final = _compute_denoised(best_h)
        best_coeff = _fit_polynomial_huber(x_smooth, t_denoised_final)
    else:
        # Fallback if time is tight: use base bandwidth and simple fit
        t_denoised_final = _compute_denoised(h_base)
        best_coeff = _fit_polynomial_huber(x_smooth, t_denoised_final)
        
    if best_coeff is None:
        # Final fallback
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, t, 1)
        except:
            pass
        return coeff
        
    return best_coeff
