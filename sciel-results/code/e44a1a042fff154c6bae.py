<<<<<<< SEARCH
def local_refinement(h, start_time, max_time):
    """Local refinement using sensitivity-based peak reduction with FFT.
    
    At each step, we identify lags where the autoconvolution is near its max,
    compute how much each bin contributes to those peaks, and reduce bins
    that contribute more to peaks than to off-peak regions.
    Uses FFT for fast convolution and vectorized sensitivity computation.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    threshold_frac = 0.95
    
    # Precompute FFT size
    n_fft = 2 * N
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        conv = fft_convolve(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity using FFT:
        # conv[m] = sum_j f[j] * f[m-j]
        # d(conv[m])/d(f[j]) = f[m-j]
        # For peak lags M = {m1, m2, ...}, sensitivity[j] = sum_{m in M} f[m-j]
        # This is a cross-correlation of f with the indicator of peak lags
        peak_indicator = np.zeros(2*N-1)
        peak_indicator[peak_indices] = 1.0
        
        # sensitivity[j] = sum_m peak_indicator[m] * f[m-j]
        # This is the cross-correlation of f with peak_indicator
        # Using FFT: ifft(fft(f) * conj(fft(peak_indicator)))
        F = np.fft.rfft(current_h, n=n_fft)
        P = np.fft.rfft(peak_indicator, n=n_fft)
        # Cross-correlation: ifft(F * conj(P))
        corr = np.fft.irfft(F * np.conj(P), n=n_fft)
        # corr[k] = sum_j f[j] * peak_indicator[j+k] (circular)
        # We want sensitivity[j] = sum_m peak_indicator[m] * f[m-j]
        # = sum_m peak_indicator[m] * f[m-j]
        # Let k = m-j, so m = j+k: sum_k peak_indicator[j+k] * f[k]
        # This is the cross-correlation at lag j
        # corr[j] in the above gives sum_k f[k] * peak_indicator[j+k]
        # So sensitivity = corr (appropriately indexed)
        sensitivity = corr[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_convolve(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def spectral_polish(h, start_time, max_time):
    """High-precision polish pass using subgradient method with exact max.
    
    Uses a subgradient approach: at each step, compute the exact maximum
    of the autoconvolution, identify all lags achieving near-max, and
    adjust h to reduce those peaks.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    lr = 0.01
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_convolve(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Identify peak lags (within 1% of max)
        threshold = max_c * 0.99
        peak_mask = conv >= threshold
        
        # Subgradient: for the max of conv, the subgradient w.r.t. h[j] is
        # sum_{m: conv[m] in max} h[m-j]
        # We want to decrease h in directions that increase peak conv values
        peak_indicator = peak_mask.astype(np.float64)
        
        F = np.fft.rfft(current_h, n=2*N)
        P = np.fft.rfft(peak_indicator, n=2*N)
        corr = np.fft.irfft(F * np.conj(P), n=2*N)
        sensitivity = corr[:N]
        
        # Normalize and apply update
        sens_norm = sensitivity / (np.max(np.abs(sensitivity)) + 1e-12)
        h_new = current_h - lr * sens_norm
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_convolve(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            lr = min(lr * 1.02, 0.1)
        else:
            lr *= 0.9
            if lr < 1e-10:
                break
    
    return best_h, best_val
=======
def project_simplex(v):
    """Project vector v onto the probability simplex (sum=1, v>=0)."""
    n = len(v)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def smooth_pgd(h, start_time, max_time):
    """Smooth Projected Gradient Descent using log-sum-exp surrogate.
    
    Minimizes the smooth approximation of the normalized max autoconvolution.
    Uses projected gradient steps onto the simplex.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    current_h = project_simplex(current_h)
    
    beta = 20.0 # Start with moderate smoothness
    lr = 0.1
    n_fft = 2 * N
    
    # Precompute indices for convolution
    # conv[m] = sum_j h[j] h[m-j]
    # We need gradient of sum_m exp(beta conv[m]) w.r.t h[k]
    # grad[k] = sum_m exp(beta conv[m]) * d(conv[m])/dh[k]
    # d(conv[m])/dh[k] = h[m-k] (for linear convolution, with appropriate zero padding)
    
    while time.time() - start_time < max_time:
        conv = fft_convolve(current_h)
        
        # Compute log-sum-exp for stability
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            break
            
        shifted = conv - max_conv
        exp_terms = np.exp(beta * shifted)
        sum_exp = np.sum(exp_terms)
        
        # Objective value (for tracking)
        # log_sum_exp = max_conv + log(sum_exp)
        # val = (max_conv + log(sum_exp)) / beta / (sum(h)^2) ? 
        # We are minimizing max conv / (sum h)^2.
        # Since we project to sum=1, we minimize max conv.
        # Smooth objective: (1/beta) log(sum exp(beta conv))
        
        # Gradient of smooth objective
        # weights = exp_terms / sum_exp
        weights = exp_terms / sum_exp
        
        # grad[k] = sum_m weights[m] * d(conv[m])/dh[k]
        # d(conv[m])/dh[k] = h[m-k]
        # This is a weighted average of shifted h.
        # This is a convolution of weights with h?
        # Let's use FFT.
        # grad = ifft(fft(weights) * conj(fft(h))) ?
        # cross-correlation of weights and h.
        
        W = np.fft.rfft(weights, n=n_fft)
        H = np.fft.rfft(current_h, n=n_fft)
        
        # We want grad[k] = sum_m weights[m] h[m-k]
        # This is the cross-correlation of weights and h at lag k?
        # corr[k] = sum_m weights[m] h[m-k]
        # ifft(fft(weights) * conj(fft(h))) gives sum_j weights[j] h[j+k] ?
        # Let's verify:
        # ifft(F_W * conj(F_H))[k] = sum_j W[j] H[j+k] (circular)
        # We want sum_m weights[m] h[m-k].
        # Let m = j+k. sum_j weights[j+k] h[j].
        # This is ifft(conj(F_W) * F_H)?
        
        # Let's just compute it directly using the correlation property.
        # corr = ifft(F_H * conj(F_W))
        # corr[k] = sum_j H[j] W[j+k]
        # We want grad[k] = sum_m weights[m] h[m-k].
        # Let n = m-k => m = n+k.
        # grad[k] = sum_n weights[n+k] h[n].
        # This is corr[k] if corr is defined as sum_j H[j] W[j+k].
        
        corr = np.fft.irfft(H * np.conj(W), n=n_fft)
        grad = corr[:N]
        
        # Update
        h_new = current_h - lr * grad
        h_new = project_simplex(h_new)
        
        # Check improvement
        conv_new = fft_convolve(h_new)
        val_new = np.max(conv_new) # Since sum=1
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            
        # Adaptive learning rate
        # Simple heuristic: if val decreased, keep lr, else reduce
        # We need to compare val_new with previous val
        # We don't have prev_val easily stored, let's use a small buffer or just simple rule
        # For simplicity, if val_new < best_val, we are doing well.
        # Let's use a fixed decay or based on oscillation.
        
        # To be safe, reduce lr slowly
        lr = max(lr * 0.99, 0.01)
        
        # Increase beta slowly to sharpen
        beta = min(beta * 1.01, 100.0)
        
    return best_h, best_val
>>>>>>> REPLACE

