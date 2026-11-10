"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Handle constant x
    if np.std(x) < 1e-12:
        return np.array([np.mean(y)] + [0.0]*5)
    
    # 1. Estimate conditional variance sigma^2(x) using local binning
    # This allows us to perform heteroscedastic weighted least squares.
    # We assume the noise variance depends on the slow state x.
    n_bins = min(20, max(5, n // 20)) # Adaptive binning
    if n_bins > n:
        n_bins = max(1, n)
        
    # Create bins
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min < 1e-12:
        x_max = x_min + 1e-12
        
    # Define bin edges
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    
    # Assign each sample to a bin
    # np.digitize returns indices 1..n_bins
    bin_indices = np.digitize(x, bin_edges) - 1
    # Clip to valid range
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate local mean and variance for each bin
    local_means = np.zeros(n_bins)
    local_vars = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for b in range(n_bins):
        mask = (bin_indices == b)
        count = np.sum(mask)
        bin_counts[b] = count
        if count > 0:
            y_bin = y[mask]
            local_means[b] = np.mean(y_bin)
            if count > 1:
                local_vars[b] = np.var(y_bin)
            else:
                local_vars[b] = 0.0 # Will be handled by smoothing
    
    # Smooth the variance estimates to avoid zero variance in small bins
    # and to capture the general trend of noise vs x.
    # Use a simple moving average or just fill zeros with global variance.
    global_var = np.var(y)
    if global_var < 1e-12:
        global_var = 1e-12
        
    # Fill zeros and smooth
    smoothed_vars = local_vars.copy()
    for b in range(n_bins):
        if smoothed_vars[b] < 1e-12:
            smoothed_vars[b] = global_var
            
    # Apply a light smoothing filter to the variance profile
    # This helps if some bins are under-sampled
    if n_bins > 3:
        kernel = np.array([1, 2, 1]) / 4.0
        padded_vars = np.pad(smoothed_vars, (1, 1), mode='edge')
        smoothed_vars = np.convolve(padded_vars, kernel, mode='valid')
        
    # Ensure variance is positive
    smoothed_vars = np.maximum(smoothed_vars, 1e-12)
    
    # 2. Assign weights to each sample based on its local variance estimate
    # w_i = 1 / sigma^2(x_i)
    # We use the variance of the bin the sample belongs to.
    sample_vars = smoothed_vars[bin_indices]
    weights = 1.0 / sample_vars
    
    # Normalize weights to sum to n for numerical stability in some solvers,
    # though WLS handles it. Let's keep them raw but scaled to avoid overflow/underflow issues.
    max_w = np.max(weights)
    if max_w > 0:
        weights = weights / max_w
    else:
        weights = np.ones(n)
        
    # 3. Perform Weighted Least Squares (WLS) polynomial fit
    # We fit a degree 5 polynomial.
    # To improve numerical stability, we center and scale x.
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_norm = np.zeros(n)
    else:
        x_norm = (x - x_mean) / x_std
        
    # Build design matrix
    X = np.vander(x_norm, N=6, increasing=True)
    
    # WLS: min || W * (X @ c - y) ||^2
    # Equivalent to min || sqrt(W) * (X @ c - y) ||^2
    # Let X_w = X * sqrt(weights)[:, None]
    # Let y_w = y * sqrt(weights)
    
    sqrt_w = np.sqrt(weights)
    X_w = X * sqrt_w[:, None]
    y_w = y * sqrt_w
    
    # Solve the normal equations or use lstsq
    try:
        # Use lstsq for robustness
        coeff_norm, residuals, rank, s = np.linalg.lstsq(X_w, y_w, rcond=None)
        
        # Check for rank deficiency
        if rank < 6:
             # Fallback to regular least squares if WLS is ill-conditioned
             coeff_norm, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
             
    except np.linalg.LinAlgError:
        # Fallback to simple mean if all else fails
        coeff_norm = np.zeros(6)
        coeff_norm[0] = np.mean(y)
        
    # 4. Transform coefficients from normalized basis to original basis
    # y = sum_{k=0}^5 c_k * ((x - x_mean)/x_std)^k
    # We want: y = sum_{i=0}^5 a_i * x^i
    # c_k * z^k where z = (x - x_mean)/x_std
    # z^k = (1/x_std^k) * (x - x_mean)^k
    # (x - x_mean)^k = sum_{j=0}^k binom(k, j) x^j (-x_mean)^(k-j)
    
    a = np.zeros(6)
    for k in range(6):
        c_k = coeff_norm[k]
        if c_k == 0:
            continue
        factor = c_k / (x_std ** k)
        for i in range(k + 1):
            # Calculate binom(k, i)
            binom_val = 1.0
            for j in range(i):
                binom_val = binom_val * (k - j) / (j + 1)
                
            term = factor * binom_val * ((-x_mean) ** (k - i))
            a[i] += term
            
    # Ensure finite
    if not np.all(np.isfinite(a)):
        # Fallback to unweighted fit if numerical issues arise
        try:
            coeff_lin, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            a = np.zeros(6)
            for k in range(6):
                c_k = coeff_lin[k]
                if c_k == 0:
                    continue
                factor = c_k / (x_std ** k)
                for i in range(k + 1):
                    binom_val = 1.0
                    for j in range(i):
                        binom_val = binom_val * (k - j) / (j + 1)
                    a[i] += factor * binom_val * ((-x_mean) ** (k - i))
        except:
            a = np.zeros(6)
            a[0] = np.mean(y)
            
    return a

# EVOLVE-BLOCK-END
