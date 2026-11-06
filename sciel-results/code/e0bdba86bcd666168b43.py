import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Convert to numpy arrays
    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n = len(x)
    
    # Fallback for sparse data
    if n < 20:
        coeff = np.zeros(6)
        if n >= 2:
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except Exception:
                pass
        return coeff

    # 1. Nadaraya-Watson Smoothing
    # We estimate y as a smooth function of x using NW regression.
    # This helps denoise the subgrid tendency estimates.
    
    # Define bin centers for the x domain
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # Constant x
        coeff = np.zeros(6)
        coeff[0] = np.mean(y)
        return coeff

    # Number of bins for smoothing estimation
    # More bins -> more detailed curve, but noisier if bandwidth is small
    # Fewer bins -> smoother, but might miss features
    n_bins = 200
    
    x_centers = np.linspace(x_min, x_max, n_bins)
    
    # Try a few bandwidths
    # h is the standard deviation of the Gaussian kernel
    # Typical scale is range / 10 to range / 50
    bandwidths = [x_range / 50.0, x_range / 20.0, x_range / 10.0]
    
    best_poly = None
    best_score = np.inf
    
    # Check time before starting loop
    if time.time() > deadline - 10:
        # Fallback to direct fit if no time for smoothing
        try:
            s = (2.0 * x - (x_max + x_min)) / x_range
            leg_coeffs = np.polynomial.legende.legfit(s, y, 5)
            leg_poly = np.polynomial.legende.Legendre(leg_coeffs)
            mono_s_poly = leg_poly.convert(kind=np.polynomial.polynomial.Polynomial)
            a = 2.0 / x_range
            b = -(x_max + x_min) / x_range
            transform_poly = np.polynomial.polynomial.Polynomial([b, a])
            final_poly = mono_s_poly.compose(transform_poly)
            coeff = np.zeros(6)
            num_coeffs = min(6, len(final_poly.coef))
            coeff[:num_coeffs] = final_poly.coef[:num_coeffs]
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
        return np.zeros(6)

    for h in bandwidths:
        if time.time() > deadline - 5:
            break
            
        # Compute NW estimates at x_centers
        # y_hat[j] = sum_i exp(-(x_i - x_centers[j])^2 / (2*h^2)) * y_i / sum_i exp(...)
        
        # To avoid O(N * M) memory blowup if N is huge, we can chunk or just do it if N*M is manageable.
        # N=100k, M=200 -> 20M ops. Fast in numpy.
        
        # Vectorized calculation:
        # diff shape (M, N)
        # This might use too much memory if N is very large (e.g., 1M). 
        # If N > 100k, let's subsample x,y for the NW estimation? 
        # Or process in chunks.
        
        # Let's check N.
        if n > 200000:
            # Subsample for NW estimation to save memory/time
            idx = np.random.choice(n, 50000, replace=False)
            x_sub = x[idx]
            y_sub = y[idx]
        else:
            x_sub = x
            y_sub = y
        
        try:
            # Calculate distances: (M, 1) - (1, N) -> (M, N)
            # To save memory, we can do this in chunks of rows (bins)
            y_hat = np.zeros(n_bins)
            weights_sum = np.zeros(n_bins)
            
            # Chunk size for memory safety
            chunk_size = 50
            for i in range(0, n_bins, chunk_size):
                x_c_chunk = x_centers[i:i+chunk_size]
                # diff shape (chunk, N)
                diff = x_c_chunk[:, np.newaxis] - x_sub[np.newaxis, :]
                # Kernel weights
                k = np.exp(-0.5 * (diff / h) ** 2)
                
                # Sum weighted y
                # k @ y_sub -> (chunk,)
                num = k @ y_sub
                den = k.sum(axis=1)
                
                # Avoid division by zero
                mask_den = den > 1e-10
                y_hat[i:i+chunk_size][mask_den] = num[mask_den] / den[mask_den]
                # If den is 0, y_hat remains 0 (or we could interpolate)
                
            # Fit polynomial to (x_centers, y_hat)
            # Use Legendre fit for stability on the smoothed data
            s = (2.0 * x_centers - (x_max + x_min)) / x_range
            leg_coeffs = np.polynomial.legende.legfit(s, y_hat, 5)
            leg_poly = np.polynomial.legende.Legendre(leg_coeffs)
            mono_s_poly = leg_poly.convert(kind=np.polynomial.polynomial.Polynomial)
            
            a = 2.0 / x_range
            b = -(x_max + x_min) / x_range
            transform_poly = np.polynomial.polynomial.Polynomial([b, a])
            final_poly = mono_s_poly.compose(transform_poly)
            
            coeff = np.zeros(6)
            num_coeffs = min(6, len(final_poly.coef))
            coeff[:num_coeffs] = final_poly.coef[:num_coeffs]
            
            if not np.all(np.isfinite(coeff)):
                continue
                
            # Score: MSE on the original training data? 
            # The task says "minimize normalized error on independent validation trajectories".
            # We don't have validation data. 
            # Using training MSE on the *original* noisy data might favor overfitting to noise if we don't smooth enough.
            # Using training MSE on the *smoothed* data is circular.
            # A good proxy: Minimize the residual of the polynomial against the *smoothed* data? 
            # Or just pick the smallest bandwidth that produces a smooth curve?
            
            # Let's evaluate the fit quality against the smoothed data (which is our "target" for the smooth structure)
            # Actually, the goal is to approximate the *true* unresolved tendency.
            # The NW smoothed data is a better estimate of the true mean than the raw data.
            # So we want the polynomial to fit the NW smoothed data well.
            
            y_pred = np.polynomial.polynomial.polyval(x_centers, coeff)
            mse_smooth = np.mean((y_pred - y_hat) ** 2)
            
            # We want low MSE on the smoothed data.
            if mse_smooth < best_score:
                best_score = mse_smooth
                best_poly = coeff
                
        except Exception:
            continue

    if best_poly is not None and np.all(np.isfinite(best_poly)):
        return best_poly

    # Fallback: Direct Legendre fit on raw data
    try:
        s = (2.0 * x - (x_max + x_min)) / x_range
        leg_coeffs = np.polynomial.legende.legfit(s, y, 5)
        leg_poly = np.polynomial.legende.Legendre(leg_coeffs)
        mono_s_poly = leg_poly.convert(kind=np.polynomial.polynomial.Polynomial)
        a = 2.0 / x_range
        b = -(x_max + x_min) / x_range
        transform_poly = np.polynomial.polynomial.Polynomial([b, a])
        final_poly = mono_s_poly.compose(transform_poly)
        coeff = np.zeros(6)
        num_coeffs = min(6, len(final_poly.coef))
        coeff[:num_coeffs] = final_poly.coef[:num_coeffs]
        if np.all(np.isfinite(coeff)):
            return coeff
    except Exception:
        pass

    return np.zeros(6)
