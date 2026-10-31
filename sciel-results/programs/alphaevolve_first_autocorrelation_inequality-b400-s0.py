"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a smooth parametric family based on cosine bumps, then refine
with a sensitivity-based local optimization that targets peak reduction.

The key insight is to use a family of functions whose autoconvolution has a
flatter peak. We use a smooth max (log-sum-exp) surrogate for the parametric
optimization, then apply a gradient-like local refinement that reduces the
sensitivity of peak lags to reduce the maximum autoconvolution value.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 130  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def cosine_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from cosine bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    f = cosine_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def local_refinement(h, start_time, max_time):
    """Local refinement using sensitivity-based peak reduction.
    
    At each step, we identify lags where the autoconvolution is near its max,
    compute how much each bin contributes to those peaks, and reduce bins
    that contribute more to peaks than to off-peak regions.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    threshold_frac = 0.95
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        conv = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity: for each bin j, how much does it contribute to peak lags?
        # conv[m] = sum_j current_h[j] * current_h[m-j]
        # d(conv[m])/d(current_h[j]) = current_h[m-j]
        sensitivity = np.zeros(N)
        for m in peak_indices:
            # m ranges from 0 to 2N-2
            # j ranges such that m-j is in [0, N-1]
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_h[m - j]
        
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
        
        conv_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components
    for n_c in [2, 3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [5.0, 10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            for restart in range(6):
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.05, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        if best_params is not None:
            f = parametric_to_step(best_params, n_c)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Local refinement: sensitivity-based peak reduction
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
