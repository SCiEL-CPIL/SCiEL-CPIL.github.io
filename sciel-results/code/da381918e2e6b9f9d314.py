import numpy as np
import time
from scipy.optimize import minimize

N = 600
# Safety margin: 130s out of 150s
DEADLINE = 130.0

# Precompute grid centers
# Interval width
dx = 0.5 / N
x_centers = -0.25 + (np.arange(N) + 0.5) * dx

def compute_raw_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    Returns 1.0 for uniform f=1.
    """
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    # Use FFT for speed if N is large, but np.convolve is fine for N=600
    conv = np.convolve(f, f, mode='full')
    max_conv = np.max(conv)
    # Normalization: Uniform f=1 -> sum=N, max_conv=N. Score = N * N / N^2 = 1.
    # Formula: max_conv * N / (sum(f)^2)
    return (max_conv * N) / (s * s)

def smooth_max(x, beta):
    """
    Smooth approximation of max using log-sum-exp.
    x: array
    beta: temperature parameter (higher = sharper)
    """
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    # Shift for numerical stability
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    # log-sum-exp
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def param_to_step(params, n_components):
    """
    Convert parametric representation to step function.
    params: [c1, w1, h1, c2, w2, h2, ...]
    """
    f = np.zeros(N)
    for i in range(n_components):
        c = params[i * 3]
        w = params[i * 3 + 1]
        h = params[i * 3 + 2]
        
        # Ensure width is positive and reasonable
        w = np.abs(w) + 0.001
        w = np.minimum(w, 0.5) # Max width is half the domain
        
        # Ensure height is non-negative
        h = np.abs(h)
        
        if h < 1e-10:
            continue
            
        # Cosine bump: h * cos^2(pi * (x-c)/w) for |x-c| < w/2
        dist = np.abs(x_centers - c)
        mask = dist < w / 2.0
        if np.any(mask):
            # cos^2 is (1 + cos(2*theta))/2, but direct cos is fine
            f[mask] += h * (np.cos(np.pi * (x_centers[mask] - c) / w) ** 2)
            
    return np.maximum(f, 0.0)

def parametric_objective(params, n_components, beta):
    """
    Objective function for parametric optimization.
    """
    f = param_to_step(params, n_components)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    
    conv = np.convolve(f, f, mode='full')
    
    # Use smooth max for differentiability/smooth landscape
    sm = smooth_max(conv, beta)
    
    # Normalized score
    score = (sm * N) / (s * s)
    
    return score

def local_refinement(f, start_time, max_time):
    """
    Local refinement on the 600-dim vector.
    Uses gradient-like descent on the peaks.
    """
    best_f = f.copy()
    best_score = compute_raw_score(best_f)
    current_f = f.copy()
    current_score = best_score
    alpha = 0.05
    
    while time.time() - start_time < max_time:
        s = np.sum(current_f)
        if s < 1e-12:
            break
        # Normalize to sum N for consistent scaling in refinement
        current_f = current_f * (N / s)
        
        conv = np.convolve(current_f, current_f, mode='full')
        max_c = np.max(conv)
        current_score = (max_c * N) / (N * N) # Since sum is N, s^2=N^2
        
        if current_score < best_score:
            best_score = current_score
            best_f = current_f.copy()
            
        # Find peak indices
        threshold = max_c * 0.95
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) == 0:
            break
            
        # Compute sensitivity: how much each f[i] contributes to the peaks
        # d(conv[k])/d(f[j]) = f[k-j] (for valid k-j)
        # We want to reduce f[j] if it contributes strongly to peaks.
        sensitivity = np.zeros(N)
        for k in peak_indices:
            # conv[k] = sum_{j} f[j] f[k-j]
            # Contribution of f[j] to conv[k] is f[k-j]
            # So sensitivity[j] += f[k-j]
            # We can vectorize this using convolution of sensitivity with f?
            # No, simpler:
            # For each peak k, add f shifted by k to sensitivity
            # f[k-j] is f reversed and shifted?
            # Let's just do the loop for peaks, it's small.
            k_min = max(0, k - (N - 1))
            k_max = min(N - 1, k)
            # f[k-j] for j in [k_min, k_max]
            # indices of f: k - j
            # j = k - i => i = k - j
            # We need f[i] where i = k - j.
            # j ranges k_min to k_max.
            # i ranges k - k_max to k - k_min.
            i_start = k - k_max
            i_end = k - k_min + 1
            sensitivity[k_min:k_max+1] += f[i_start:i_end]
            
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Update: reduce high sensitivity parts
        f_new = current_f * (1.0 - alpha * sens_norm)
        f_new = np.maximum(f_new, 0.0)
        
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            break
            
        f_new = f_new * (N / s_new)
        
        conv_new = np.convolve(f_new, f_new, mode='full')
        score_new = (np.max(conv_new) * N) / (N * N)
        
        if score_new < current_score:
            current_f = f_new
            current_score = score_new
        else:
            alpha *= 0.9
            if alpha < 1e-6:
                break
                
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Initial best: uniform
    best_f = np.ones(N)
    best_score = compute_raw_score(best_f)
    
    rng = np.random.RandomState(42)
    
    # Try different numbers of components
    for n_c in [2, 3, 4, 5]:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        # Annealed beta
        betas = [5.0, 10.0, 20.0, 40.0]
        
        for beta in betas:
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Number of restarts depends on time left
            time_left = DEADLINE - (time.time() - start_time)
            # Estimate time per restart: ~0.1s
            max_restarts = int(min(20, time_left / 0.2))
            
            for r in range(max_restarts):
                if time.time() - start_time > DEADLINE - 5:
                    break
                    
                # Initialize parameters
                # Centers in [-0.2, 0.2]
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                # Widths in [0.05, 0.2]
                widths = rng.uniform(0.05, 0.2, size=n_c)
                # Heights in [0.5, 2.0]
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        parametric_objective, 
                        x0, 
                        args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                    )
                    
                    if result.fun < best_score:
                        # Convert back to step
                        f_candidate = param_to_step(result.x, n_c)
                        # Check actual score (not smooth)
                        actual_score = compute_raw_score(f_candidate)
                        if actual_score < best_score:
                            best_score = actual_score
                            best_f = f_candidate
                            
                except Exception:
                    pass
                    
    # Local refinement
    if time.time() - start_time < DEADLINE - 5:
        refined_f, refined_score = local_refinement(
            best_f, start_time, DEADLINE - 2
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
            
    # Ensure non-negative
    best_f = np.maximum(best_f, 0.0)
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_raw_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
