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

def fast_peak_shaving(h, start_time, max_time):
    """Fast local refinement using FFT-based convolution and gradient-like steps.
    
    Targets the maximum of the autoconvolution by reducing bins that contribute
    most to the peak, while maintaining non-negativity and constant sum.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    patience = 0
    max_patience = 500
    step_count = 0
    
    # Precompute FFT size for efficiency
    fft_size = 2 * N
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        # Compute autoconvolution using FFT
        f_fft = np.fft.rfft(current_h, n=fft_size)
        conv_fft = f_fft * f_fft
        conv = np.fft.irfft(conv_fft, n=fft_size)
        conv = conv[:2*N-1] # Trim to full convolution length
        
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        # Find the index of the maximum
        k_star = np.argmax(conv)
        
        # Compute gradient: d(conv[k_star])/d(f[j]) = f[k_star - j]
        # We need to be careful with indices.
        # conv[m] = sum_{j=0}^{N-1} f[j] f[m-j]
        # If we fix m = k_star, the term involving f[j] is f[j] * f[k_star - j].
        # The derivative with respect to f[j] is f[k_star - j].
        # Note: f is defined on [0, N-1]. f[k_star - j] is 0 if k_star - j is out of bounds.
        
        # We want to construct a vector g such that reducing f in directions of high g reduces the peak.
        # g[j] should be proportional to f[k_star - j].
        
        # Create an array to hold f[k_star - j] for j in 0..N-1
        g = np.zeros(N)
        # k_star - j ranges from k_star - (N-1) to k_star
        # Valid indices for f are 0..N-1
        # So we need k_star - j to be in [0, N-1]
        # j <= k_star and j >= k_star - (N-1)
        
        j_min = max(0, k_star - (N - 1))
        j_max = min(N - 1, k_star)
        
        if j_min <= j_max:
            indices = np.arange(j_min, j_max + 1)
            # f[k_star - j] for j in indices
            # k_star - indices gives the indices into f
            f_indices = k_star - indices
            g[indices] = current_h[f_indices]
            
            # To avoid reducing the sum, we center g
            g_mean = np.mean(g)
            g = g - g_mean
            
            # Update
            h_new = current_h - alpha * g
            h_new = np.maximum(h_new, 0.0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                break
            h_new = h_new * (N / s_new)
            
            # Check improvement
            # We can estimate improvement or just compute it
            # Computing score is O(N log N), so it's fine
            val_new = compute_score(h_new)
            
            if val_new < current_val:
                current_h = h_new
                alpha = min(alpha * 1.05, 0.5)
                step_count += 1
            else:
                alpha *= 0.9
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
    
    # Local refinement: Fast peak shaving
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = fast_peak_shaving(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
            
    # Also try starting from a simple cosine bump if it's better or if parametric failed
    # Generate a cosine bump initial guess
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    f_cosine = np.cos(np.pi * x) ** 2
    f_cosine = np.maximum(f_cosine, 0)
    cos_score = compute_score(f_cosine)
    
    if cos_score < best_score and time.time() - start_time < TIME_LIMIT - 20:
        best_f = f_cosine
        best_score = cos_score
        # Refine the cosine bump too
        refined_cos, refined_cos_score = fast_peak_shaving(
            f_cosine, start_time, TIME_LIMIT - 5
        )
        if refined_cos_score < best_score:
            best_f = refined_cos
            best_score = refined_cos_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
