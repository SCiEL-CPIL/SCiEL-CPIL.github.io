import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 130  # safety margin before 150s limit

def _compute_score(h):
    """Compute the normalized maximum of the autoconvolution.
    For uniform h=1, sum=N, max_conv=N, score = N*N/N^2 = 1.
    """
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    c = np.convolve(h, h, mode='full')
    return np.max(c) * N / (s * s)

def _smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def _parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N using cosine bumps.
    params: [centers..., widths..., heights...]
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N  # interval centers
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    # Ensure widths are positive and reasonable
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    # Ensure heights are nonnegative
    heights = np.abs(heights)
    
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return np.maximum(result, 0.0)

def _parametric_objective(params, n_components, beta=10.0):
    """Objective for parametric optimization with smooth max."""
    f = _parametric_to_step(params, n_components)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    return _smooth_max(conv, beta) * N / (s * s)

def _local_refinement(h, start_time, max_time):
    """Local refinement using gradient-based peak reduction.
    Returns improved h and its score.
    """
    best_h = h.copy()
    best_val = _compute_score(h)
    current_h = h.copy()
    current_val = best_val
    alpha = 0.1
    
    while time.time() - start_time < max_time:
        # Normalize h to have sum = N
        s = np.sum(current_h)
        if s < 1e-12:
            current_h = np.ones(N)
            s = N
        current_h = current_h * (N / s)
        
        # Compute autoconvolution
        c = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(c)
        current_val = max_c * N / (s * s)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where c is close to max
        threshold = max_c * 0.95
        peak_indices = np.where(c >= threshold)[0]
        
        # Compute sensitivity
        sensitivity = np.zeros(N)
        for m in peak_indices:
            k_min = max(0, m - (N-1))
            k_max = min(N-1, m)
            for j in range(k_min, k_max+1):
                sensitivity[j] += current_h[m-j]
        
        # Adjust h
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        c_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(c_new) * N / (s_new * s_new)
        
        if val_new < current_val:
            current_h = h_new
        else:
            alpha *= 0.9
            if alpha < 1e-6:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = _compute_score(best_f)
    
    # Try different numbers of cosine bump components with annealed beta
    for n_c in [2, 3, 4]:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start low, increase
        betas = [5.0, 10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > DEADLINE - 15:
                break
            
            for restart in range(6):
                if time.time() - start_time > DEADLINE - 10:
                    break
                
                # Initialize parameters
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.05, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        _parametric_objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        if best_params is not None:
            f = _parametric_to_step(best_params, n_c)
            actual_score = _compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Local refinement on the best found
    if time.time() - start_time < DEADLINE - 10:
        refined_f, refined_score = _local_refinement(
            best_f, start_time, DEADLINE - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {_compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
