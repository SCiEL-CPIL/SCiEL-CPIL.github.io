import numpy as np
from scipy.optimize import minimize
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 140

def _autoconv_fft(h):
    """Compute linear auto-convolution of h using FFT."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    S = H * np.conj(H)
    conv = np.fft.irfft(S, n=n_fft)
    return conv[:2*N-1]

def _soft_max(x, beta=50.0):
    """Soft maximum approximation."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _evaluate_exact(h):
    """Compute the exact value as the evaluator would."""
    if not np.all(np.isfinite(h)) or abs(np.sum(h)) < 1e-10:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (np.sum(h) ** 2))

def _random_walk_to_h(steps, drift):
    """
    Convert a sequence of steps to heights via random walk with drift.
    h[i] = cumulative sum of (steps - drift), then normalize to zero mean.
    This naturally produces diverse signal structures.
    """
    walk = np.cumsum(steps - drift)
    # Subtract mean to ensure zero-sum
    h = walk - np.mean(walk)
    return h

def _coordinate_descent(h, max_iter=50, beta=50.0):
    """
    Coordinate descent on individual heights, maintaining zero-sum constraint.
    For each coordinate, try small perturbations and accept if value decreases.
    """
    current_val = _evaluate_exact(h)
    if current_val == float('inf'):
        return h, current_val
    
    h = h.copy()
    step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        improved = False
        for i in range(N):
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
            for ss in step_sizes:
                # Try increasing h[i] and decreasing another coordinate to maintain sum
                # Use a simple approach: perturb h[i] and h[(i+1)%N]
                j = (i + 1) % N
                h_trial = h.copy()
                h_trial[i] += ss
                h_trial[j] -= ss  # Maintain sum approximately
                
                trial_val = _evaluate_exact(h_trial)
                if trial_val < current_val:
                    h = h_trial
                    current_val = trial_val
                    improved = True
                    break
        
        if not improved:
            break
    
    return h, current_val

def _sparse_to_h(spikes, spike_vals):
    """
    Create a height vector from sparse spikes.
    spikes: indices where spikes occur
    spike_vals: values at those spikes
    Fill rest with a baseline to maintain zero mean.
    """
    h = np.zeros(N)
    for idx, val in zip(spikes, spike_vals):
        h[idx] = val
    
    # Compute baseline to make sum zero
    sum_spikes = np.sum(spike_vals)
    baseline = -sum_spikes / (N - len(spikes)) if len(spikes) < N else 0.0
    mask = np.ones(N, dtype=bool)
    for idx in spikes:
        mask[idx] = False
    h[mask] = baseline
    
    return h

def _objective_sparse(spike_params, n_spikes):
    """
    Objective for sparse spike parameterization.
    spike_params: [spike_positions (n_spikes), spike_values (n_spikes), baseline_scale]
    """
    if len(spike_params) < 2 * n_spikes + 1:
        return 1e10
    
    spike_pos = spike_params[:n_spikes].astype(int)
    spike_vals = spike_params[n_spikes:2*n_spikes]
    baseline_scale = spike_params[2*n_spikes]
    
    # Ensure valid spike positions
    spike_pos = np.clip(spike_pos, 0, N-1)
    
    h = _sparse_to_h(spike_pos, spike_vals)
    h *= baseline_scale
    
    if not np.all(np.isfinite(h)) or abs(np.sum(h)) < 1e-10:
        return 1e10
    
    conv = _autoconv_fft(h)
    max_conv = _soft_max(conv, beta=50.0)
    
    return 2.0 * N * max_conv / (np.sum(h) ** 2)

def construct():
    global START_TIME, TIME_LIMIT
    
    # Fallback: Uniform
    best_h = np.ones(N) / N
    best_val = 2.0
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Random walk with various drifts
    for trial in range(5):
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
        
        # Generate random steps
        steps = rng.normal(0, 1, N)
        # Try different drift values
        for drift in [0.0, 0.1, -0.1, 0.2, -0.2]:
            if time.time() - START_TIME > TIME_LIMIT - 30:
                break
            
            h = _random_walk_to_h(steps, drift)
            
            # Normalize to have sum = 1 for consistent scaling
            h_sum = np.sum(h)
            if abs(h_sum) > 1e-10:
                h = h / h_sum
            
            val = _evaluate_exact(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    # Strategy 2: Sparse spike optimization
    for n_spikes in [2, 3, 4, 5]:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        # Initialize with random spikes
        for trial in range(3):
            if time.time() - START_TIME > TIME_LIMIT - 20:
                break
            
            spike_pos = rng.choice(N, size=n_spikes, replace=False)
            spike_vals = rng.normal(0, 1, n_spikes)
            baseline_scale = 1.0
            
            params = np.concatenate([spike_pos.astype(float), spike_vals, [baseline_scale]])
            
            try:
                result = minimize(
                    _objective_sparse,
                    params,
                    args=(n_spikes,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-6}
                )
                
                if result.fun < best_val:
                    spike_pos_opt = result.x[:n_spikes].astype(int)
                    spike_pos_opt = np.clip(spike_pos_opt, 0, N-1)
                    spike_vals_opt = result.x[n_spikes:2*n_spikes]
                    baseline_scale_opt = result.x[2*n_spikes]
                    
                    h_opt = _sparse_to_h(spike_pos_opt, spike_vals_opt)
                    h_opt *= baseline_scale_opt
                    
                    h_sum = np.sum(h_opt)
                    if abs(h_sum) > 1e-10:
                        h_opt = h_opt / h_sum
                    
                    val_opt = _evaluate_exact(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt
            except Exception:
                pass
    
    # Strategy 3: Coordinate descent polish on best found
    if time.time() - START_TIME < TIME_LIMIT - 10:
        polished_h, polished_val = _coordinate_descent(best_h, max_iter=30)
        if polished_val < best_val:
            best_val = polished_val
            best_h = polished_h
    
    # Final check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N) / N
        
    return best_h

if __name__ == "__main__":
    h = construct()
    s = np.sum(h)
    conv = np.convolve(h, h)
    val = abs(2 * N * np.max(conv) / s**2)
    print(f"Score: {val}")
