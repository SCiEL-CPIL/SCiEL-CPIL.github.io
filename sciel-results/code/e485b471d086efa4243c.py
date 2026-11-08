import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
def _compute_overlap_values(h):
    """
    Compute the overlap of h with shifted 1-h for all shifts k in 0..N-1.
    O(k) = sum_i h[i] * (1 - h[(i-k) % N]) = N/2 - R(k)
    where R(k) is the cyclic autocorrelation of h.
    """
    H = np.fft.fft(h)
    power_spectrum = H * np.conj(H)
    R = np.fft.ifft(power_spectrum).real
    return N/2 - R

def _soft_max_and_grad(overlaps, beta):
    """
    Compute soft-max and its gradient.
    soft_max = max + (1/beta) * log(sum(exp(beta*(x - max))))
    grad_i = exp(beta*(x_i - max)) / sum(exp(beta*(x_j - max)))
    """
    max_val = np.max(overlaps)
    shifted = beta * (overlaps - max_val)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    soft_max = max_val + np.log(sum_exp) / beta
    # Gradient: softmax weights
    grad = exp_shifted / sum_exp
    return soft_max, grad

def _objective_value_grad(h, beta):
    """
    Objective: soft-max of overlaps, with analytic gradient.
    O(k) = N/2 - R(k), where R = ifft(|FFT(h)|^2)
    dO(k)/dh = -dR(k)/dh
    
    R(k) = sum_i h[i] * h[(i-k) % N]
    dR(k)/dh[j] = h[(j-k) % N] + h[(j+k) % N]  (for k != 0)
    dR(0)/dh[j] = 2*h[j]
    
    So dO(k)/dh[j] = -h[(j-k) % N] - h[(j+k) % N] for k != 0
    dO(0)/dh[j] = -2*h[j]
    
    grad_obj[j] = sum_k grad_soft(k) * dO(k)/dh[j]
    """
    overlaps = _compute_overlap_values(h)
    soft_max, soft_grad = _soft_max_and_grad(overlaps, beta)
    
    # Compute gradient analytically
    # grad_obj[j] = -sum_k soft_grad[k] * (h[(j-k)%N] + h[(j+k)%N]) for k>0, 
    #              -2*soft_grad[0]*h[j] for k=0
    # This is a convolution-like operation.
    # Let's compute it via FFT for efficiency.
    # dO(k)/dh = -2 * (circular shift of h by k), but we need sum over k.
    # Actually: grad_obj = -sum_k soft_grad[k] * (shift_left(h, k) + shift_right(h, k))
    # where shift_left(h, k)[j] = h[(j-k)%N], shift_right(h, k)[j] = h[(j+k)%N]
    
    # Use FFT: sum_k soft_grad[k] * shift_left(h, k) is cross-correlation
    # Let's just do it directly for N=100, it's fast enough.
    grad_obj = np.zeros(N)
    for k in range(N):
        if k == 0:
            grad_obj -= 2.0 * soft_grad[k] * h
        else:
            # shift_left: h[(j-k)%N]
            shift_left = np.roll(h, k)
            # shift_right: h[(j+k)%N]  
            shift_right = np.roll(h, -k)
            grad_obj -= soft_grad[k] * (shift_left + shift_right)
    
    return soft_max, grad_obj

def _project_to_box(h):
    """Project h onto [0,1]^N while preserving sum = N/2 as much as possible."""
    h_clipped = np.clip(h, 0, 1)
    # Water-filling to restore sum
    target_sum = N / 2.0
    current_sum = np.sum(h_clipped)
    if abs(current_sum - target_sum) < 1e-10:
        return h_clipped
    
    # Iterative water filling
    for _ in range(100):
        diff = target_sum - np.sum(h_clipped)
        if abs(diff) < 1e-12:
            break
        # Find elements not at bounds
        can_increase = h_clipped < 1.0
        can_decrease = h_clipped > 0.0
        
        if diff > 0:
            # Need to increase some elements
            mask = can_increase
            if not np.any(mask):
                break
            n_mask = np.sum(mask)
            step = diff / n_mask
            # Can't exceed 1
            max_step = 1.0 - h_clipped[mask]
            actual_step = min(step, np.min(max_step))
            h_clipped[mask] += actual_step
        else:
            # Need to decrease some elements
            mask = can_decrease
            if not np.any(mask):
                break
            n_mask = np.sum(mask)
            step = diff / n_mask
            # Can't go below 0
            min_step = -h_clipped[mask]
            actual_step = max(step, np.min(min_step))
            h_clipped[mask] += actual_step
    
    return h_clipped

def construct():
    start_time = time.time()
    deadline = 120.0
    margin = 15.0
    
    # Initial guess: Uniform
    h0 = np.full(N, 0.5)
    
    # Multi-start strategy
    best_h = h0.copy()
    best_score = _compute_overlap_values(h0).max()
    
    bounds = [(0, 1)] * N
    rng = np.random.default_rng(42)
    
    num_starts = 30
    beta_schedule = [5.0, 10.0, 20.0, 40.0]
    
    for i in range(num_starts):
        if time.time() > deadline - margin:
            break
        
        # Generate diverse initial points
        if i == 0:
            h_init = h0.copy()
        elif i == 1:
            # Binary-like pattern
            h_init = np.zeros(N)
            h_init[::2] = 1.0
            h_init = _project_to_box(h_init)
        elif i == 2:
            # Another binary pattern
            h_init = np.zeros(N)
            h_init[::3] = 1.5
            h_init = _project_to_box(h_init)
        else:
            # Random Fourier modes
            h_init = h0.copy()
            # Add random low-frequency components
            num_modes = rng.integers(1, 5)
            for _ in range(num_modes):
                freq = rng.integers(1, N//2 + 1)
                phase = rng.uniform(0, 2*np.pi)
                amp = rng.uniform(0.1, 0.3)
                h_init += amp * np.cos(2*np.pi*freq*np.arange(N)/N + phase)
            # Add small noise
            h_init += rng.normal(0, 0.05, N)
            h_init = _project_to_box(h_init)
        
        # Annealing over beta values
        current_h = h_init.copy()
        for beta in beta_schedule:
            if time.time() > deadline - margin:
                break
            
            # Run L-BFGS-B with current beta
            res = minimize(
                lambda x: _objective_value_grad(x, beta),
                current_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 50}
            )
            
            h_res = res.x
            h_res = _project_to_box(h_res)
            
            score = _compute_overlap_values(h_res).max()
            
            if score < best_score:
                best_score = score
                best_h = h_res.copy()
            
            current_h = h_res
        
        # Final polish with high beta
        if time.time() > deadline - margin:
            break
        
        beta_final = 100.0
        res = minimize(
            lambda x: _objective_value_grad(x, beta_final),
            current_h,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
        )
        
        h_res = res.x
        h_res = _project_to_box(h_res)
        
        score = _compute_overlap_values(h_res).max()
        
        if score < best_score:
            best_score = score
            best_h = h_res.copy()
    
    # Final check
    final_h = _project_to_box(best_h)
    final_score = _compute_overlap_values(final_h).max()
    
    if final_score < best_score:
        best_h = final_h
        best_score = final_score
    
    return best_h
# EVOLVE-BLOCK-END
