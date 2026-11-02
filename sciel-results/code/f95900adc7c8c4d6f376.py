"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _autoconvolve_max_ratio(h):
    """Compute max_x (f*f)(x) / (∫f)^2 for step function h on N equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    integral = np.sum(h)
    if integral <= 1e-12:
        return 1e10
    # Autoconvolution of step function: use np.convolve
    # The step function on [-1/4, 1/4] with N intervals, each of width w = 1/(2N)
    # Autoconvolution (f*f)(x) = ∫ f(t) f(x-t) dt
    # For step functions, this is a piecewise linear function
    # We can compute it via discrete convolution scaled appropriately
    # 
    # Actually, for the normalized ratio, the width cancels out in a specific way.
    # Let's think: if f(t) = h_i on interval i, width w, then ∫f = w * sum(h_i)
    # (f*f)(x) involves convolution. The peak value relative to (∫f)^2 is what matters.
    # 
    # For a step function, the autoconvolution at the center (x=0) is:
    # (f*f)(0) = sum_i h_i^2 * w  (since each pair (i,i) contributes h_i * h_i * w)
    # Wait, that's not quite right for continuous convolution.
    #
    # Let me use a simpler approach: sample the autoconvolution on a fine grid.
    # Actually, for step functions, the autoconvolution is piecewise quadratic/linear.
    # 
    # Simpler: use np.convolve to get the discrete autoconvolution, then interpret.
    # The discrete autoconvolution c_k = sum_i h_i * h_{k-i} (with appropriate bounds)
    # For the continuous case with equal-width intervals, the peak of the continuous
    # autoconvolution is related to the max of the discrete convolution.
    #
    # Let's just compute: the continuous autoconvolution peak / (integral)^2
    # For uniform h=1, N intervals, width w=1/(2N):
    #   integral = N*w = 1/2
    #   (f*f)(0) = integral of f(t)f(-t)dt = integral of 1*1 dt over [-1/4,1/4] = 1/2
    #   ratio = (1/2) / (1/2)^2 = 2
    #
    # For general h, I'll compute the autoconvolution numerically.
    # Use a fine grid to evaluate (f*f)(x) at many points.
    
    # Actually, let's use the fact that for step functions, the autoconvolution
    # can be computed exactly. But for optimization, let's use a sampled version.
    
    # Compute autoconvolution via FFT for speed
    # Pad to avoid circular convolution
    n_fft = 2 * N + 1
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    conv = np.real(np.fft.ifft(np.fft.fft(h_padded, n=2*n_fft) ** 2))[:2*N-1]
    # The continuous autoconvolution peak is approximately max(conv) * w, where w is interval width
    # And (∫f)^2 = (w * sum(h))^2
    # So ratio = max(conv) * w / (w^2 * sum(h)^2) = max(conv) / (w * sum(h)^2)
    # But w = 1/(2N), so ratio = max(conv) * 2*N / sum(h)^2
    # Wait, let me re-derive.
    
    # f(t) = h_i for t in [(i-N/2)*w, (i-N/2+1)*w), i=0..N-1, w=1/(2N)
    # ∫f = w * sum(h)
    # (f*f)(x) = ∫ f(t) f(x-t) dt
    # For x=0: (f*f)(0) = ∫ f(t) f(-t) dt = sum_i h_i^2 * w (each interval i contributes h_i * h_i * w)
    #   Wait, that's only if the intervals align symmetrically. Let me be more careful.
    #
    # Actually, the discrete convolution approach: if we define g_k = (f*f)(k*w) approximately,
    # then g_k ≈ w * sum_i h_i * h_{k-i} (with h_j=0 outside [0,N-1])
    # So max_x (f*f)(x) ≈ w * max_k sum_i h_i * h_{k-i} = w * max(conv)
    # And (∫f)^2 = (w * sum(h))^2
    # Ratio = w * max(conv) / (w^2 * sum(h)^2) = max(conv) / (w * sum(h)^2)
    
    w = 1.0 / (2.0 * N)
    max_conv = np.max(conv)
    integral = w * np.sum(h)
    if integral <= 1e-12:
        return 1e10
    return max_conv * w / (integral * integral)


def construct():
    start_time = time.time()
    
    # Start with uniform
    h = np.ones(N)
    
    # Compute initial objective
    best_h = h.copy()
    best_val = _autoconvolve_max_ratio(h)
    
    # Use projected gradient descent to minimize the ratio
    # The ratio is homogeneous, so we can fix sum(h) = 1 and minimize max(conv)
    # Then the ratio = max(conv) / w where w=1/(2N), so minimize max(conv)
    
    # Normalize: let h such that sum(h) = 1
    h_norm = h / np.sum(h)
    
    # Objective: minimize max of autoconvolution (discrete)
    def objective(h_vec):
        h_vec = np.maximum(h_vec, 0)  # project to nonnegative
        if np.sum(h_vec) < 1e-12:
            return 1e10
        h_vec = h_vec / np.sum(h_vec)  # normalize
        n_fft = 2 * N + 1
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h_vec
        conv = np.real(np.fft.ifft(np.fft.fft(h_padded, n=2*n_fft) ** 2))[:2*N-1]
        return np.max(conv)
    
    # Use L-BFGS-B with nonnegativity constraints
    bounds = [(0, None)] * N
    result = minimize(objective, h_norm, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if time.time() - start_time > DEADLINE - 10:
        return best_h.tolist()
    
    if result.fun < best_val / (1.0/(2.0*N)):  # compare properly
        # Convert back
        h_opt = np.maximum(result.x, 0)
        h_opt = h_opt / np.sum(h_opt)
        # The actual ratio for this h
        val = _autoconvolve_max_ratio(h_opt)
        if val < best_val:
            best_h = h_opt
            best_val = val
    
    # Try a few different initializations
    for init_type in ['bump', 'triangle', 'random']:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        if init_type == 'bump':
            h_init = np.exp(-10 * (np.linspace(-0.5, 0.5, N))**2)
        elif init_type == 'triangle':
            x = np.linspace(-0.5, 0.5, N)
            h_init = np.maximum(0, 1 - 2*abs(x))
        else:
            rng = np.random.RandomState(42)
            h_init = rng.random(N)
        
        h_init = np.maximum(h_init, 0)
        if np.sum(h_init) < 1e-12:
            continue
        h_init = h_init / np.sum(h_init)
        
        result = minimize(objective, h_init, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if time.time() - start_time > DEADLINE - 10:
            break
        
        if result.fun < 1e-10:
            h_opt = np.maximum(result.x, 0)
            h_opt = h_opt / np.sum(h_opt)
            val = _autoconvolve_max_ratio(h_opt)
            if val < best_val:
                best_h = h_opt
                best_val = val
    
    # Final polish: try more iterations on best
    if time.time() - start_time < DEADLINE - 20:
        h_best_norm = best_h / np.sum(best_h)
        result = minimize(objective, h_best_norm, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.fun < 1e-10:
            h_opt = np.maximum(result.x, 0)
            h_opt = h_opt / np.sum(h_opt)
            val = _autoconvolve_max_ratio(h_opt)
            if val < best_val:
                best_h = h_opt
                best_val = val
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
