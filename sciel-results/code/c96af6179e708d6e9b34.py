"""Minimize normalized max autoconvolution of nonnegative step function on [-1/4,1/4]."""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
T_START = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _time_left():
    return TIME_LIMIT - (time.time() - T_START)

def _autoconv_peak(heights):
    """Compute max |autoconv| / sum(heights)^2 * N (normalized)."""
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    total = h.sum()
    if total < 1e-15:
        return 1e10
    # autoconv: (h*h)[k] = sum_i h[i]*h[i+k-n+1] for circular? 
    # Actually for linear autoconvolution of step function on equal intervals:
    # The continuous autoconvolution at x is integral f(t)f(x-t)dt.
    # For step function with heights h[i] on interval i, autoconv is a piecewise quadratic.
    # But we can compute the discrete autoconv and scale.
    # Let's use FFT for circular autoconv as approximation, but the task says autoconvolution.
    # For a step function on [a,b], the autoconv support is [2a,2b].
    # With 600 equal intervals on [-1/4,1/4], each interval width w = (1/4 - (-1/4))/600 = 0.5/600.
    # The autoconv at the center is maximized when the function is concentrated.
    # Let me just compute the linear autoconv via FFT on a padded array.
    h_lin = np.zeros(2*n-1)
    h_lin[:n] = h
    ac = np.convolve(h, h, mode='full')
    # The continuous autoconv value at a point is approximately (w^2) * sum of products
    # where w is interval width. The normalized max is max(ac) * w^2 / (sum(h*w))^2 * something.
    # Actually: f(x) = h[i] on interval i. Autoconv g(x) = ∫f(t)f(x-t)dt.
    # g is max at x=0 if f is symmetric, but we can shift.
    # The peak of g is what matters. For step functions, the autoconv is a convolution.
    # Let's just compute max(ac) * w^2 / (total*w)^2 = max(ac)/total^2 since w cancels in ratio? 
    # No: total mass = sum(h[i]*w) = w*sum(h). 
    # g_peak ≈ max(ac) * w^2 (since each product h[i]h[j] contributes over area w^2)
    # normalized = g_peak / (mass)^2 = max(ac)*w^2 / (w^2 * total^2) = max(ac)/total^2.
    return max(ac) / (total * total)

def construct():
    t0 = time.time()
    
    # Initial guess: try a few different starting points
    best_h = None
    best_val = 1e10
    
    # Try uniform (baseline)
    h0 = np.ones(N)
    val0 = _autoconv_peak(h0)
    if val0 < best_val:
        best_val = val0
        best_h = h0.copy()
    
    # Try a bump (Gaussian-like)
    x = np.linspace(-1, 1, N)
    h_bump = np.exp(-x**2 * 5)
    h_bump = np.maximum(h_bump, 0)
    val_bump = _autoconv_peak(h_bump)
    if val_bump < best_val:
        best_val = val_bump
        best_h = h_bump.copy()
    
    # Try a flat-top
    h_flat = np.zeros(N)
    h_flat[N//4:3*N//4] = 1.0
    val_flat = _autoconv_peak(h_flat)
    if val_flat < best_val:
        best_val = val_flat
        best_h = h_flat.copy()
    
    # Use L-BFGS-B to optimize
    def objective(h):
        return _autoconv_peak(h)
    
    def gradient(h):
        # Numerical gradient (central differences)
        eps = 1e-6
        grad = np.zeros_like(h)
        for i in range(N):
            hp = h.copy()
            hp[i] += eps
            hm = h.copy()
            hm[i] -= eps
            grad[i] = (objective(hp) - objective(hm)) / (2*eps)
        return grad
    
    # We need to be careful: N=600, numerical gradient is expensive (600 evaluations per step)
    # Each evaluation is O(N log N) with FFT or O(N^2) with convolve.
    # N=600, N^2 = 360000, that's fine for a few iterations but not for hundreds.
    # Let me use a smarter approach: since the objective is smooth, use analytical gradient if possible.
    # Or use a smaller number of parameters.
    
    # Alternative: parameterize as a sum of a few Gaussians or use a low-dimensional basis.
    # Let me try: h[i] = max(0, a * exp(-((x_i - mu)^2)/sigma^2))
    # But that's 3 parameters. The optimal might be more complex.
    
    # Let me try using scipy.optimize.minimize with L-BFGS-B and finite differences,
    # but limit the number of iterations.
    
    # Actually, let me try a different approach: 
    # The autoconv peak is maximized when the function is concentrated.
    # To minimize the peak, we want to spread the mass out.
    # But the L1 norm is fixed (we can normalize).
    # Wait, the normalization: the score is max(autoconv)/sum(h)^2 * N? 
    # Let me re-read the task.
    
    # "minimize the normalized maximum of its autoconvolution"
    # The normalization is likely by the square of the L1 norm.
    # So we want to minimize max(autoconv) / (L1)^2.
    
    # For a step function with 600 intervals, the autoconv at the center for a uniform function:
    # Each pair (i,j) contributes h[i]*h[j]*w^2 to the autoconv at position (i+j)*w.
    # The peak is at position 0 (center), which gets contributions from all pairs where i+j is near N/2.
    # For uniform h=1, the autoconv at center is approximately N*w^2 (since there are ~N pairs).
    # L1 = N*w, so (L1)^2 = N^2*w^2.
    # normalized = N*w^2/(N^2*w^2) = 1/N.
    # So uniform gives 1/600 ≈ 0.00167.
    
    # Can we do better? If we spread the mass more, the peak might decrease.
    # But the mass is conserved. The minimum possible peak for a given L1 is achieved
    # by a function that is as "flat" as possible, but constrained to be nonnegative.
    # Actually, the autoconv peak is minimized when the function is as spread out as possible.
    # But we're already on a fixed domain.
    
    # Let me just try to optimize with a few iterations of L-BFGS-B.
    
    # Use a simpler parameterization: h = 1 + eps * u, where u has zero mean and is small.
    # But that might not be enough.
    
    # Let me try: optimize h directly with L-BFGS-B, but use a smaller effective N.
    # Block the 600 values into groups of 10, so 60 parameters.
    
    block_size = 10
    n_blocks = N // block_size  # 60
    h_init = np.tile(best_h[:n_blocks*block_size].reshape(n_blocks, block_size).mean(axis=1), (1, block_size)).ravel()
    
    def obj_block(b):
        h = np.repeat(b, block_size)
        return _autoconv_peak(h)
    
    result = minimize(obj_block, h_init, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
    
    h_opt = np.repeat(result.x, block_size)
    val_opt = _autoconv_peak(h_opt)
    
    if val_opt < best_val:
        best_val = val_opt
        best_h = h_opt
    
    # Check time
    if _time_left() < 5:
        return list(np.maximum(best_h, 0))
    
    # Try to refine with full N
    h_start = best_h.copy()
    h_start = np.maximum(h_start, 0)
    
    # Use a few iterations of gradient descent with analytical gradient
    # The autoconv ac[k] = sum_i h[i]*h[i+k-N+1] for circular, but for linear it's different.
    # Let me just use numerical gradient but with a larger step size and fewer dimensions.
    
    # Actually, let me try a different strategy: 
    # The optimal function for minimizing the peak autoconv given fixed L1 is 
    # related to the prolate spheroidal wave function or similar.
    # But for step functions, it might be a flat function with some shaping.
    
    # Let me try: h[i] = 1 + a*cos(2*pi*i/N) for small a, and optimize a.
    for a in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        h_cos = np.maximum(1 + a*np.cos(2*np.pi*np.arange(N)/N), 0)
        val_cos = _autoconv_peak(h_cos)
        if val_cos < best_val:
            best_val = val_cos
            best_h = h_cos.copy()
    
    # Also try sin
    for a in [0.1, 0.2, 0.3, 0.5]:
        h_sin = np.maximum(1 + a*np.sin(2*np.pi*np.arange(N)/N), 0)
        val_sin = _autoconv_peak(h_sin)
        if val_sin < best_val:
            best_val = val_sin
            best_h = h_sin.copy()
    
    return list(np.maximum(best_h, 0))
