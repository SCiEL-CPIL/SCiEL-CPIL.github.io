"""
Third autocorrelation inequality: minimize the upper bound on C3.
The score is seed/value where value is the raw upper bound (lower is better).
We search over signed step functions with 400 intervals on [-1/4, 1/4].
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 115  # safety margin before 120s kill

def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    
    For a step function f(x) = sum_i h_i * 1_{[x_i, x_{i+1}]}(x) on [-1/4, 1/4],
    the autoconvolution (f * f)(t) = integral f(s) f(t-s) ds.
    
    The C3 bound relates to the L-infinity norm of the autoconvolution.
    Specifically, for the third autocorrelation inequality, the bound is:
        C3 <= max_t |(f*f)(t)|
    
    where f is the signed step function. We compute this via FFT for efficiency.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    # Interval width
    dx = 0.5 / n  # total length 0.5, n intervals
    
    # The autoconvolution of a piecewise constant function:
    # (f*f)(t) for t in [-0.5, 0.5]
    # We can compute this by treating h as samples and using convolution.
    # The autoconvolution at lag k (k from -n+1 to n-1) is:
    #   (f*f)_k = dx * sum_{i} h_i * h_{i+k}  (with appropriate indexing)
    # Actually, for a step function with constant value h_i on interval [i*dx, (i+1)*dx),
    # the autoconvolution at offset m*dx (for m = -(n-1), ..., n-1) is:
    #   (f*f)(m*dx) = dx * sum_{i=max(0,-m)}^{min(n-1, n-1-m)} h_i * h_{i+m}
    # But actually the autoconvolution of a piecewise constant function is a piecewise 
    # linear function. The maximum of |f*f| occurs at the endpoints of the intervals
    # where the linear pieces change, which are at multiples of dx.
    
    # Let's compute the autoconvolution at the grid points.
    # Using FFT: the DFT of the autoconvolution relates to |DFT(f)|^2.
    # But for a piecewise constant function, we need to be careful.
    
    # Simpler approach: direct computation via convolution
    # f*f at lag k for k = 0, 1, ..., n-1 (and symmetric for negative)
    # (f*f)(k*dx) = dx * sum_{i=0}^{n-1-k} h_i * h_{i+k}  for k >= 0
    # By symmetry, (f*f)(-k*dx) = (f*f)(k*dx)
    
    # Actually, let me reconsider. The autoconvolution (f*f)(t) = integral_{-1/4}^{1/4} f(s) f(t-s) ds.
    # For t = m*dx where m is an integer:
    # The integral is over s such that s in [-1/4, 1/4] and t-s in [-1/4, 1/4].
    # This gives s in [max(-1/4, t-1/4), min(1/4, t+1/4)].
    
    # Let's just compute it numerically at the grid points using a straightforward method.
    # For efficiency, use FFT-based convolution.
    
    # Pad h to length 2n-1 for linear convolution
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    # Linear autoconvolution via FFT
    f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    # The autoconvolution values at the grid points
    # conv[k] = sum_i h_pad[i] * h_pad[k-i] for valid k
    # We want the autoconvolution of the step function, which is:
    # (f*f)(t) = dx * (discrete autoconvolution) at appropriate points
    
    # Actually, the continuous autoconvolution at t = m*dx is:
    # (f*f)(m*dx) = dx * sum_{i} h_i * h_{i+m}  (for the part where both are in support)
    # This is exactly dx * conv[m] for m >= 0, and by symmetry dx * conv[-m] for m < 0.
    
    # But conv from FFT gives the linear convolution, which for autoconvolution is:
    # conv[k] = sum_{i=0}^{n-1} h[i] * h[k-i] where h is zero-padded
    # For k = 0: sum h[i]^2
    # For k = 1: sum h[i] * h[i-1] (with h[-1]=0)
    
    # The continuous autoconvolution at offset m*dx (m integer) is:
    # (f*f)(m*dx) = dx * sum_{i=max(0,-m)}^{min(n-1, n-1-m)} h[i] * h[i+m]
    # For m >= 0: dx * sum_{i=0}^{n-1-m} h[i] * h[i+m] = dx * conv[m]
    # For m < 0: by symmetry, same as |m|
    
    # So the max |autoconvolution| is dx * max_k |conv[k]|
    # But wait, the autoconvolution of a piecewise constant function is piecewise linear,
    # and its maximum on each linear segment occurs at the endpoints, which are at multiples of dx.
    # So the global max is indeed at one of the grid points.
    
    # However, I need to double-check: is the autoconvolution maximum really at the grid points?
    # The autoconvolution (f*f)(t) for a step function is a piecewise linear function.
    # On each interval between consecutive grid points, it's linear, so the max is at an endpoint.
    # The endpoints are at t = m*dx for integer m. So yes.
    
    # Compute max absolute value
    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs
    
    return c3_bound


def _local_search(heights, max_iter=5000, step_size=0.5):
    """
    Local search: try flipping individual bits (changing sign) to reduce the C3 bound.
    """
    h = np.array(heights, dtype=np.float64)
    current_bound = _compute_c3_bound(h)
    
    # Precompute for efficiency: we'll try flipping each position
    # For efficiency, we can compute the effect of flipping h[i] on the autoconvolution.
    # Flipping h[i] to -h[i] changes the autoconvolution.
    # This is complex, so we'll just recompute for now but limit iterations.
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        best_gain = 0
        best_idx = -1
        
        for i in range(N):
            old_val = h[i]
            h[i] = -old_val
            new_bound = _compute_c3_bound(h)
            h[i] = old_val
            
            gain = current_bound - new_bound
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            h[best_idx] = -h[best_idx]
            current_bound -= best_gain
            improved = True
    
    return h, current_bound


def construct():
    start_time = time.time()
    
    # Initialize with random signs
    np.random.seed(42)
    best_h = np.ones(N)
    best_bound = _compute_c3_bound(best_h)
    
    # Try a few random initializations
    for trial in range(5):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local search from best initialization
    if time.time() - start_time < TIME_LIMIT - 10:
        best_h, best_bound = _local_search(best_h, max_iter=3000)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        for trial in range(3):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            h = np.random.choice([-1.0, 1.0], size=N)
            bound = _compute_c3_bound(h)
            if bound < best_bound:
                h, bound = _local_search(h, max_iter=1000)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    # Final check
    final_bound = _compute_c3_bound(best_h)
    if final_bound < best_bound:
        best_bound = final_bound
    
    return best_h


# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")
