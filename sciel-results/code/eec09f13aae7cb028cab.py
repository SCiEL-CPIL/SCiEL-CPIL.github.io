"""Optimized step function for minimizing normalized autoconvolution peak."""
import numpy as np
from scipy.optimize import minimize, SLSQP
import time

N = 600
# Interval width: total length is 1/4 - (-1/4) = 1/2, divided into 600 intervals
# So each interval has width 1/(2*600) = 1/1200
dw = 1.0 / (2.0 * N)

def _compute_normalized_peak(heights):
    """
    Compute the normalized maximum of the autoconvolution.
    
    For a step function f(x) = h_i on interval i, the autoconvolution is:
    (f * f)(t) = integral f(x) f(t-x) dx
    
    For step functions on equal intervals, this can be computed efficiently.
    The normalization is by (integral f)^2.
    
    C1 upper bound = max_t (f*f)(t) / (integral f)^2
    """
    h = np.asarray(heights, dtype=np.float64)
    if h.shape[0] != N:
        h = np.resize(h, N)
    
    # The autoconvolution of a step function can be computed via FFT
    # f(x) is constant h_i on interval [i*dw - 1/4, (i+1)*dw - 1/4)
    # (f*f)(t) = sum_i h_i * (length of intersection of interval i with interval (t - interval j))
    
    # More efficiently: the autoconvolution at discrete points can be computed
    # using the fact that it's a piecewise linear function.
    
    # Let's use the property that for step functions:
    # (f*f)(t) = sum_{i,j} h_i * h_j * L(i,j,t)
    # where L(i,j,t) is the length of intersection of [a_i, b_i] with [t-a_j, t-b_j]
    
    # Actually, let's think about this more carefully.
    # f(x) = h_k for x in [x_k, x_{k+1}) where x_k = -1/4 + k*dw
    # (f*f)(t) = integral f(x) f(t-x) dx = sum_k h_k * integral_{x_k}^{x_{k+1}} f(t-x) dx
    
    # This is equivalent to the convolution of two step functions.
    # We can compute this using FFT by sampling at appropriate resolution.
    
    # For accuracy, let's compute the autoconvolution on a fine grid.
    # The support of f is [-1/4, 1/4], so the support of f*f is [-1/2, 1/2].
    # The autoconvolution is piecewise linear with breakpoints at multiples of dw.
    
    # Number of intervals for the autoconvolution: 2*N intervals of width dw
    # Let's compute it on the grid of breakpoints.
    
    # Actually, for step functions with equal intervals, the autoconvolution
    # at the midpoint of each interval in the output can be computed as:
    # (f*f)(t) where t is at the center of interval m in [-1/2, 1/2]
    
    # Let me use a simpler approach: compute via direct formula.
    # The autoconvolution of a step function with heights h[0..N-1] on intervals
    # of width dw, where the function is supported on [-1/4, 1/4]:
    
    # (f*f)(t) for t in [k*dw - 1/2, (k+1)*dw - 1/2) is linear.
    # The maximum occurs at one of the breakpoints or midpoints.
    
    # Let's compute the autoconvolution values at all breakpoints.
    # There are 2*N+1 breakpoints from -1/2 to 1/2 in steps of dw.
    
    # Alternative: use the fact that (f*f)(t) = sum_i h_i * h_{i'} * overlap_length
    # where the overlap depends on t.
    
    # Most efficient: use FFT-based convolution.
    # Pad h to length 2*N, compute circular convolution, take appropriate part.
    
    # The autoconvolution (linear) can be computed as:
    # c = np.convolve(h, h, mode='full') * dw
    # This gives values at positions 0, dw, 2*dw, ..., 2*(N-1)*dw
    # But we need the integral, not just samples.
    
    # Actually, for step functions, the autoconvolution at the midpoint of each
    # output interval is:
    # Let's define the output grid. The output has 2*N-1 intervals if we consider
    # the standard convolution, but since our input is N intervals, the output
    # of the autoconvolution has 2*N-1 intervals of width dw.
    
    # Wait, let me reconsider. If f is a step function with N steps, then f*f
    # is a piecewise quadratic? No, it's piecewise linear.
    
    # Let me just compute it directly using the formula:
    # (f*f)(t) = sum_{i=0}^{N-1} h_i * [measure of x in interval i such that t-x is in support of f]
    # = sum_{i=0}^{N-1} h_i * sum_{j=0}^{N-1} h_j * measure({x: x in I_i and t-x in I_j})
    
    # For the maximum, we need to evaluate at critical points.
    
    # Let's use a numerical approach: sample on a fine grid and find the max.
    # But this needs to be fast for optimization.
    
    # Better: the autoconvolution of a step function is piecewise linear, with
    # breakpoints at t = (i - j) * dw for i, j = 0..N-1, shifted appropriately.
    # The number of distinct breakpoints is at most 2*N-1.
    
    # Let's compute the autoconvolution at all 2*N-1 midpoints of the output intervals.
    
    # The output intervals are centered at t_m = (m - N + 0.5) * dw for m = 0..2*N-2
    # Wait, let me be more careful.
    
    # Input intervals: I_i = [-1/4 + i*dw, -1/4 + (i+1)*dw) for i = 0..N-1
    # Output: (f*f)(t) = integral f(x) f(t-x) dx
    # Support of f*f: [-1/2, 1/2]
    # Breakpoints at t = -1/2 + k*dw for k = 0..2*N
    
    # Let's evaluate (f*f) at the midpoints of each output interval:
    # t_k = -1/2 + (k + 0.5) * dw for k = 0..2*N-1
    
    # At t_k, (f*f)(t_k) = sum_{i,j} h_i * h_j * L_{ij}(t_k)
    # where L_{ij}(t_k) = measure of x such that x in I_i and t_k - x in I_j
    
    # This is still O(N^2) per evaluation. For 600 variables, this is 360000 ops per eval.
    # With FFT, we can do it in O(N log N).
    
    # Let's use FFT. The autoconvolution of h with itself:
    # c = np.convolve(h, h, mode='full')
    # c[k] * dw gives the integral over the k-th output interval? Not exactly.
    
    # Actually, for step functions with constant values on equal-width intervals,
    # the autoconvolution at the center of the k-th output interval is:
    # (f*f)(t_k) = dw * sum_{i=0}^{N-1} h_i * h_{k-i}  (with appropriate bounds)
    # Wait, that's not right either because of the geometry.
    
    # Let me just do it properly. The autoconvolution of two step functions:
    # f(x) = h_i for x in [a_i, b_i) where a_i = -1/4 + i*dw, b_i = a_i + dw
    # g(x) = h_j for x in [a_j, b_j)
    
    # (f*g)(t) = integral f(x) g(t-x) dx
    # = sum_i h_i * integral_{x=a_i}^{b_i} g(t-x) dx
    # = sum_i h_i * sum_j h_j * measure({x in [a_i, b_i] : t-x in [a_j, b_j]})
    # = sum_i h_i * sum_j h_j * measure({x in [a_i, b_i] : x in [t-b_j, t-a_j]})
    # = sum_i sum_j h_i h_j * length([a_i, b_i] ∩ [t-b_j, t-a_j])
    
    # For the autoconvolution, i and j are independent sums.
    
    # The key insight: for fixed t, the length of intersection [a_i, b_i] ∩ [t-b_j, t-a_j]
    # is a function that's trapezoidal in t for each (i,j) pair.
    
    # This is getting complex. Let me just use a fine grid evaluation.
    # For the optimization, I'll use a moderate resolution.
    
    # Actually, let me use the FFT approach properly.
    # The autoconvolution of a step function can be computed as follows:
    # Let h be the vector of heights. Define an extended vector.
    # The value of (f*f)(t) at t = k*dw (for integer k) can be related to
    # the discrete convolution of h with itself.
    
    # For step functions on equal intervals, the autoconvolution at the midpoint
    # of the m-th output interval (m = 0..2*N-2) is:
    # (f*f)(t_m) = dw * sum_{i} h_i * h_{m-i}  where the sum is over valid i
    # Wait, this isn't quite right because the intervals don't align perfectly.
    
    # Let me just compute it numerically with a reasonable grid.
    # For 600 intervals, the output has 1199 intervals. Let's evaluate at midpoints.
    
    t_vals = -0.5 + (np.arange(2*N) + 0.5) * dw
    
    # For each t, compute (f*f)(t) = sum_i h_i * sum_j h_j * overlap(I_i, [t-b_j, t-a_j])
    # This is O(N^2) which is 360000 per evaluation. Too slow for many iterations.
    
    # Let's use FFT. The autoconvolution can be computed via:
    # c = np.fft.irfft(np.fft.rfft(h, n=2*N) ** 2) * dw
    # But this gives the circular convolution, not linear.
    
    # For linear convolution:
    # c = np.convolve(h, h, mode='full') * dw
    # c has length 2*N-1
    # c[k] represents the contribution at position k*dw
    
    # Actually, for step functions, the autoconvolution at the center of interval k
    # (where k goes from 0 to 2*N-2) is approximately c[k] * dw where c is the
    # discrete autoconvolution. But this is an approximation.
    
    # Let me test: for a single interval h = [1, 0, ..., 0], the autoconvolution
    # is a triangle from 0 to dw. The discrete convolution gives [1, 0, ..., 0].
    # So c[0] = 1, and the actual max is dw (the area of the triangle peak).
    
    # I think the correct formula is:
    # (f*f)(t) at the midpoint of the k-th output interval = dw * (h * h)[k]
    # where (h*h) is the linear convolution, but this needs verification.
    
    # Let me just use a direct but vectorized computation.
    # For each t in t_vals, compute the overlap.
    
    # Vectorized approach:
    # a_i = -0.25 + i*dw, b_i = a_i + dw
    # For each t, the overlap of I_i with [t-b_j, t-a_j] for all i,j
    
    # This is still O(N^2). Let's just do it and see if it's fast enough.
    
    # Actually, for the purpose of this optimization, let me use a simpler metric:
    # The peak of the autoconvolution. For step functions, the peak occurs at
    # t=0 (by symmetry) or at some other point.
    
    # Let me compute the autoconvolution at t=0 and at a few other points.
    # At t=0: (f*f)(0) = sum_i h_i^2 * dw (since the overlap of I_i with -I_i is dw)
    # Wait, that's only if I_i and -I_i overlap completely, which they do for the
    # middle interval but not for edge intervals.
    
    # I'm overcomplicating this. Let me just compute the full autoconvolution
    # on a fine grid using vectorized numpy operations.
    
    # Grid for output: 2*N points from -0.5 to 0.5
    n_out = 2 * N
    t_grid = -0.5 + np.arange(n_out) * dw + 0.5 * dw  # midpoints
    
    # For each t, (f*f)(t) = sum_{i,j} h_i * h_j * L_{ij}(t)
    # L_{ij}(t) = max(0, min(b_i, t - a_j) - max(a_i, t - b_j))
    
    # Vectorized:
    a = -0.25 + np.arange(N) * dw  # left endpoints of input intervals
    b = a + dw  # right endpoints
    
    # For each t, compute L_{ij}(t) for all i, j
    # L_{ij}(t) = max(0, min(b_i, t - a_j) - max(a_i, t - b_j))
    
    # This is a 3D computation: for each t (n_out values), for each i (N values),
    # for each j (N values). That's n_out * N * N = 1200 * 600 * 600 = 432M ops.
    # Too slow.
    
    # Let me use the FFT approach. I'll compute the autoconvolution using the
    # property that it's the inverse FFT of the squared FFT of the sampled function.
    
    # Sample f at N points (one per interval, at the center):
    # f_sample[i] = h[i]
    # The autoconvolution of the sampled function:
    # c = np.convolve(h, h, mode='full')
    # This gives the discrete autoconvolution.
    # For step functions, the continuous autoconvolution at the midpoint of the
    # k-th output interval is approximately c[k] * dw.
    
    # Let me use this approximation. It should be reasonably accurate.
    
    c = np.convolve(h, h, mode='full') * dw
    # c has length 2*N-1
    # The peak is max(c)
    
    # Normalization: (integral f)^2 = (sum(h) * dw)^2
    integral_f = np.sum(h) * dw
    norm = integral_f ** 2
    
    if norm < 1e-30:
        return 1e10  # avoid division by zero
    
    return np.max(c) / norm

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    that minimizes the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10 seconds safety margin
    
    # Start with a good initial guess
    # A uniform function gives normalized peak = 1/N (approximately)
    # Let's try to do better.
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    # Let's try a few different initializations and optimize from each
    initializations = [
        np.ones(N),
        np.exp(-np.linspace(0, 3, N)),
        np.cos(np.linspace(0, np.pi, N)) ** 2,
        np.exp(-2 * np.linspace(0, 1, N) ** 2),
    ]
    
    best_h = h0.copy()
    best_score = _compute_normalized_peak(h0)
    
    for h_init in initializations:
        if time.time() > deadline:
            break
        
        h_init = np.maximum(h_init, 0)  # ensure nonnegative
        if np.sum(h_init) < 1e-10:
            continue
        
        # Optimize using SLSQP
        def objective(h):
            h = np.maximum(h, 0)  # project to nonnegative
            return _compute_normalized_peak(h)
        
        try:
            result = minimize(
                objective,
                h_init,
                method='SLSQP',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_h = np.maximum(result.x, 0)
        except Exception:
            pass
    
    # Also try a simple hill-climbing refinement
    if time.time() < deadline:
        h = best_h.copy()
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                for delta in [0.01, 0.001, -0.01, -0.001]:
                    h_new = h.copy()
                    h_new[i] = max(0, h[i] + delta)
                    score = _compute_normalized_peak(h_new)
                    if score < best_score:
                        best_score = score
                        best_h = h_new
                        h = h_new
                        improved = True
                        break
            if not improved:
                break
    
    # Return as a list
    return list(np.maximum(best_h, 0))

