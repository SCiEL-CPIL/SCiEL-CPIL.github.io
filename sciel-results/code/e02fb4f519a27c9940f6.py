import numpy as np
from scipy.optimize import minimize
import time

N = 600
W = 1.0 / 1200.0  # interval width = (1/4 - (-1/4)) / 600 = 0.5/600

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s wall-clock limit
    
    def compute_convolution_max(h):
        """Compute normalized max of autoconvolution of step function h."""
        # h: array of N nonnegative heights
        # Step function on [-1/4, 1/4] with N equal intervals
        
        # The autoconvolution g(x) = ∫ f(t) f(x-t) dt
        # For step functions, g is piecewise linear with breakpoints at x = a_i - a_j
        # where a_i are interval boundaries.
        
        # Efficient computation: 
        # g(x) = sum_i h[i] * (∫_{a_i}^{a_{i+1}} f(x-t) dt)
        # The inner integral is the measure of overlap between [x - a_{i+1}, x - a_i] and [-1/4, 1/4]
        # weighted by the step heights of f.
        
        # Alternative: use the fact that for a step function, the autoconvolution can be
        # computed by summing over all pairs (i,j) of the product h[i]*h[j] times the
        # length of the set {x : x-t ∈ interval_j for some t in interval_i}
        
        # Let's use a direct but efficient approach:
        # The breakpoints of g are at x = a_i - a_j for all i,j, which are multiples of W.
        # So g is linear on intervals of width W, centered at multiples of W.
        
        # Number of distinct breakpoints: from -1/2 to 1/2 in steps of W, so 1200+1 points
        # g is determined by its values at these breakpoints (it's linear between them).
        
        # Actually, for a step function with N intervals, the autoconvolution has at most 
        # 2N-1 linear pieces. Let me compute it properly.
        
        # Let's define the interval boundaries
        # a[i] = -1/4 + i*W for i=0..N
        # f(t) = h[i] for t in [a[i], a[i+1])
        
        # g(x) = ∫ f(t) f(x-t) dt
        # = sum_i h[i] * ∫_{a[i]}^{a[i+1]} f(x-t) dt
        # = sum_i h[i] * sum_j h[j] * length({t in [a[i], a[i+1]] : x-t in [a[j], a[j+1]]})
        # = sum_i sum_j h[i] h[j] * length([a[i], a[i+1]] ∩ [x - a[j+1], x - a[j]])
        
        # The length of intersection of two intervals [p1,p2] and [q1,q2] is:
        # max(0, min(p2,q2) - max(p1,q1))
        
        # For fixed x, this is complex. Let's use a different approach.
        
        # Since N=600, let's just sample g at many points and find the max.
        # g is piecewise linear with breakpoints at multiples of W in [-1/2, 1/2].
        # There are 1201 breakpoints. g is linear between consecutive breakpoints.
        # So the max occurs at one of the breakpoints or we can evaluate at all breakpoints.
        
        # Let's evaluate g at x = k*W for k = -600, ..., 600 (1201 points)
        # Actually, the breakpoints are at x = (i-j)*W for i,j in 0..N, so x in [-N*W, N*W] = [-0.5, 0.5]
        
        n_break = 2 * N + 1  # 1201 breakpoints
        x_break = np.arange(-N, N+1) * W
        
        # For each x_break[k], compute g(x_break[k])
        # g(x) = sum_i h[i] * (∫_{a[i]}^{a[i+1]} f(x-t) dt)
        
        # For a given x, ∫_{a[i]}^{a[i+1]} f(x-t) dt = sum_j h[j] * length([a[i], a[i+1]] ∩ [x-a[j+1], x-a[j]])
        
        # This is O(N^2) per x, and we have O(N) x values, so O(N^3) total = 216M operations.
        # That's too slow. Let's use FFT.
        
        # Better: represent f on a fine grid and use FFT convolution.
        n_fine = 4800  # 8x oversampling
        w_fine = 1.0 / (2 * n_fine)  # total range 1, so bin width
        # Grid: x_k = -0.5 + (k+0.5)*w_fine for k=0..n_fine-1
        x_fine = -0.5 + (np.arange(n_fine) + 0.5) * w_fine
        
        # Map h to fine grid
        # h[i] is on [a[i], a[i+1]) where a[i] = -0.25 + i*W
        # For each fine grid point, determine which interval it falls in
        idx = ((x_fine + 0.25) / W).astype(int)
        idx = np.clip(idx, 0, N-1)
        mask = (x_fine >= -0.25) & (x_fine < 0.25)
        f_fine = np.zeros(n_fine)
        f_fine[mask] = h[idx[mask]]
        
        # FFT convolution
        f_fft = np.fft.rfft(f_fine)
        g_fine = np.fft.irfft(f_fft * f_fft, n=n_fine)
        g_fine *= w_fine
        
        # Normalize
        integral_f = np.sum(f_fine) * w_fine
        if integral_f < 1e-12:
            return 1e10
        norm = integral_f ** 2
        return np.max(g_fine) / norm
    
    # Initial guesses
    best_h = np.ones(N)
    best_val = compute_convolution_max(best_h)
    
    # Try a few initial shapes
    initial_guesses = [
        np.ones(N),
        np.linspace(0.5, 1.5, N),
        0.5 * (1 + np.cos(np.linspace(0, np.pi, N))),
        np.exp(-2 * ((np.linspace(-1, 1, N))**2)),
    ]
    
    for h0 in initial_guesses:
        h0 = np.clip(h0, 0, None)
        val0 = compute_convolution_max(h0)
        if val0 < best_val:
            best_h = h0.copy()
            best_val = val0
    
    # Optimize from best initial guess
    bounds = [(0, None)] * N
    
    try:
        res = minimize(compute_convolution_max, best_h, method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-6})
        if res.fun < best_val:
            best_h = res.x
            best_val = res.fun
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return list(best_h)
    
    # Try more optimization from different starts
    for h0 in initial_guesses[1:]:
        if time.time() > deadline - 10:
            break
        h0 = np.clip(h0, 0, None)
        try:
            res = minimize(compute_convolution_max, h0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-6})
            if res.fun < best_val:
                best_h = res.x
                best_val = res.fun
        except Exception:
            pass
    
    return list(best_h)
