import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3_fft(signed_step):
    """
    Compute C3 upper bound using FFT with exact linear interpolation.
    C3 = max |autoconvolution| where autoconv is the autoconvolution of f with itself.
    """
    # The autoconvolution of a step function with N intervals on [0,1]
    # can be computed exactly. The autoconvolution is piecewise linear
    # with breakpoints at multiples of 1/N.
    # 
    # (f*f)(t) = sum_{i,j} f_i * f_j * overlap(i,j,t)
    # where overlap(i,j,t) is the length of intersection of [i/N,(i+1)/N) 
    # and [t-j/N, t-(j+1)/N)
    #
    # For a step function, the autoconvolution at t = k/N for k=0,...,2N-1
    # can be computed. But the maximum may occur at non-grid points.
    #
    # Actually, for step functions, the autoconvolution is piecewise linear
    # and its extrema occur at the breakpoints t = k/N.
    #
    # Let's compute autoconv at all breakpoints t = k/N for k = 0, 1, ..., 2N
    # (the support is [0, 2] in the shifted domain where f is on [0,1])
    
    h = 1.0 / N
    f = signed_step.astype(np.float64)
    
    # Use FFT to compute convolution efficiently
    # Pad to length 2N for linear convolution
    n_fft = 2 * N
    F = np.fft.fft(f, n=n_fft)
    autoconv_fft = F * F  # since it's autoconvolution, not cross
    autoconv = np.fft.ifft(autoconv_fft).real
    
    # The autoconv values at grid points t_k = k*h for k=0,...,2N-1
    # But we need to be careful about the scaling.
    # Actually, let's compute directly using the formula.
    
    # More efficient: use the fact that for step functions,
    # autoconv(k*h) = h * sum_i f_i * f_{i-k} for k in range
    # where f_{i-k} is 0 if i-k is out of [0,N-1]
    
    # Let's use FFT properly. The discrete convolution of f with f:
    # (f*f)[k] = sum_j f[j] * f[k-j]
    # This gives us values at t = k*h.
    
    conv = np.fft.ifft(np.fft.fft(f, n=2*N) ** 2).real
    # conv[k] for k=0,...,2N-1 corresponds to t = k*h
    # But we need to scale by h since we're approximating the integral
    autoconv_values = conv * h
    
    # The autoconvolution is piecewise linear between these points.
    # The maximum of |autoconv| occurs at one of the breakpoints.
    c3 = np.max(np.abs(autoconv_values))
    
    return c3

def _compute_c3_exact(signed_step):
    """
    Compute C3 exactly by evaluating autoconvolution at all critical points.
    For a step function with N intervals, the autoconvolution is piecewise
    linear with breakpoints at t = k/N for k = 0, ..., 2N.
    """
    f = signed_step.astype(np.float64)
    h = 1.0 / N
    
    # autoconv(t) for t in [0, 2]
    # autoconv(k*h) = h * sum_{i=max(0,k-N+1)}^{min(N-1,k)} f[i] * f[i-k+N-1]... 
    # Let me think more carefully.
    
    # f(s) = f[i] for s in [i*h, (i+1)*h), i=0,...,N-1
    # (f*f)(t) = integral f(s) f(t-s) ds
    # For t = k*h, this equals h * sum_{i,j} f[i] f[j] * [i*h <= t-j*h < (i+1)*h]
    # = h * sum_{i,j} f[i] f[j] * [j*h <= t-i*h < (j+1)*h]
    # = h * sum_{i,j} f[i] f[j] * [j <= k-i < j+1]
    # = h * sum_i f[i] * f[k-i]  (when 0 <= k-i < N)
    
    # So autoconv(k*h) = h * sum_{i=max(0,k-N+1)}^{min(N-1,k)} f[i] * f[k-i]
    # Wait, that's not quite right. Let me redo.
    
    # (f*f)(t) = integral_0^1 f(s) f(t-s) ds where f is extended by 0 outside [0,1]
    # For t = k*h:
    # = sum_i f[i] * integral_{s in [i*h,(i+1)*h)} f(t-s) ds
    # = sum_i f[i] * sum_j f[j] * length([i*h,(i+1)*h) ∩ [t-j*h, t-(j+1)*h))
    
    # This is getting complex. Let's just use the FFT approach which is exact
    # for the discrete convolution, and the autoconvolution of step functions
    # evaluated at grid points equals h times the discrete autoconvolution.
    
    # Use FFT for O(N log N) computation
    n_fft = 2 * N
    F = np.fft.fft(f, n=n_fft)
    conv = np.fft.ifft(F * F).real
    autoconv_values = conv * h
    
    c3 = np.max(np.abs(autoconv_values))
    return c3

def construct():
    start_time = time.time()
    
    # Strategy: Use a spectral approach. The C3 bound is related to the
    # L-infinity norm of the autoconvolution. We can try to minimize this
    # by choosing signs that make the autoconvolution as flat as possible.
    #
    # Key insight: If we choose f such that its Fourier transform has small
    # magnitude at high frequencies, the autoconvolution will be smoother.
    #
    # Alternative: Use a greedy/local search approach but with a much
    # faster evaluation and a better initial guess.
    
    # Start with a structured initial guess: alternating signs
    # This minimizes the DC component of the autoconvolution
    best_step = np.ones(N, dtype=np.int8)
    best_step[::2] = -1  # alternating pattern
    
    best_c3 = _compute_c3_exact(best_step)
    
    # Local search with multiple flips per iteration
    np.random.seed(42)
    current_temp = 0.1
    cooling_rate = 0.999
    min_temp = 0.001
    max_iterations = 100000
    
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE - 3.0:
            break
        
        # Try flipping 1-3 random positions
        n_flips = np.random.choice([1, 1, 1, 2, 3], p=[0.6, 0.2, 0.1, 0.05, 0.05])
        idxs = np.random.choice(N, size=n_flips, replace=False)
        
        old_vals = best_step[idxs].copy()
        best_step[idxs] = -best_step[idxs]
        
        new_c3 = _compute_c3_exact(best_step)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
        else:
            delta = new_c3 - best_c3
            if delta > 0 and np.random.rand() < np.exp(-delta / max(current_temp, 1e-10)):
                best_c3 = new_c3
            else:
                best_step[idxs] = old_vals
        
        current_temp = max(current_temp * cooling_rate, min_temp)
    
    # Final polish: try single flips until no improvement
    improved = True
    while improved and time.time() - start_time < DEADLINE - 1.0:
        improved = False
        for i in range(N):
            if time.time() - start_time > DEADLINE - 1.0:
                break
            best_step[i] = -best_step[i]
            c3 = _compute_c3_exact(best_step)
            if c3 < best_c3:
                best_c3 = c3
                improved = True
            else:
                best_step[i] = -best_step[i]
    
    return best_step
