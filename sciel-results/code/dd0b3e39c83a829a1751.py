import numpy as np
import time

N = 400
DEADLINE = 115.0  # Leave 5s buffer for Python overhead

def _compute_c3_fft(signed_step):
    """
    Fast computation of the C3 upper bound using FFT.
    The autoconvolution of a step function f on [0,1] is:
    (f*f)(t) = integral f(s)f(t-s)ds.
    For step functions with N intervals, the autoconvolution is piecewise linear.
    The maximum of |f*f| occurs at the breakpoints t = k/N.
    The value at t=k/N is h * sum_i f[i]f[k-i] (with appropriate bounds).
    This is exactly h * (f * f)_discrete[k].
    """
    f = signed_step.astype(np.float64)
    h = 1.0 / N
    
    # Use FFT for O(N log N) convolution
    # We need linear convolution of length 2N-1, so pad to next power of 2 or just 2N
    n_fft = 2 * N
    F = np.fft.fft(f, n=n_fft)
    autoconv = np.fft.ifft(F * F).real
    
    # The relevant part of the autoconvolution is from index 0 to 2N-2
    # autoconv[k] corresponds to the integral value at t=k*h
    # We need to scale by h because the discrete sum approximates the integral
    # Actually, for step functions, the convolution integral at grid points
    # is exactly h * sum f[i] f[j] where i+j=k? No, it's cross-correlation-like.
    # Let's verify:
    # f(s) = sum f[i] * 1_{[i/N, (i+1)/N)}(s)
    # (f*f)(t) = sum_i sum_j f[i] f[j] * length( [i/N, (i+1)/N) ∩ [t-j/N, t-(j+1)/N) )
    # If t = k/N, the intersection length is non-zero only if the intervals overlap.
    # For step functions, the autoconvolution is a tent function if f is constant.
    # The discrete convolution sum S[k] = sum_i f[i] f[k-i] (with f=0 outside [0,N-1])
    # represents the coefficient of the piecewise linear function.
    # Specifically, (f*f)(k/N) = h * S[k].
    # Let's check for f=1: S[k] = min(k+1, N, 2N-1-k).
    # (f*f)(k/N) = h * min(k+1, N, 2N-1-k).
    # For k=0: h*1 = 1/N. Integral of 1*1 over [0,1/N] is 1/N. Correct.
    # For k=N-1: h*N = 1. Integral of 1*1 over [0,1] is 1. Correct.
    # So C3 = max |h * S[k]|.
    
    # We only care about k from 0 to 2N-2
    # autoconv from FFT gives S[k] for k=0..2N-2 (and wraps around for larger k if not padded enough)
    # With n_fft=2N, we have indices 0..2N-1. The linear convolution length is 2N-1.
    # So autoconv[0:2N-1] contains S[0..2N-2].
    
    c3 = np.max(np.abs(autoconv[:2*N-1])) * h
    return c3

def construct():
    start_time = time.time()
    
    # 1. Generate initial candidates
    # Barker sequences or pseudo-random starts
    candidates = []
    
    # Barker-like: try to make autocorrelation flat
    # A simple structured start: alternating signs
    alt = np.ones(N, dtype=np.int8)
    alt[::2] = -1
    candidates.append(alt)
    
    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        r = rng.choice([-1, 1], size=N)
        candidates.append(r)
        
    # Evaluate all candidates
    best_step = None
    best_c3 = np.inf
    
    for cand in candidates:
        c3 = _compute_c3_fft(cand)
        if c3 < best_c3:
            best_c3 = c3
            best_step = cand.copy()
            
    # 2. Simulated Annealing
    current_temp = 1e-3  # Start with a small temperature since the landscape is rugged
    cooling_rate = 0.995
    min_temp = 1e-8
    max_iterations = 500000
    
    current_step = best_step
    current_c3 = best_c3
    
    for i in range(max_iterations):
        if time.time() - start_time > DEADLINE:
            break
            
        # Pick a random position to flip
        idx = rng.integers(0, N)
        
        # Flip
        old_val = current_step[idx]
        current_step[idx] = -current_val
        
        new_c3 = _compute_c3_fft(current_step)
        delta = new_c3 - current_c3
        
        if delta < 0 or (np.exp(-delta / max(current_temp, 1e-10)) > rng.random()):
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            current_step[idx] = old_val
            
        current_temp *= cooling_rate
        if current_temp < min_temp:
            current_temp = min_temp

    # 3. Local Search Polish (Steepest Descent)
    # Try single flips until no improvement
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        for i in range(N):
            if time.time() - start_time > DEADLINE:
                break
            best_step[i] = -best_step[i]
            c3 = _compute_c3_fft(best_step)
            if c3 < best_c3:
                best_c3 = c3
                improved = True
            else:
                best_step[i] = -best_step[i]
                
    return best_step
