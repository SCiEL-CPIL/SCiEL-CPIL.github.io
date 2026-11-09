import numpy as np
from scipy.optimize import minimize
import time

N = 600
DX = 1.0 / 1200.0  # interval width

def autoconvolution_max(h):
    """Compute the maximum of the autoconvolution of step function h."""
    # h has N values, each on an interval of width DX
    # Autoconvolution: (f*f)(t) for step function
    # The autoconvolution of two step functions is a piecewise linear function
    # with 2N-1 segments (or 2N-1 peaks in the discrete sense)
    
    # Using FFT for efficiency
    n_fft = 2 * N
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    conv = np.fft.irfft(np.fft.rfft(h_padded) ** 2) * DX
    # The autoconvolution values at the "peak" positions
    # For a step function, the autoconvolution at t=0 is sum(h_i^2) * DX
    # But we need the maximum over all t
    # The autoconvolution is maximum at t=0 for nonnegative symmetric functions
    # But for general nonnegative functions, we should check all positions
    
    # Actually, for nonnegative f, (f*f)(t) is maximized at t=0 if f is symmetric
    # For general case, let's compute the max of the full autoconvolution
    # The autoconvolution has 2N-1 "corners" but we can sample it
    
    # More carefully: the autoconvolution of step functions with N steps each
    # gives a piecewise linear function with breakpoints at multiples of DX
    # from -(N-1)*DX to (N-1)*DX, so 2N-1 intervals
    # The value at each breakpoint can be computed
    
    # Let's use the fact that for step functions, the autoconvolution at 
    # t = k*DX (for integer k) is:
    # sum_i h[i] * h[j] where the overlap of interval i and interval (i-k) 
    # is DX (full overlap) or 0
    
    # Actually, let me think more carefully. If f(x) = h[i] for x in [i*DX, (i+1)*DX)
    # then (f*f)(t) = integral f(tau) f(t-tau) dtau
    # For t = m*DX, the integral breaks into pieces where both f(tau) and f(t-tau) 
    # are constant
    
    # For simplicity and correctness, let's use FFT-based computation
    # The autoconvolution sampled at t = 0, DX, 2DX, ..., (2N-2)*DX - (N-1)*DX
    # Actually, let's just compute it properly
    
    # The linear autoconvolution of h with h (length N each) gives length 2N-1
    # conv[k] = sum_i h[i] * h[k-i] for valid i, k-i
    # This is the value at t = (k - (N-1))*DX approximately
    
    conv_linear = np.convolve(h, h, mode='full') * DX
    # conv_linear has 2N-1 elements
    # The maximum of the autoconvolution is max(conv_linear)
    # But wait, this is the value at discrete points. The actual autoconvolution
    # is piecewise linear, and its maximum could be at a non-integer multiple of DX
    
    # For nonnegative step functions, the autoconvolution is unimodal (if h is 
    # unimodal) and the maximum is at t=0, which corresponds to conv_linear[N-1]
    # But to be safe, let's take the max
    
    return np.max(conv_linear)

def normalized_max_autoconv(h):
    """Compute the normalized maximum of the autoconvolution."""
    total = np.sum(h) * DX
    if total == 0:
        return 0.0
    max_conv = autoconvolution_max(h)
    return max_conv / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # Initial guess: a smooth bell-shaped profile
    x = np.linspace(-0.25, 0.25, N)
    h0 = np.exp(-10 * x**2)
    h0 = np.maximum(h0, 0.0)
    
    # Normalize so that sum(h) * DX = 1 (unit area)
    area = np.sum(h0) * DX
    if area > 0:
        h0 = h0 / area
    
    best_h = h0.copy()
    best_score = normalized_max_autoconv(h0)
    
    # Use scipy minimize with BFGS
    def objective(h_flat):
        h = np.maximum(h_flat, 0.0)
        return normalized_max_autoconv(h)
    
    # Try BFGS optimization
    try:
        result = minimize(objective, h0, method='BFGS', 
                         options={'maxiter': 500, 'gtol': 1e-8})
        if result.fun < best_score:
            best_h = np.maximum(result.x, 0.0)
            best_score = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return list(best_h)
    
    # Try coordinate descent as a refinement
    h = best_h.copy()
    for iteration in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = h[i]
            # Try a few perturbation values
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                new_val = max(0.0, old_val + delta)
                h[i] = new_val
                score = normalized_max_autoconv(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
                    improved = True
                else:
                    h[i] = old_val
                    break
            if not improved:
                # Try smaller steps
                for delta in [0.001, -0.001, 0.005, -0.005]:
                    new_val = max(0.0, h[i] + delta)
                    h[i] = new_val
                    score = normalized_max_autoconv(h)
                    if score < best_score:
                        best_score = score
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old_val
                        break
        if not improved:
            break
    
    return list(best_h)
