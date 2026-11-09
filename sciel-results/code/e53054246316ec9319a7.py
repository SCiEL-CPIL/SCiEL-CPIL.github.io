import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_metric(h):
    """Compute the upper bound on C3: max abs autocorrelation at non-zero lags / N."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    # R[0] is the DC term, R[1:] and R[-(N-1):] are non-zero lags
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def _metric_from_params(params):
    """
    Objective function for minimization.
    params: [frequency, phase]
    Returns the metric for the sign of the sinusoid.
    """
    f, phi = params
    # Ensure f is positive and within a reasonable range
    # The frequency is in cycles per N samples.
    # We want to explore different "harmonics".
    # f can be any real number.
    t = np.arange(N)
    # Construct the sinusoid
    signal = np.sin(2 * np.pi * f * t / N + phi)
    # Signed step function: sign of the signal
    # Handle zeros by mapping to 1 (or -1, doesn't matter much for metric usually)
    h = np.sign(signal)
    h[h == 0] = 1.0
    
    return _compute_metric(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_h = None
    best_score = np.inf
    
    # We will run multiple optimizations from different starting points
    # The parameters are [frequency, phase]
    # Frequency range: We can try a wide range. 
    # Low frequency (smooth) vs High frequency (noisy).
    # Usually, autocorrelation minimization favors specific structures.
    # Let's try frequencies from 0.1 to 10.0 cycles per window.
    
    np.random.seed(42)
    
    num_starts = 50
    
    for i in range(num_starts):
        if time.time() > deadline - 5:
            break
            
        # Random start
        f0 = np.random.uniform(0.1, 10.0)
        phi0 = np.random.uniform(0, 2 * np.pi)
        
        # Bounds for optimization
        # We can bound f to [0.1, 20.0] and phi to [0, 2*pi]
        bounds = [(0.1, 20.0), (0, 2 * np.pi)]
        
        try:
            res = minimize(
                _metric_from_params, 
                x0=[f0, phi0], 
                bounds=bounds, 
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            
            if res.success:
                # Extract best parameters
                f_opt, phi_opt = res.x
                t = np.arange(N)
                signal = np.sin(2 * np.pi * f_opt * t / N + phi_opt)
                h_candidate = np.sign(signal)
                h_candidate[h_candidate == 0] = 1.0
                
                score = _compute_metric(h_candidate)
                
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
                    
        except Exception:
            continue
            
    # If we didn't find any valid result (unlikely), fallback to ones
    if best_h is None:
        best_h = np.ones(N, dtype=np.float64)
        best_score = _compute_metric(best_h)
        
    # Optional: Quick discrete polish on the best continuous result
    # This can help if the sinusoid approximation is slightly off
    h_polish = best_h.copy()
    polish_score = best_score
    
    # Simple local search: flip random bits
    for _ in range(100):
        if time.time() > deadline - 2:
            break
        idx = np.random.randint(N)
        old_val = h_polish[idx]
        h_polish[idx] = -old_val
        new_score = _compute_metric(h_polish)
        if new_score < polish_score:
            polish_score = new_score
        else:
            h_polish[idx] = old_val
            
    if polish_score < best_score:
        best_h = h_polish
        best_score = polish_score

    return best_h
