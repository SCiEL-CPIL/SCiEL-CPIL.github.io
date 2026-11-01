"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_c3_bound(h):
    """
    Compute the C3 autoconvolution upper bound.
    For a ±1 sequence, this is based on the third moment of the autocorrelation.
    """
    # Compute linear autocorrelation using FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    
    # acf[0] = N (energy), acf[1:] are sidelobes
    # C3 bound: (1/N) * sum_{k=1}^{N-1} |acf[k]|^3 / N^2
    # Or possibly just the sum of cubes normalized
    sidelobes = acf[1:]
    c3 = np.sum(np.abs(sidelobes) ** 3) / (N ** 3)
    return c3

def _compute_max_sidelobe(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return np.max(np.abs(acf[1:]))

def _local_search_c3(h, deadline):
    """Greedy local search: flip bits to reduce C3 bound."""
    best_h = h.copy()
    best_val = _compute_c3_bound(best_h)
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            h[i] = -h[i]
            val = _compute_c3_bound(h)
            h[i] = -h[i]  # revert
            
            if val < best_val:
                h[i] = -h[i]  # keep flip
                best_val = val
                best_h = h.copy()
                improved = True
        
        if not improved:
            break
    
    return best_h, best_val

def _smoothed_objective(x, T=5.0, eps=1e-8):
    """
    Smoothed objective for continuous relaxation.
    x is a vector of length N, h = tanh(x) is in (-1, 1).
    We minimize a soft-max approximation of the max sidelobe normalized by L1 norm squared.
    """
    h = np.tanh(x)
    
    # Compute autocorrelation
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    
    sidelobes = acf[1:]
    abs_sidelobes = np.sqrt(sidelobes**2 + eps)
    
    # Soft max
    max_abs = np.max(abs_sidelobes)
    log_sum_exp = np.log(np.sum(np.exp((abs_sidelobes - max_abs) / T))) + max_abs / T
    soft_max = T * log_sum_exp
    
    # Denominator: (sum |h_i|)^2
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    value = soft_max / (sum_abs_h ** 2)
    
    return value

def _construct_continuous(start_time, deadline):
    """
    Perform continuous relaxation optimization.
    Returns best h found and its value.
    """
    best_h = None
    best_val = float('inf')
    
    # Try a few random initializations in the continuous space
    rng = np.random.default_rng(123)
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialize x such that h = tanh(x) is near ±1
        # Start with random ±1 signs, scaled to be near the boundary
        signs = rng.choice([-1.0, 1.0], size=N)
        x0 = signs * 5.0  # tanh(5) ~ 0.9999
        
        try:
            res = minimize(
                _smoothed_objective, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            h_opt = np.tanh(res.x)
            val = _compute_c3_bound(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception as e:
            continue
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Phase 1: Continuous Relaxation
    best_h = None
    best_val = float('inf')
    
    cont_h, cont_val = _construct_continuous(start_time, deadline)
    if cont_h is not None and cont_val < best_val:
        best_h = cont_h
        best_val = cont_val
    
    # If continuous relaxation didn't find anything, or to have a fallback,
    # try random discrete sequences
    if best_h is None or time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for restart in range(10):
            if time.time() > deadline - 20:
                break
            
            h = rng.choice([-1.0, 1.0], size=N)
            val = _compute_c3_bound(h)
            if val < best_val:
                best_h = h.copy()
                best_val = val
        
        # If we still don't have a best_h (unlikely), initialize with ones
        if best_h is None:
            best_h = np.ones(N)
            best_val = _compute_c3_bound(best_h)
    
    # Phase 2: Discrete Local Search
    # Round the continuous solution to ±1 if it exists, or use the discrete one
    if best_h is not None:
        # If we came from continuous, round to nearest ±1
        if np.any((best_h > -0.5) & (best_h < 0.5)):
            # This heuristic check isn't perfect, but if values are close to 0, it was likely a continuous interior point
            # However, tanh outputs are always in (-1,1). 
            # A better check: if the variance is high, it's likely discrete-like.
            # Let's just round the current best_h to ±1 to start the discrete search
            discrete_h = np.where(best_h >= 0, 1.0, -1.0)
            discrete_val = _compute_c3_bound(discrete_h)
            
            if discrete_val < best_val:
                best_h = discrete_h
                best_val = discrete_val
            else:
                # Keep the rounded version for local search even if value is slightly higher,
                # as local search might find a better discrete neighbor
                best_h = discrete_h
                best_val = discrete_val
        
        # Run local search on the discrete candidate
        if time.time() < deadline - 5:
            result_h, result_val = _local_search_c3(best_h, deadline)
            if result_val < best_val:
                best_h = result_h
                best_val = result_val
    
    return best_h
# EVOLVE-BLOCK-END
