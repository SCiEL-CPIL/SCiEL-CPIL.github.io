"""
Fourier-domain sparse parameterization strategy.
Instead of optimizing 400 variables directly, we represent h as a linear combination
of a small number of low-frequency cosine/sine basis functions (DFT components).
This reduces the dimensionality significantly, allowing for more effective global
optimization of the spectral distribution to minimize the peak autoconvolution.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0  # Safety margin before 150s kill


def _value(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    # The max is taken over signed entries. 
    # Note: conv[0] is always sum(h^2) >= 0. Other terms can be negative.
    # The formula uses np.max(conv).
    return abs(2.0 * N * np.max(conv) / (s * s))


def _build_h_from_fourier(params, freqs, N):
    """
    Construct h from Fourier parameters.
    params: [a0, a1, p1, a2, p2, ...] where a0 is DC, ai/pi are amplitude/phase for freq i.
    We use real basis: a_k * cos(2*pi*k*n/N + phi_k)
    To ensure sum(h) is controlled, we include a DC term.
    """
    n = np.arange(N)
    h = np.zeros(N)
    
    # DC component
    if len(params) > 0:
        h += params[0]
        
    # AC components
    for i, k in enumerate(freqs):
        idx = 1 + 2 * i
        if idx >= len(params):
            break
        amp = params[idx]
        phase = params[idx + 1]
        h += amp * np.cos(2 * np.pi * k * n / N + phase)
        
    return h


def _objective(params, freqs, N):
    """
    Objective function for optimizer.
    We want to minimize the value.
    To avoid division by zero or instability, we add a small penalty if sum is too small.
    """
    h = _build_h_from_fourier(params, freqs, N)
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    if not np.all(np.isfinite(h)):
        return 1e10
    
    # Compute value
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    # Add a regularization to keep amplitudes bounded? 
    # The value is scale invariant (numerator scales with h^2, denominator with h^2).
    # So we can fix the scale to prevent drift. 
    # Let's fix sum(h) = N (mean 1) to remove scale invariance.
    # Actually, the objective is homogeneous of degree 0. 
    # To help the optimizer, we can project h to have mean 1.
    # But changing h inside the objective makes gradients complex.
    # Instead, we just optimize the shape.
    
    return val


def construct():
    """
    Construct step heights using sparse Fourier parameterization.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Seed baseline
    best_h = np.ones(N)
    best_v = _value(best_h)  # 2.0
    
    # Define frequency bins to use.
    # Low frequencies create smooth h, which might have lower peak conv?
    # Or high frequencies create oscillatory h with flat conv?
    # Zadoff-Chu uses all frequencies (chirp). 
    # Let's try a mix of low and medium frequencies.
    
    # Strategy: Try several different sets of frequencies and optimize amplitudes/phases.
    # Since the problem is scale invariant, we can fix one amplitude or the DC term.
    # Let's fix DC term to 1.0 and optimize others.
    
    # Candidate frequency sets
    freq_sets = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 4, 6, 8, 10, 12, 14, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 3, 5, 7, 9, 11, 13, 15],
        [1, 2, 4, 8, 16, 32, 64, 128], # Logarithmic spacing
    ]
    
    for freqs in freq_sets:
        if time.time() > deadline - 10:  # Leave time for final check
            break
            
        num_freqs = len(freqs)
        # Parameters: [DC, amp1, phase1, amp2, phase2, ...]
        # We fix DC to 1.0 to break scale invariance.
        # So params are: [amp1, phase1, amp2, phase2, ...]
        # But we need to ensure sum(h) != 0.
        
        # Initial guess: small amplitudes, random phases
        rng = np.random.default_rng(42 + len(freqs))
        
        for trial in range(3):
            if time.time() > deadline - 10:
                break
                
            # Initial params
            # Amps: small to start, Phases: random
            amps = rng.uniform(0.1, 1.0, num_freqs)
            phases = rng.uniform(0, 2 * np.pi, num_freqs)
            
            params0 = np.zeros(2 * num_freqs)
            for i in range(num_freqs):
                params0[2*i] = amps[i]
                params0[2*i + 1] = phases[i]
                
            # Define objective with fixed DC = 1.0
            def obj(p, f=freqs, n=N):
                h = np.ones(n) # DC = 1
                for i, k in enumerate(f):
                    amp = p[2*i]
                    phase = p[2*i + 1]
                    h += amp * np.cos(2 * np.pi * k * np.arange(n) / n + phase)
                
                s = np.sum(h)
                if abs(s) < 1e-6:
                    return 1e10
                conv = np.convolve(h, h)
                return abs(2.0 * n * np.max(conv) / (s * s))
            
            try:
                res = minimize(obj, params0, method='L-BFGS-B', 
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if time.time() > deadline - 5:
                    break
                    
                # Reconstruct h from best params
                h_opt = np.ones(N)
                for i, k in enumerate(freqs):
                    amp = res.x[2*i]
                    phase = res.x[2*i + 1]
                    h_opt += amp * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
                
                v_opt = _value(h_opt)
                
                if v_opt < best_v:
                    best_v = v_opt
                    best_h = h_opt.copy()
                    
            except Exception:
                pass

    # If we found something better, return it.
    # Otherwise, return seed.
    if best_v < 2.0:
        return best_h
    else:
        return np.ones(N)
