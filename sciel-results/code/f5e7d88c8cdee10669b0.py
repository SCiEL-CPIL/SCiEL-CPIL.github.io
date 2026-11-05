"""
Strategy: Frequency-Domain Parameterization with Low-Dimensional Optimization.

Instead of optimizing 400 variables (h[0]...h[399]) directly, we represent the
step function h as a sum of a constant offset and a few low-frequency Fourier
modes.
    h[k] = offset + sum_{i=1}^{M} (a_i * cos(2*pi*i*k/N) + b_i * sin(2*pi*i*k/N))

This reduces the optimization space from 400 dimensions to 2*M + 1 dimensions.
We use FFT to compute the auto-correlation efficiently.
The objective is to minimize V = abs(2 * N * max(conv(h,h)) / sum(h)^2).

Since the problem is scale-invariant (scaling h by c scales conv by c^2 and sum^2 by c^2, so V is invariant),
we can fix the sum of h to a specific value (e.g., 1) or just let the optimizer handle it.
However, fixing the sum simplifies the objective to minimizing max(conv(h,h)) subject to sum(h)=1.
Actually, the formula is:
V = | 2 * N * max(conv) / (sum(h))^2 |

If we enforce sum(h) = 1, then V = 2 * N * max(conv).
We minimize this.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400
_DEADLINE = 145.0  # Safety margin for 150s limit

def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    # Use fftconvolve for speed if possible, but np.convolve is fine for N=400
    # np.convolve is O(N^2) = 160k ops, which is very fast.
    # fftconvolve is O(N log N) ~ 3000 ops.
    conv = fftconvolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _parametric_h(params, M):
    """
    Reconstruct h from Fourier parameters.
    params = [offset, a1, b1, a2, b2, ..., aM, bM]
    """
    n = np.arange(N)
    h = np.full(N, params[0])
    for i in range(1, M + 1):
        a = params[2*i - 1]
        b = params[2*i]
        h += a * np.cos(2 * np.pi * i * n / N)
        h += b * np.sin(2 * np.pi * i * n / N)
    return h

def _objective(params, M):
    """
    Objective function for the optimizer.
    We want to minimize V.
    To handle scale invariance and numerical stability, we can normalize h internally
    or just let the optimizer explore the scale. 
    However, Nelder-Mead works well with bounded variables. 
    Let's normalize the output h to have sum=1 before evaluating V, 
    effectively optimizing the shape.
    Wait, if I normalize sum(h)=1, V = 2*N*max(conv).
    If I don't normalize, V is scale invariant.
    Let's just compute V directly. It's robust enough.
    """
    h = _parametric_h(params, M)
    if not np.all(np.isfinite(h)):
        return 1e10
    return _evaluate(h)

def construct():
    t0 = time.time()
    
    # Choose M, the number of frequency modes.
    # More modes allow more complex shapes but increase dimensionality.
    # M=20 means 41 variables. Nelder-Mead is efficient in ~100-200 dims.
    M = 20
    dim = 2 * M + 1
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Generate diverse initial candidates in the parameter space
    rng = np.random.default_rng(42)
    
    # Candidate 1: All zeros (h=offset). Offset=1 gives seed.
    # If params are all 0, h=0, sum=0 -> inf. 
    # So we need a non-zero offset.
    # Let's initialize offset=1.0 and random small frequencies.
    
    candidates = []
    
    # 1. Seed-like: offset=1, others 0
    p_seed = np.zeros(dim)
    p_seed[0] = 1.0
    candidates.append(p_seed)
    
    # 2. Random low-frequency perturbations
    for _ in range(20):
        p = np.zeros(dim)
        p[0] = 1.0 # Base offset
        # Random amplitudes for modes
        # Scale amplitudes to be comparable to offset
        for i in range(1, M+1):
            # Lower frequencies should have larger amplitudes potentially
            decay = 1.0 / (1.0 + 0.2 * i)
            p[2*i - 1] = rng.normal(0, 0.5 * decay)
            p[2*i] = rng.normal(0, 0.5 * decay)
        candidates.append(p)
        
    # 3. Structured patterns: e.g., pure cosine modes
    for i in range(1, min(10, M+1)):
        p = np.zeros(dim)
        p[0] = 1.0
        p[2*i - 1] = 0.5 # Strong cosine component
        candidates.append(p)
        
    # 4. Random "chirp" or phase-shifted patterns
    for _ in range(10):
        p = np.zeros(dim)
        p[0] = 1.0
        # Create a smooth random signal in frequency domain
        # Low pass filter effect
        freqs = np.arange(1, M+1)
        amps = np.exp(-freqs / 10.0) * rng.uniform(0.5, 1.0)
        phases = rng.uniform(0, 2*np.pi, size=M)
        for i in range(1, M+1):
            idx_a = 2*i - 1
            idx_b = 2*i
            p[idx_a] = amps[i-1] * np.cos(phases[i-1])
            p[idx_b] = amps[i-1] * np.sin(phases[i-1])
        candidates.append(p)

    # Optimize each candidate using Nelder-Mead
    # Nelder-Mead is derivative-free and robust for non-smooth functions.
    # We use maxiter to control time.
    
    for p_init in candidates:
        if time.time() > _DEADLINE - 5:
            break
            
        try:
            res = minimize(
                _objective,
                p_init,
                args=(M,),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if res.success or res.fun < best_val:
                p_opt = res.x
                h_opt = _parametric_h(p_opt, M)
                
                # Check validity
                if np.all(np.isfinite(h_opt)) and np.sum(h_opt) != 0:
                    v_opt = _evaluate(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        
        except Exception:
            continue

    # Final refinement: If we have time, do a local greedy search on the best h found
    # This helps with the "sharp" peaks that Fourier modes might miss.
    if time.time() < _DEADLINE - 10:
        best_h, best_val = _local_refine(best_h, best_val, deadline=_DEADLINE - 2)
        
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

def _local_refine(h, best_val, deadline):
    """
    Local greedy refinement on the full 400-dim vector.
    """
    best_h = h.copy()
    rng = np.random.default_rng(99)
    step_size = 0.05
    
    # Sort indices by magnitude of h? No, just random.
    # To be faster, we can try perturbing groups of indices.
    
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        # Pick a small number of indices
        n_perturb = rng.integers(1, 5)
        idx = rng.choice(N, size=n_perturb, replace=False)
        
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, step_size, n_perturb)
        
        v_new = _evaluate(h_new)
        
        if np.isfinite(v_new) and v_new < best_val:
            best_val = v_new
            best_h = h_new
            step_size *= 0.9
        else:
            step_size *= 1.05
            if step_size > 0.5:
                step_size = 0.5
                
    return best_h, best_val
