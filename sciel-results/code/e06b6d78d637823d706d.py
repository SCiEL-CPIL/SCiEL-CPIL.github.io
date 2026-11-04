"""
Discrete Combinatorial Search for Signed Step Functions.
Strategy:
1. Generate candidate sequences from Zadoff-Chu (ZC) families and random polyphase sequences.
2. Use a fast bit-flip / level-flip local search to minimize the maximum absolute autoconvolution.
3. Exploit the fact that ZC sequences have near-perfect periodic autocorrelation, which often implies low linear autocorrelation sidelobes (especially for prime lengths close to N).
"""

import numpy as np
import time
from scipy.fft import rfft, irfft

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Compute the maximum absolute value of the linear autoconvolution."""
    fx = rfft(x, _FFT_SIZE)
    conv = irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    """
    Score to minimize: max |autoconv| / (sum x)^2.
    """
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _generate_zadoff_chu_real(n, u, v=1):
    """
    Generate a real-valued sequence inspired by Zadoff-Chu.
    Standard ZC are complex. For real sequences, we can use:
    x[n] = cos(2*pi*(u*n^2 + v*n)/p) or similar polyphase structures.
    However, true ZC is complex. For real, we can use:
    1. Real part of ZC
    2. Cosine chirps: x[n] = cos(2*pi * (a*n^2 + b*n) / N)
    
    Let's use a generalized cosine chirp which is real and has good autocorrelation properties.
    """
    n = np.arange(n)
    # Normalize frequency to be within [0, 1] range for the quadratic term
    # We want to explore different 'chirp rates'
    a = u / n
    b = v / n
    # Avoid division by zero if n=0, but n starts at 0.
    # Actually, ZC uses n^2 mod p. For continuous real approximation:
    # x[n] = cos(2*pi * (a * n^2 + b * n))
    # We want the phase to wrap around nicely.
    # Let's just use a grid of (a, b) parameters.
    phase = 2 * np.pi * (a * n**2 + b * n)
    return np.cos(phase)

def _generate_random_polyphase(n, levels, rng):
    """Generate a random sequence with discrete levels."""
    vals = rng.integers(0, len(levels), size=n)
    return levels[vals].astype(float)

def _local_search_discrete(x, deadline, rng, levels):
    """
    Local search by flipping individual elements to the best alternative level.
    This is a greedy coordinate descent.
    """
    x = x.copy()
    best_val = _metric(x)
    n_levels = len(levels)
    
    # Precompute indices
    indices = np.arange(N)
    
    while time.time() < deadline:
        improved = False
        # Shuffle order to avoid bias
        order = rng.permutation(N)
        
        for idx in order:
            if time.time() > deadline:
                return x, best_val
            
            current_val = x[idx]
            best_new_val = current_val
            best_local_metric = best_val
            
            for lv in levels:
                if lv == current_val:
                    continue
                x[idx] = lv
                m = _metric(x)
                if m < best_local_metric:
                    best_local_metric = m
                    best_new_val = lv
            
            if best_new_val != current_val:
                x[idx] = best_new_val
                best_val = best_local_metric
                improved = True
        
        if not improved:
            break
            
    return x, best_val

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(2024)
    
    best_x = None
    best_val = float('inf')
    
    # Define candidate levels
    # Binary {-1, 1}
    levels_bin = np.array([-1.0, 1.0])
    # Ternary {-1, 0, 1}
    levels_ter = np.array([-1.0, 0.0, 1.0])
    # 5-level
    levels_5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # 1. Try Cosine Chirps (Continuous init, then discretize? Or just use as is?)
    # The task asks for "signed step heights". It doesn't strictly forbid continuous values,
    # but "signed step functions" often implies discrete steps.
    # However, the metric is invariant to scaling. 
    # If we allow continuous values, we can just use the cosine chirp directly.
    # But the "discrete" strategy suggests exploring the combinatorial space.
    # Let's try both: 
    # A) Direct continuous chirps
    # B) Discrete sequences (flipped to {-1, 1} or similar)
    
    # --- Strategy A: Continuous Chirps ---
    # Generate a set of chirps
    chirps = []
    for a in np.linspace(0.01, 0.5, 10):
        for b in np.linspace(0.0, 0.5, 5):
            x = _generate_zadoff_chu_real(N, a, b)
            # Normalize to have sum 1 to avoid division by small numbers?
            # Metric is scale invariant, so we can just evaluate as is.
            # But if sum is close to 0, metric is huge.
            # Chirps usually have sum close to 0 if they oscillate.
            # We need max |conv| / sum^2.
            # If sum is small, we are in trouble.
            # We can shift the chirp? No, shifting changes DC.
            # Let's just evaluate.
            chirps.append(x)
            
    # --- Strategy B: Discrete Sequences ---
    # 1. Random Binary
    for _ in range(5):
        x = _generate_random_polyphase(N, levels_bin, rng)
        chirps.append(x)
        
    # 2. Random Ternary
    for _ in range(5):
        x = _generate_random_polyphase(N, levels_ter, rng)
        chirps.append(x)
        
    # 3. Structured Binary: Alternating, Blocks
    x_alt = np.ones(N)
    x_alt[::2] = -1.0
    chirps.append(x_alt)
    
    x_block = np.ones(N)
    x_block[N//2:] = -1.0
    chirps.append(x_block)

    # Evaluate all initial candidates
    candidates = []
    for x in chirps:
        # Check if sum is too small
        s = _integral(x)
        if abs(s) < 1e-3:
            continue
        val = _metric(x)
        candidates.append((val, x))
        
    if not candidates:
        # Fallback: all ones
        x = np.ones(N)
        return x

    # Sort by metric
    candidates.sort(key=lambda t: t[0])
    
    # Keep top 10 for local search
    top_candidates = [c[1] for c in candidates[:10]]
    
    # Perform local search on each top candidate
    # Allocate time budget
    time_per_candidate = 10.0
    
    for x_init in top_candidates:
        if time.time() > deadline - 5.0:
            break
            
        cand_deadline = min(deadline - 2.0, time.time() + time_per_candidate)
        
        # Determine levels for this candidate
        # If it looks binary, use binary levels. Else, use continuous?
        # Local search on continuous space is harder (gradient based).
        # Our local search is discrete.
        # So we should project the candidate to the nearest discrete levels?
        # Or just run discrete search on a discrete version.
        
        # Let's create a discrete version of the candidate
        # If the candidate is continuous (chirp), we can threshold it to {-1, 1}
        # Or just use the discrete candidates we generated.
        
        # For chirps, let's just evaluate them directly without discrete local search,
        # because flipping a continuous value to -1 or 1 might destroy the structure.
        # Instead, for chirps, we can do a simple gradient step?
        # No, let's stick to the discrete strategy for the "different approach".
        # The chirps are just good initializations for the discrete search if we threshold them.
        
        x_disc = np.sign(x_init)
        # Ensure it's in levels_bin
        x_disc = np.where(x_disc > 0, 1.0, -1.0)
        
        x_opt, val_opt = _local_search_discrete(x_disc, cand_deadline, rng, levels_bin)
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
            
        # Also try the original continuous candidate if it was good?
        # The task allows "signed step heights". Continuous values are step heights too.
        # But the discrete search is the "new strategy".
        # Let's also keep the best continuous candidate found in the initial pool.
        
    # Also, we should probably try a few continuous local searches if we have time?
    # The prompt asks for a DIFFERENT strategy. Discrete search is different.
    # But if the continuous chirps are actually better, we should return them.
    
    # Let's re-evaluate the best continuous candidate from the initial pool
    if candidates:
        best_cont_val, best_cont_x = candidates[0]
        if best_cont_val < best_val:
            best_val = best_cont_val
            best_x = best_cont_x

    # If we still have time, try a few more random discrete restarts
    while time.time() < deadline - 5.0:
        x_rand = _generate_random_polyphase(N, levels_bin, rng)
        x_opt, val_opt = _local_search_discrete(x_rand, min(deadline - 2.0, time.time() + 5.0), rng, levels_bin)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt

    if best_x is None:
        best_x = np.ones(N)
        
    return best_x
