"""Rewritten program for the third autocorrelation inequality task."""
import time
import math
import random
import numpy as np

N = 400

# EVOLVE-BLOCK-START

def _c3_bound(s):
    """Upper bound on C3: max |third-order autocorrelation| via FFT."""
    n = len(s)
    # Need convolution of three length-n sequences -> length 3n-2
    nfft = 1 << (3 * n - 1).bit_length()
    S = np.fft.fft(s, n=nfft)
    C3 = np.fft.ifft(S ** 3).real
    return float(np.max(np.abs(C3)))


def _c3_bound_delta(s, idx, old_val):
    """Compute C3 bound after flipping s[idx] from old_val to -old_val, using delta method."""
    n = len(s)
    nfft = 1 << (3 * n - 1).bit_length()
    S = np.fft.fft(s, n=nfft)
    delta = -2.0 * old_val
    S_new = S.copy()
    S_new[idx] += delta
    # (S+delta)^3 = S^3 + 3 S^2 delta + 3 S delta^2 + delta^3
    C3_new = np.fft.ifft(S ** 3 + 3 * S ** 2 * delta + 3 * S * delta ** 2 + delta ** 3).real
    return float(np.max(np.abs(C3_new)))


def _generate_chaotic_signs(seed_val, n=N):
    """Generate sign sequence from logistic map at r=4."""
    x = seed_val
    signs = np.empty(n)
    for i in range(n):
        x = 4.0 * x * (1.0 - x)
        signs[i] = 1.0 if x > 0.5 else -1.0
    return signs


def _greedy_polish(signs, time_limit):
    """Greedy single-flip polish until no improvement or time limit."""
    current = _c3_bound(signs)
    improved = True
    start = time.time()
    while improved and (time.time() - start) < time_limit:
        improved = False
        for i in range(N):
            old_val = signs[i]
            signs[i] = -old_val
            new_val = _c3_bound_delta(signs, i, -old_val)  # s[i] is now -old_val, delta from -old_val to old_val
            # Actually: we flipped signs[i] from old_val to -old_val.
            # _c3_bound_delta expects: s is the array AFTER flip, idx, old_val is the value BEFORE flip.
            # So old_val here should be old_val (the value before this flip).
            # Let me recompute properly.
            signs[i] = old_val  # revert temporarily to compute correctly
            # Flip: signs[i] goes from old_val to -old_val
            # S_new = S + (-2*old_val) at index i
            S = np.fft.fft(signs, n=1 << (3 * N - 1).bit_length())
            delta = -2.0 * old_val
            S_new = S.copy()
            S_new[i] += delta
            C3_new = np.fft.ifft(S ** 3 + 3 * S ** 2 * delta + 3 * S * delta ** 2 + delta ** 3).real
            new_bound = float(np.max(np.abs(C3_new)))
            if new_bound < current - 1e-12:
                signs[i] = -old_val
                current = new_bound
                improved = True
            # else keep signs[i] = old_val
    return signs, current


def _simulated_annealing(signs, time_limit, seed=42):
    """Simulated annealing with delta-method C3 evaluation."""
    rng = random.Random(seed)
    current = _c3_bound(signs)
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    start = time.time()
    
    nfft = 1 << (3 * N - 1).bit_length()
    
    while T > T_min and (time.time() - start) < time_limit:
        i = rng.randint(0, N - 1)
        old_val = signs[i]
        signs[i] = -old_val
        
        # Compute new C3 bound using delta method
        S = np.fft.fft(signs, n=nfft)
        # We need S_before, not S_after. Let's compute S_before by undoing the flip in frequency domain.
        # Actually easier: compute S_before = S_after - delta where delta = -2*old_val
        # S_after = S_before + delta, so S_before = S_after - delta
        delta = -2.0 * old_val
        S_before = S - delta  # this is the FFT of the array BEFORE the flip
        
        C3_before = np.fft.ifft(S_before ** 3).real
        C3_after = np.fft.ifft(S ** 3).real
        
        new_bound = float(np.max(np.abs(C3_after)))
        delta_bound = new_bound - current
        
        if delta_bound < 0 or rng.random() < math.exp(-delta_bound / T):
            current = new_bound
        else:
            signs[i] = old_val
        
        T *= alpha
    
    return signs, current


def _multi_start_local_search(time_limit):
    """Multi-start: generate many chaotic candidates, pick best, then polish."""
    start = time.time()
    remaining = time_limit - 5  # leave 5s for final polish
    
    # Generate diverse chaotic seeds
    seeds = []
    # Irrational-like seeds
    for k in range(1, 30):
        seeds.append((math.sqrt(k) % 1.0))
    seeds.append(math.pi % 1.0)
    seeds.append(math.e % 1.0)
    seeds.append((1 + math.sqrt(5)) / 2 % 1.0)
    # More random-ish seeds
    rng = random.Random(12345)
    for _ in range(50):
        seeds.append(rng.random())
    
    best_signs = None
    best_bound = float('inf')
    
    # Evaluate all seeds
    for s in seeds:
        if (time.time() - start) > remaining * 0.3:
            break
        signs = _generate_chaotic_signs(s)
        bound = _c3_bound(signs)
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()
    
    if best_signs is None:
        best_signs = _generate_chaotic_signs(0.3)
        best_bound = _c3_bound(best_signs)
    
    # Simulated annealing on best
    elapsed = time.time() - start
    sa_time = max(1.0, remaining - elapsed - 3)
    best_signs, best_bound = _simulated_annealing(best_signs, sa_time, seed=42)
    
    # Greedy polish
    elapsed = time.time() - start
    polish_time = max(0.5, remaining - elapsed)
    best_signs, best_bound = _greedy_polish(best_signs, polish_time)
    
    return best_signs


def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    Uses chaotic map generation with multi-start, simulated annealing, and greedy polish
    to minimize the upper bound on C3 (max absolute value of third-order autocorrelation).
    """
    total_time_limit = 110  # seconds, leaving buffer for overhead
    
    signs = _multi_start_local_search(total_time_limit)
    
    # Ensure output is exactly ±1
    signs = np.where(signs > 0, 1.0, -1.0)
    
    return signs

# EVOLVE-BLOCK-END
