"""Optimized program: sparse Sidon-like support with local search to minimize autoconvolution peak."""
import time
import numpy as np
from itertools import combinations

N = 50

# Deadline guard
_DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _autoconvolution_peak(x):
    """Compute max of autoconvolution of x (length N) using FFT."""
    # Autoconvolution: (x * x)[k] = sum_i x[i] * x[i+k] for appropriate k
    # Full convolution length is 2*N-1
    n_fft = 1
    while n_fft < 2 * N:
        n_fft <<= 1
    X = np.fft.rfft(x, n=n_fft)
    conv = np.fft.irfft(X * X, n=n_fft)
    # The autoconvolution (same) is centered; full conv gives all shifts
    # Max over all lags
    return np.max(np.abs(conv[:2*N-1]))

def _score(x):
    """Compute L2^2 / (L1 * sup_autoconv)."""
    x = np.asarray(x, dtype=np.float64)
    l2_sq = np.sum(x * x)
    l1 = np.sum(x)
    if l1 < 1e-15:
        return 0.0
    peak = _autoconvolution_peak(x)
    if peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def _sidon_set(n, size):
    """Find a Sidon (B2) subset of {0,...,n-1} of given size using greedy."""
    # Greedy: pick indices such that all pairwise differences are distinct
    chosen = []
    diffs = set()
    for i in range(n):
        ok = True
        for j in chosen:
            d = abs(i - j)
            if d in diffs:
                ok = False
                break
        if ok:
            chosen.append(i)
            for j in chosen[:-1]:
                diffs.add(abs(i - j))
        if len(chosen) >= size:
            break
    return chosen

def _local_search(x, max_iters=5000):
    """Local search: swap elements or adjust support to improve score."""
    x = np.array(x, dtype=np.float64)
    best_score = _score(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        improved = False
        # Try moving one unit of mass from index i to index j
        # Or try swapping values
        # Simpler: try all pairs (i,j) where x[i] > 0 and try moving some mass
        # Even simpler: try permuting the support set
        
        # Try: pick two indices, swap their values
        # Try: move all mass from one index to another
        for i in range(N):
            if x[i] <= 1e-12:
                continue
            for j in range(N):
                if i == j:
                    continue
                # Move all mass from i to j
                trial = best_x.copy()
                trial[j] += trial[i]
                trial[i] = 0.0
                s = _score(trial)
                if s > best_score + 1e-9:
                    best_x = trial
                    best_score = s
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try adding mass to a zero index (if we have budget)
            # Actually, we want to keep total mass fixed for L1 normalization
            # Let's try splitting: move half from i to j
            for i in range(N):
                if x[i] <= 1e-12:
                    continue
                for j in range(N):
                    if i == j:
                        continue
                    trial = best_x.copy()
                    trial[j] += trial[i] * 0.5
                    trial[i] *= 0.5
                    s = _score(trial)
                    if s > best_score + 1e-9:
                        best_x = trial
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    return best_x, best_score

def construct():
    start_time = time.time()
    
    # Strategy: Use sparse support with Sidon-like structure
    # The key insight: if support is a Sidon set, the autoconvolution has minimal collisions
    # For a set S of size k, the number of pairs (a,b) in S×S with a-b = d is at most 1 for each d
    # This minimizes the peak of the autoconvolution
    
    # Start with a Sidon set of size ~10-15
    # Normalize so that L1 = 1 (or any constant, ratio is scale-invariant in L1)
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Try various Sidon set sizes
    for k in [8, 10, 12, 15, 20, 25]:
        if time.time() > _DEADLINE - 5:
            break
        sidon = _sidon_set(N, k)
        if len(sidon) < k:
            continue
        # Uniform mass on Sidon set
        x = np.zeros(N)
        for idx in sidon:
            x[idx] = 1.0 / k
        s = _score(x)
        if s > best_score:
            best_score = s
            best_x = x.copy()
        
        # Local search on this support
        if time.time() < _DEADLINE - 15:
            lx, ls = _local_search(x, max_iters=2000)
            if ls > best_score:
                best_score = ls
                best_x = lx
    
    # Also try random sparse supports
    np.random.seed(42)
    for trial in range(50):
        if time.time() > _DEADLINE - 5:
            break
        k = np.random.randint(8, 20)
        support = np.random.choice(N, size=k, replace=False)
        x = np.zeros(N)
        x[support] = 1.0 / k
        s = _score(x)
        if s > best_score:
            best_score = s
            best_x = x.copy()
    
    # Final local search on best found
    if time.time() < _DEADLINE - 10:
        lx, ls = _local_search(best_x, max_iters=5000)
        if ls > best_score:
            best_score = ls
            best_x = lx
    
    # Return as list of floats
    return [float(v) for v in best_x]
