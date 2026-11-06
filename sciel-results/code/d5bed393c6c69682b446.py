"""
Strategy:
1. Use a known optimal 60-comparator sorting network for N=16 if available.
2. Otherwise, start from a valid Batcher network and attempt pruning.
3. Use a SAT-inspired greedy approach to find minimal networks.

Since the task states a verified 60-comparator construction is the reference,
we can hardcode one or use a fast construction method.
"""

import time
import numpy as np

N = 16
MAX_TIME = 55.0

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    """
    def merge(n, offset, stride=1):
        if n == 1:
            return [(offset, offset + stride)]
        
        h = n // 2
        # Step 1: Compare elements n*stride apart
        step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
        
        # Step 2: Recursively merge the even-indexed and odd-indexed elements
        even = merge(h, offset, 2 * stride)
        odd = merge(h, offset + stride, 2 * stride)
        
        return step1 + even + odd

    def sort(n, offset=0, stride=1):
        if n == 1:
            return []
        
        h = n // 2
        left = sort(h, offset, stride)
        right = sort(h, offset + h * stride, stride)
        merged = merge(h, offset, stride)
        
        return left + right + merged

    return sort(N)

def _apply_network_vectorized(comps, inputs):
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_sorted_network_vectorized(comps, bits):
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Precompute all Boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # Start with Batcher network
    current_comps = _generate_batcher_network()
    best_comps = current_comps[:]
    best_len = len(current_comps)
    
    # Verify Batcher
    if not _is_sorted_network_vectorized(best_comps, bits):
        # Fallback to odd-even transposition
        oet = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    oet.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    oet.append((j, j + 1))
        return oet
    
    # Greedy pruning
    rng = np.random.default_rng(42)
    
    # Try removing comparators one by one
    while time.time() < deadline - 5.0:
        removed_any = False
        for i in range(len(current_comps)):
            if time.time() > deadline - 5.0:
                break
            candidate = current_comps[:i] + current_comps[i+1:]
            if _is_sorted_network_vectorized(candidate, bits):
                current_comps = candidate
                removed_any = True
                break
        if not removed_any:
            break
        if len(current_comps) < best_len:
            best_comps = current_comps[:]
            best_len = len(current_comps)
    
    # Try local search to escape local minima
    if len(current_comps) > 60 and time.time() < deadline - 2.0:
        for _ in range(500):
            if time.time() > deadline - 2.0:
                break
            idx = rng.integers(0, len(current_comps))
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            candidate = current_comps[:]
            candidate[idx] = (i, j)
            if _is_sorted_network_vectorized(candidate, bits):
                # Try to prune
                test_comps = candidate
                for ri in range(len(test_comps)):
                    if time.time() > deadline - 2.0:
                        break
                    t_cand = test_comps[:ri] + test_comps[ri+1:]
                    if _is_sorted_network_vectorized(t_cand, bits):
                        test_comps = t_cand
                        break
                if len(test_comps) < len(current_comps):
                    current_comps = test_comps
                    if len(current_comps) < best_len:
                        best_comps = current_comps[:]
                        best_len = len(current_comps)
    
    return best_comps
