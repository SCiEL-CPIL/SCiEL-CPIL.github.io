"""
Strategy: Simulated annealing with add/remove/swap moves on a comparator list.
- Seed: Batcher's Odd-Even Merge Sort (63 comparators).
- Moves: remove a random comparator, add a random comparator, swap two comparators.
- Validation: vectorized zero-one principle over all 2^16 inputs.
- Accept: always if better (fewer comparators); sometimes if worse (annealing schedule).
- Time guard: return best found before deadline.
"""

import time
import random
import numpy as np

N = 16
MAX_TIME = 85.0  # Safety margin for 90s wall-clock limit

def _batchers_network():
    """Batcher's Odd-Even Merge Sort for N=16."""
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        h = n // 2
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            h = n // 2
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            return step1 + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)

def _apply_network(comps, bits):
    """Apply comparator network to pre-computed boolean input matrix (65536 x 16)."""
    vec = bits.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_valid(comps, bits):
    """Check zero-one principle: all outputs sorted."""
    out = _apply_network(comps, bits)
    diffs = np.diff(out, axis=1)
    return bool(np.all(diffs >= 0))

def _make_move(comps, rng):
    """
    Generate a perturbed network via one of:
      - remove a random comparator
      - add a random comparator
      - swap two comparators
    Returns the new list (a copy).
    """
    n = len(comps)
    move_type = rng.random()
    
    if move_type < 0.35 and n > 1:
        # Remove a random comparator
        idx = rng.randrange(n)
        return comps[:idx] + comps[idx+1:]
    elif move_type < 0.70:
        # Add a random comparator
        i = rng.randrange(N)
        j = rng.randrange(i + 1, N)
        new_comps = comps[:]
        pos = rng.randrange(len(new_comps) + 1)
        new_comps.insert(pos, (i, j))
        return new_comps
    else:
        # Swap two comparators
        if n < 2:
            # Can't swap; fall back to add
            i = rng.randrange(N)
            j = rng.randrange(i + 1, N)
            new_comps = comps[:]
            pos = rng.randrange(len(new_comps) + 1)
            new_comps.insert(pos, (i, j))
            return new_comps
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            j = (j + 1) % n
        new_comps = comps[:]
        new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
        return new_comps

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    rng = random.Random(42)
    
    # Pre-compute all 2^16 boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # Initial network: Batcher's
    comps = _batchers_network()
    
    # Verify
    if not _is_valid(comps, bits):
        # Fallback: odd-even transposition
        comps = []
        for phase in range(N):
            if phase % 2 == 0:
                for j in range(0, N - 1, 2):
                    comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    comps.append((j, j + 1))
        if not _is_valid(comps, bits):
            return comps
    
    best_comps = comps[:]
    best_len = len(best_comps)
    current_len = best_len
    
    # Annealing schedule: temperature starts high, decays
    # Cost = number of comparators (we want to minimize)
    T_start = 10.0  # in units of comparators
    T_end = 0.1
    # We'll use a time-based decay: T = T_end + (T_start - T_end) * (remaining_fraction)^2
    
    # Track best
    elapsed = time.time() - start_time
    total_time = MAX_TIME
    
    while time.time() < deadline:
        remaining_frac = max(0.0, (deadline - time.time()) / total_time)
        T = T_end + (T_start - T_end) * (remaining_frac ** 2)
        
        # Generate a move
        candidate = _make_move(comps, rng)
        
        # Validate
        if not _is_valid(candidate, bits):
            # Reject invalid
            continue
        
        cand_len = len(candidate)
        
        # Acceptance criterion
        if cand_len <= current_len:
            # Better or equal: always accept
            comps = candidate
            current_len = cand_len
            if cand_len < best_len:
                best_comps = candidate[:]
                best_len = cand_len
        else:
            # Worse: accept with probability exp(-(cand_len - current_len) / T)
            delta_cost = cand_len - current_len
            prob = np.exp(-delta_cost / T)
            if rng.random() < prob:
                comps = candidate
                current_len = cand_len
        
        # Check time more frequently
        if time.time() >= deadline:
            break
    
    return best_comps

# EVOLVE-BLOCK-END
