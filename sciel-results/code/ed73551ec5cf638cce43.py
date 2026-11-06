import time
import random
import numpy as np

DEADLINE = 55.0  # safety margin below 60s

def _has_4ap_bitset(bitset, x, N):
    """Check if adding x (0-indexed) creates a 4-AP. Uses bit operations."""
    # Case 1: x is the 1st term (a): need a+d, a+2d, a+3d all present
    #   x = a, so need x+d, x+2d, x+3d in set
    for d in range(1, (N - x) // 3 + 1):
        if (bitset >> (x + d)) & 1 and (bitset >> (x + 2*d)) & 1 and (bitset >> (x + 3*d)) & 1:
            return True
    # Case 2: x is the 2nd term (a+d): need a, a+2d, a+3d present
    #   x = a+d => a = x-d; need x-d, x+d, x+2d
    for d in range(1, min(x, (N - x - 1) // 2) + 1):
        if d >= 1 and (bitset >> (x - d)) & 1 and (bitset >> (x + d)) & 1 and (bitset >> (x + 2*d)) & 1:
            return True
    # Case 3: x is the 3rd term (a+2d): need a, a+d, a+3d present
    #   x = a+2d => a = x-2d; need x-2d, x-d, x+d
    for d in range(1, min(x // 2, (N - x - 1) // 1) + 1):
        if d >= 1 and (bitset >> (x - 2*d)) & 1 and (bitset >> (x - d)) & 1 and (bitset >> (x + d)) & 1:
            return True
    # Case 4: x is the 4th term (a+3d): need a, a+d, a+2d present
    #   x = a+3d => a = x-3d; need x-3d, x-2d, x-d
    for d in range(1, x // 3 + 1):
        if (bitset >> (x - 3*d)) & 1 and (bitset >> (x - 2*d)) & 1 and (bitset >> (x - d)) & 1:
            return True
    return False


def _has_4ap_fast(bitset, x, N):
    """Vectorized-ish check using precomputed masks for speed."""
    # We'll use the loop version but with early exit; for N=512 the loops are short.
    # Max d values: case1 ~170, case2 ~170, case3 ~170, case4 ~170 => ~680 checks worst case
    # For N=512 this is fast enough in Python if we minimize overhead.
    
    # Actually let's use a smarter approach: precompute for each x the list of (d, positions)
    # But that's too much memory. Instead, just do the loop with bit checks.
    
    # Case 1: x is first term
    max_d1 = (N - 1 - x) // 3
    for d in range(1, max_d1 + 1):
        if (bitset >> (x + d)) & 1:
            if (bitset >> (x + 2*d)) & 1:
                if (bitset >> (x + 3*d)) & 1:
                    return True
    # Case 2: x is second term
    max_d2 = min(x, (N - 1 - x) // 2)
    for d in range(1, max_d2 + 1):
        if (bitset >> (x - d)) & 1:
            if (bitset >> (x + d)) & 1:
                if (bitset >> (x + 2*d)) & 1:
                    return True
    # Case 3: x is third term
    max_d3 = min(x // 2, N - 1 - x)
    for d in range(1, max_d3 + 1):
        if (bitset >> (x - 2*d)) & 1:
            if (bitset >> (x - d)) & 1:
                if (bitset >> (x + d)) & 1:
                    return True
    # Case 4: x is fourth term
    max_d4 = x // 3
    for d in range(1, max_d4 + 1):
        if (bitset >> (x - 3*d)) & 1:
            if (bitset >> (x - 2*d)) & 1:
                if (bitset >> (x - d)) & 1:
                    return True
    return False


def _greedy_randomized(N, rng, order=None):
    """Randomized greedy: process elements in given order, add if no 4-AP formed."""
    bitset = 0
    if order is None:
        order = list(range(N))
        rng.shuffle(order)
    for x in order:
        if not _has_4ap_fast(bitset, x, N):
            bitset |= (1 << x)
    return bitset


def _local_search_add(bitset, N, rng, max_steps=5000, deadline=None):
    """Try to add random elements that don't create 4-APs."""
    count = 0
    for _ in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        x = rng.randint(0, N - 1)
        if (bitset >> x) & 1:
            continue
        if not _has_4ap_fast(bitset, x, N):
            bitset |= (1 << x)
            count += 1
    return bitset, count


def _local_search_swap(bitset, N, rng, max_steps=5000, deadline=None):
    """Try to swap: remove one, add two (or add one after removing one)."""
    # Simple version: try removing a random element and then adding two others
    # This is complex; instead do: try adding, if fails try removing then adding
    count = 0
    for _ in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        x = rng.randint(0, N - 1)
        if (bitset >> x) & 1:
            continue
        if not _has_4ap_fast(bitset, x, N):
            bitset |= (1 << x)
            count += 1
    return bitset, count


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    N = 512
    rng = random.Random(42)
    
    best_bitset = 0
    best_size = 0
    
    # Pre-compute: for each x, the list of "conflict patterns" is too expensive.
    # Instead, use the fast check function.
    
    # Strategy: Multiple randomized greedy starts + local search improvement
    # Each restart: shuffle order, greedy, then local search adds
    
    restarts_done = 0
    while time.time() < deadline - 2.0:  # leave 2s safety
        # Randomized greedy with random order
        order = list(range(N))
        rng.shuffle(order)
        bitset = _greedy_randomized(N, rng, order)
        sz = bin(bitset).count('1')
        
        # Local search: try to add more elements
        bitset, added = _local_search_add(bitset, N, rng, max_steps=3000, deadline=deadline - 1.0)
        sz = bin(bitset).count('1')
        
        if sz > best_size:
            best_size = sz
            best_bitset = bitset
            restarts_done += 1
        
        # If we've done enough restarts and size is good, we can do more local search
        # on the best solution
        if restarts_done > 5 and time.time() < deadline - 3.0:
            # Extra local search on best
            best_bitset, _ = _local_search_add(best_bitset, N, rng, max_steps=2000, deadline=deadline - 1.0)
            sz = bin(best_bitset).count('1')
            if sz > best_size:
                best_size = sz
                best_bitset = best_bitset
        
        # Occasionally do a "perturbation" restart: start from best, remove random, re-add
        if restarts_done > 3 and time.time() < deadline - 2.0 and rng.random() < 0.3:
            perturbed = best_bitset
            # Remove 5 random elements
            for _ in range(5):
                elems = [i for i in range(N) if (perturbed >> i) & 1]
                if elems:
                    rem = rng.choice(elems)
                    perturbed &= ~(1 << rem)
            # Re-greedy add
            order = list(range(N))
            rng.shuffle(order)
            for x in order:
                if (perturbed >> x) & 1:
                    continue
                if not _has_4ap_fast(perturbed, x, N):
                    perturbed |= (1 << x)
            # Local search
            perturbed, _ = _local_search_add(perturbed, N, rng, max_steps=2000, deadline=deadline - 1.0)
            sz = bin(perturbed).count('1')
            if sz > best_size:
                best_size = sz
                best_bitset = perturbed
                restarts_done += 1
    
    # Final: convert to sorted list of 1-indexed values
    result = [i + 1 for i in range(N) if (best_bitset >> i) & 1]
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result
