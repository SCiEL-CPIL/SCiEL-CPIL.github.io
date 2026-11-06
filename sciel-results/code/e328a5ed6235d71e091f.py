import time
import random
import math
from collections import deque

N = 12

def _compute_reset_length(a, b):
    """
    Compute the exact shortest reset word length via BFS on subset automaton using bitmasks.
    Returns -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    
    # Precompute transition tables for each letter
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    for mask in range(1, 1 << N):
        m = mask
        new_a = 0
        new_b = 0
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_a |= (1 << a[idx])
            new_b |= (1 << b[idx])
            m ^= lsb
        trans_a[mask] = new_a
        trans_b[mask] = new_b
    
    visited = {full_mask}
    queue = deque([(full_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        if mask & (mask - 1) == 0:
            return dist
        
        new_mask_a = trans_a[mask]
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append((new_mask_a, dist + 1))
            
        new_mask_b = trans_b[mask]
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append((new_mask_b, dist + 1))
            
    return -1

def construct():
    deadline = time.time() + 85.0
    rng = random.Random(42)
    
    # Initialize with Pinwheel automaton (known to give n^2-n for N=12, i.e., 132)
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] + [(i - 1) % N for i in range(1, N)]
    best_len = _compute_reset_length(best_a, best_b)
    
    if best_len < 0:
        best_a = list(range(N))
        best_b = list(range(N - 1)) + [0]
        best_len = _compute_reset_length(best_a, best_b)
    
    current_a = list(best_a)
    current_b = list(best_b)
    current_len = best_len
    
    # Simulated Annealing
    T = 20.0
    cooling = 0.998
    min_T = 0.1
    
    while time.time() < deadline - 10.0 and T > min_T:
        a_new = list(current_a)
        b_new = list(current_b)
        
        n_muts = rng.randint(1, 2)
        for _ in range(n_muts):
            if rng.random() < 0.5:
                pos = rng.randrange(N)
                a_new[pos] = rng.randrange(N)
            else:
                pos = rng.randrange(N)
                b_new[pos] = rng.randrange(N)
        
        new_len = _compute_reset_length(a_new, b_new)
        
        if new_len < 0:
            continue
        
        if new_len > current_len:
            current_a, current_b, current_len = a_new, b_new, new_len
            if new_len > best_len:
                best_a, best_b, best_len = list(current_a), list(current_b), new_len
        else:
            delta = new_len - current_len
            if delta < 0:
                prob = math.exp(delta / max(T, 1e-10))
                if rng.random() < prob:
                    current_a, current_b, current_len = a_new, b_new, new_len
            else:
                if rng.random() < 0.3:
                    current_a, current_b, current_len = a_new, b_new, new_len
        
        T *= cooling
    
    # Final local search polish
    if time.time() < deadline - 3.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for pos in range(N):
                for new_val in range(N):
                    if new_val == best_a[pos]:
                        continue
                    test_a = list(best_a)
                    test_a[pos] = new_val
                    l = _compute_reset_length(test_a, best_b)
                    if l > best_len:
                        best_a = test_a
                        best_len = l
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            for pos in range(N):
                for new_val in range(N):
                    if new_val == best_b[pos]:
                        continue
                    test_b = list(best_b)
                    test_b[pos] = new_val
                    l = _compute_reset_length(best_a, test_b)
                    if l > best_len:
                        best_b = test_b
                        best_len = l
                        improved = True
                        break
                if improved:
                    break
    
    return [best_a, best_b]
