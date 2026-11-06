"""Program using fast simulated annealing with precomputed transition tables for Černý problem (N=12)."""
import time
import random
import math
from collections import deque

N = 12

def _build_tables(a, b):
    """Precompute next-state masks for all subsets."""
    size = 1 << N
    next_a = [0] * size
    next_b = [0] * size
    
    # Precompute bit positions for speed? Or just use bit_length.
    # Since N=12, bit_length is fast.
    for mask in range(size):
        ma = 0
        mb = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            ma |= 1 << a[idx]
            mb |= 1 << b[idx]
            m ^= lsb
        next_a[mask] = ma
        next_b[mask] = mb
    return next_a, next_b

def _bfs_length(next_a, next_b):
    """Compute exact reset length using precomputed tables."""
    full = (1 << N) - 1
    visited = {full}
    queue = deque([(full, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        if mask & (mask - 1) == 0:
            return dist
        
        na = next_a[mask]
        if na not in visited:
            visited.add(na)
            queue.append((na, dist + 1))
            
        nb = next_b[mask]
        if nb not in visited:
            visited.add(nb)
            queue.append((nb, dist + 1))
            
    return -1

def construct():
    deadline = time.time() + 85.0
    rng = random.Random(12345)
    
    # Initialize with Pinwheel
    # a: i -> i+1
    # b: 0 -> 0, i -> i-1
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] + [(i - 1) % N for i in range(1, N)]
    
    best_next_a, best_next_b = _build_tables(best_a, best_b)
    best_len = _bfs_length(best_next_a, best_next_b)
    
    # Local Search to improve initial Pinwheel
    improved = True
    while improved:
        improved = False
        for letter_idx in range(2):
            orig = list(best_a) if letter_idx == 0 else list(best_b)
            for pos in range(N):
                for new_val in range(N):
                    if new_val == orig[pos]:
                        continue
                    if letter_idx == 0:
                        test_a = list(best_a)
                        test_a[pos] = new_val
                        test_b = best_b
                    else:
                        test_a = best_a
                        test_b = list(best_b)
                        test_b[pos] = new_val
                    
                    na, nb = _build_tables(test_a, test_b)
                    l = _bfs_length(na, nb)
                    if l > best_len:
                        if letter_idx == 0:
                            best_a = test_a
                        else:
                            best_b = test_b
                        best_len = l
                        best_next_a, best_next_b = na, nb
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Simulated Annealing
    current_a = list(best_a)
    current_b = list(best_b)
    current_next_a, current_next_b = best_next_a, best_next_b
    current_len = best_len
    
    T_start = 10.0
    T_end = 0.01
    max_iterations = 50000
    no_improve_count = 0
    patience = 1000
    
    for i in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
            
        progress = i / max_iterations
        T = T_start * math.exp(-progress * math.log(T_start / T_end))
        
        # Mutate
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
        
        # Evaluate
        na_new, nb_new = _build_tables(a_new, b_new)
        new_len = _bfs_length(na_new, nb_new)
        
        if new_len < 0:
            continue # Not synchronizing
        
        if new_len > current_len:
            current_a, current_b = a_new, b_new
            current_next_a, current_next_b = na_new, nb_new
            current_len = new_len
            no_improve_count = 0
            if new_len > best_len:
                best_a, best_b = list(current_a), list(current_b)
                best_next_a, best_next_b = list(current_next_a), list(current_next_b)
                best_len = new_len
        else:
            delta = new_len - current_len
            if delta < 0 and rng.random() < math.exp(delta / max(T, 1e-10)):
                current_a, current_b = a_new, b_new
                current_next_a, current_next_b = na_new, nb_new
                current_len = new_len
            no_improve_count += 1
        
        if no_improve_count > patience:
            break
            
    # Final Local Search on Best
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for letter_idx in range(2):
                orig = list(best_a) if letter_idx == 0 else list(best_b)
                for pos in range(N):
                    for new_val in range(N):
                        if new_val == orig[pos]:
                            continue
                        if letter_idx == 0:
                            test_a = list(best_a)
                            test_a[pos] = new_val
                            test_b = best_b
                        else:
                            test_a = best_a
                            test_b = list(best_b)
                            test_b[pos] = new_val
                        
                        na, nb = _build_tables(test_a, test_b)
                        l = _bfs_length(na, nb)
                        if l > best_len:
                            if letter_idx == 0:
                                best_a = test_a
                            else:
                                best_b = test_b
                            best_len = l
                            best_next_a, best_next_b = na, nb
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return [best_a, best_b]
