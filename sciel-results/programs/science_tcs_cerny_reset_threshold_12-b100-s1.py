"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque

def _reset_length_bitmask(a, b):
    """
    Compute shortest reset word length using bitmasks for efficiency.
    a, b: lists of ints 0..N-1
    Returns length or -1 if not synchronizing.
    """
    start_mask = (1 << N) - 1
    
    # Check if a or b is already a reset
    # If a is constant, length 1.
    if len(set(a)) == 1 or len(set(b)) == 1:
        return 1

    visited = {start_mask}
    queue = deque([(start_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return dist
            
        # Apply a
        next_a = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            next_a |= (1 << a[i])
            m -= lsb
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
            
        # Apply b
        next_b = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            next_b |= (1 << b[i])
            m -= lsb
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
            
    return -1

def construct():
    # Strategy: Search a restricted family of automata known to have high thresholds.
    # Family: a is a cycle, b is a permutation with one "defect" or a near-identity map.
    
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] + [i - 1 for i in range(1, N)] # Standard Cerny
    best_score = _reset_length_bitmask(best_a, best_b)
    
    rng = random.Random(12345)
    
    # Candidate 1: Cycle + Near-Identity with one defect
    # b(i) = i for most i, b(k) = j
    for _ in range(5000):
        a = [(i + 1) % N for i in range(N)]
        b = list(range(N))
        # Introduce 1 or 2 defects in b
        num_defects = rng.choice([1, 2])
        for _ in range(num_defects):
            k = rng.randint(0, N-1)
            j = rng.randint(0, N-1)
            b[k] = j
            
        score = _reset_length_bitmask(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
            
    # Candidate 2: Cycle + Shift with wrap defect
    # b(i) = (i+1) % N for i < N-1, b(N-1) = random
    for _ in range(2000):
        a = [(i + 1) % N for i in range(N)]
        b = [(i + 1) % N for i in range(N-1)]
        b.append(rng.randint(0, N-1))
        
        score = _reset_length_bitmask(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
            
    # Candidate 3: Two Cycles? No, must be binary.
    # Candidate 4: Random Permutation a, Random b with high "entropy" reduction
    # Let's try a few random permutations for a, and b as a "star" or "cluster" map
    for _ in range(2000):
        a = list(range(N))
        rng.shuffle(a)
        # b maps everything to a small set
        target_size = rng.randint(1, 3)
        targets = rng.sample(range(N), target_size)
        b = [rng.choice(targets) for _ in range(N)]
        
        score = _reset_length_bitmask(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
            
    return best_a, best_b
# EVOLVE-BLOCK-END
