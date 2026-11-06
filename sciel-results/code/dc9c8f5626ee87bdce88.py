"""Optimized construction of a 4-AP-free subset of [1,256] maximizing 3-APs."""
import time
import random
import numpy as np

N = 256
DEADLINE_MARGIN = 5.0  # seconds to spare before the 60s limit

# Precompute 3-APs and 4-AP conflicts
three_aps = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 2 + 1):
        b = a + d
        c = a + 2 * d
        if c <= N:
            three_aps.append((a - 1, b - 1, c - 1))

conflicts = [[] for _ in range(N)]
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        if e <= N:
            # 4-AP: a, b, c, e
            m = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[a - 1].append(m)
            m = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[b - 1].append(m)
            m = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
            conflicts[c - 1].append(m)
            m = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflicts[e - 1].append(m)

def count_3aps(mask):
    count = 0
    for a, b, c in three_aps:
        if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1:
            count += 1
    return count

def is_valid_add(mask, x):
    """Check if adding x (0-indexed) to mask creates a 4-AP."""
    bit = 1 << x
    if mask & bit:
        return False
    for m in conflicts[x]:
        if (mask & m) == m:
            return False
    return True

def construct():
    start_time = time.time()
    
    best_mask = 0
    best_score = 0
    
    # Greedy initialization with multiple orders
    orders = [
        list(range(N)),
        list(range(N - 1, -1, -1)),
        list(range(0, N, 2)) + list(range(1, N, 2)),
        list(range(1, N, 2)) + list(range(0, N, 2)),
    ]
    rng = random.Random(42)
    for _ in range(10):
        o = list(range(N))
        rng.shuffle(o)
        orders.append(o)
        
    for order in orders:
        m = 0
        for x in order:
            if is_valid_add(m, x):
                m |= (1 << x)
        s = count_3aps(m)
        if s > best_score:
            best_score = s
            best_mask = m
            
    # Simulated Annealing
    current_mask = best_mask
    current_score = best_score
    T = 100.0
    T_min = 1e-3
    alpha = 0.999
    
    while time.time() - start_time < DEADLINE_MARGIN:
        # Pick a random element to remove from current_mask
        # Convert to list of indices for efficient random pick?
        # Or just pick random index and check if in mask.
        # To avoid bias, we should pick uniformly from set bits.
        # Number of set bits is ~100.
        # We can maintain a list, but updating it is O(N).
        # Let's just pick random index and retry if not in mask.
        # This is inefficient if density is low, but density is ~40%.
        
        u = -1
        while u == -1:
            idx = random.randint(0, N - 1)
            if (current_mask >> idx) & 1:
                u = idx
                
        if u == -1:
            break
            
        new_mask = current_mask & ~(1 << u)
        
        # Pick a random element to add
        v = random.randint(0, N - 1)
        if (new_mask >> v) & 1:
            continue # v is already in new_mask
            
        if not is_valid_add(new_mask, v):
            continue
            
        new_score = count_3aps(new_mask | (1 << v))
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_mask = new_mask | (1 << v)
            current_score = new_score
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    return [i + 1 for i in range(N) if (current_mask >> i) & 1]
