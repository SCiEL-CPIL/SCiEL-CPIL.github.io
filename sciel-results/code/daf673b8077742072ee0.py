"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256

def _count_mono4(c):
    """Count monochromatic 4-term APs. c is a numpy array of 0/1 of length N (0-indexed for 1..N)."""
    count = 0
    for d in range(1, N - 2):
        max_a = N - 3 * d
        if max_a < 1:
            break
        # indices: a-1, a-1+d, a-1+2d, a-1+3d for a in 1..max_a
        # That's indices 0..max_a-1 for the start
        idx0 = np.arange(max_a)
        idx1 = idx0 + d
        idx2 = idx0 + 2 * d
        idx3 = idx0 + 3 * d
        # Check all four are equal
        count += np.sum((c[idx0] == c[idx1]) & (c[idx1] == c[idx2]) & (c[idx2] == c[idx3]))
    return int(count)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    tm = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        # Count trailing ones in (i-1) or use bit count
        tm[i] = tm[i >> 1] ^ (i & 1)
    return tm

def _greedy_init():
    """Greedy: assign each position to minimize mono-4APs so far."""
    c = np.zeros(N, dtype=np.int8)
    # Precompute for each position i, which 4-APs it participates in
    # For efficiency, just do simple greedy
    for i in range(N):
        # Try 0
        c[i] = 0
        cost0 = _count_mono4(c)
        c[i] = 1
        cost1 = _count_mono4(c)
        c[i] = 0 if cost0 <= cost1 else 1
    return c

def _local_search(c, max_iter=50000, start_temp=1.0, end_temp=0.01):
    """Simulated annealing local search."""
    c = c.copy()
    current_cost = _count_mono4(c)
    best_c = c.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if i % 1000 == 0:
            elapsed = time.time() - _local_search._start
            if elapsed > 140:
                break
        temp = start_temp * (end_temp / start_temp) ** (i / max_iter)
        
        # Pick a random position to flip
        pos = random.randint(0, N - 1)
        old_val = c[pos]
        c[pos] = 1 - old_val
        new_cost = _count_mono4(c)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_c = c.copy()
        else:
            c[pos] = old_val
    
    return best_c, best_cost

def construct():
    _local_search._start = time.time()
    deadline = _local_search._start + 140  # safety margin
    
    best_c = None
    best_cost = float('inf')
    
    # Strategy 1: Thue-Morse + local search
    tm = _thue_morse(N)
    c, cost = _local_search(tm, max_iter=30000)
    if cost < best_cost:
        best_cost = cost
        best_c = c.copy()
    
    # Check time
    if time.time() > deadline - 30:
        return [int(x) for x in best_c]
    
    # Strategy 2: Random restarts with local search
    for restart in range(20):
        if time.time() > deadline - 20:
            break
        c_rand = np.random.randint(0, 2, N).astype(np.int8)
        c, cost = _local_search(c_rand, max_iter=10000)
        if cost < best_cost:
            best_cost = cost
            best_c = c.copy()
    
    # Strategy 3: Greedy initialization + local search
    if time.time() < deadline - 30:
        c_greedy = _greedy_init()
        c, cost = _local_search(c_greedy, max_iter=20000)
        if cost < best_cost:
            best_cost = cost
            best_c = c.copy()
    
    # Final: a few more random restarts if time permits
    while time.time() < deadline - 10:
        c_rand = np.random.randint(0, 2, N).astype(np.int8)
        c, cost = _local_search(c_rand, max_iter=5000)
        if cost < best_cost:
            best_cost = cost
            best_c = c.copy()
    
    return [int(x) for x in best_c]
# EVOLVE-BLOCK-END
