import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s buffer for safety
    
    # Pre-compute for each element x (0-indexed), the list of (a, b, c) indices
    # such that {x, a, b, c} forms a 4-AP with x being one of the terms.
    # We'll store for each x, a list of triples of other elements that would complete a 4-AP.
    
    # For speed, we'll use a bitset (bytearray or numpy array) to represent the current set.
    # can_add(x, bitset): check if any of the pre-computed triples for x are all present in bitset.
    
    # Pre-compute conflict triples for each position (0-indexed)
    # conflicts[x] = list of (a, b, c) such that {x, a, b, c} is a 4-AP
    conflicts = [[] for _ in range(N)]
    
    for d in range(1, N):
        # 4-AP: a, a+d, a+2d, a+3d where a+3d < N
        max_a = N - 3 * d
        for a in range(0, max_a):
            x0 = a
            x1 = a + d
            x2 = a + 2 * d
            x3 = a + 3 * d
            
            # For each element in the AP, the other three form a conflict triple
            conflicts[x0].append((x1, x2, x3))
            conflicts[x1].append((x0, x2, x3))
            conflicts[x2].append((x0, x1, x3))
            conflicts[x3].append((x0, x1, x2))
    
    # For faster checking, convert conflicts to numpy arrays
    conflict_arrays = []
    for x in range(N):
        if conflicts[x]:
            arr = np.array(conflicts[x], dtype=np.int32)
            conflict_arrays.append(arr)
        else:
            conflict_arrays.append(None)
    
    # Helper to check if adding x (0-indexed) to the bitset creates a 4-AP
    def can_add(x, bitset):
        arr = conflict_arrays[x]
        if arr is None:
            return True
        # Check if any row has all three elements set in bitset
        # arr has shape (num_conflicts, 3)
        # For each conflict (a, b, c), check if bitset[a] and bitset[b] and bitset[c] are all 1
        # Vectorized: 
        a = arr[:, 0]
        b = arr[:, 1]
        c = arr[:, 2]
        # Check all three at once
        mask = (bitset[a] & bitset[b] & bitset[c])
        return not mask.any()
    
    # Randomized Greedy with bitset
    current_best_size = 1
    current_best_set = {1}
    
    while time.time() < deadline:
        candidates = list(range(N))
        random.shuffle(candidates)
        bitset = np.zeros(N, dtype=np.uint8)
        current_set = set()
        
        for x in candidates:
            if can_add(x, bitset):
                bitset[x] = 1
                current_set.add(x + 1)  # Convert back to 1-indexed
        
        if len(current_set) > current_best_size:
            current_best_size = len(current_set)
            current_best_set = current_set.copy()
            
    return sorted(current_best_set)
