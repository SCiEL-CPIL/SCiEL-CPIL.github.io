import numpy as np
import time
from itertools import combinations
import random

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58  # Safety margin

    # Precompute masks for blocking
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)

    # Step 1: Find a [17,8,6] linear code
    k = 8
    linear_code = None

    # Try random generator matrices
    for attempt in range(200):
        if time.time() - start_time > 5.0:
            break
        G = np.random.randint(0, 2, size=(k, N))
        
        # Convert G rows to integers
        G_ints = []
        for i in range(k):
            val = 0
            for j in range(N):
                if G[i, j] == 1:
                    val |= (1 << j)
            G_ints.append(val)
        
        # Generate all codewords via iterative XOR
        codewords = [0]
        for i in range(k):
            new_codewords = [c ^ G_ints[i] for c in codewords]
            codewords.extend(new_codewords)
        
        # Check min distance: for linear codes, min weight of nonzero codewords = min distance
        min_d = N + 1
        for c in codewords:
            if c == 0:
                continue
            w = bin(c).count('1')
            if w < min_d:
                min_d = w
                if min_d < MIN_DIST:
                    break
        
        if min_d >= MIN_DIST:
            linear_code = codewords
            break

    # Step 2: If no linear code found, use greedy from 0
    if linear_code is None:
        blocked = bytearray(1 << N)
        code = []
        
        def add_word(w):
            if blocked[w]:
                return
            code.append(w)
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 1
        
        add_word(0)
        for w in range(1, 1 << N):
            if time.time() - start_time > time_limit:
                break
            if not blocked[w]:
                add_word(w)
        return code

    # Step 3: Expand the linear code with greedy
    blocked = bytearray(1 << N)
    code = list(linear_code)  # 256 words
    
    # Block all neighbors of the linear code
    for w in code:
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                blocked[w ^ mask] = 1

    # Greedy scan for additional words
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if not blocked[w]:
            code.append(w)
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 1

    # Step 4: Simulated annealing to escape local optima
    # Model as spin glass: energy = penalty for close pairs - reward for size
    if time.time() - start_time < time_limit - 2:
        code_set = set(code)
        best_code = list(code)
        best_len = len(code)
        
        # Precompute popcount for all 17-bit values
        popcount = np.zeros(1 << N, dtype=np.uint8)
        for i in range(1, 1 << N):
            popcount[i] = popcount[i >> 1] + (i & 1)
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.1
        total_steps = 5000
        time_per_step = (time_limit - 2 - (time.time() - start_time)) / total_steps
        
        for step in range(total_steps):
            if time.time() - start_time > time_limit - 1:
                break
            
            # Cooling schedule (geometric)
            T = T_start * (T_end / T_start) ** (step / total_steps)
            
            # Choose a random word to remove
            idx_remove = random.randint(0, len(code) - 1)
            w_remove = code[idx_remove]
            
            # Find a candidate to add: random unblocked word
            # First, check how many unblocked words exist
            # We'll pick randomly and check
            candidate = random.randint(0, (1 << N) - 1)
            
            # Check if candidate is blocked
            if blocked[candidate]:
                # Try a few more random candidates
                found = False
                for _ in range(10):
                    candidate = random.randint(0, (1 << N) - 1)
                    if not blocked[candidate]:
                        found = True
                        break
                if not found:
                    continue
            
            # Compute energy change
            # Remove w_remove: loses its contribution to size (-1)
            # Add candidate: gains its contribution to size (+1)
            # Net size change: 0 (swap)
            
            # But we need to check if the swap is valid
            # After removing w_remove, some words become unblocked
            # After adding candidate, some words become blocked
            
            # For a swap to be valid, candidate must be at distance >= MIN_DIST from all other code words
            # (except w_remove which is being removed)
            
            # Check distance from candidate to all other code words
            valid_swap = True
            for j, w in enumerate(code):
                if j == idx_remove:
                    continue
                d = popcount[candidate ^ w]
                if d < MIN_DIST:
                    valid_swap = False
                    break
            
            if not valid_swap:
                # Accept with probability based on energy
                # Energy increase = number of violations
                # For simplicity, reject if invalid
                continue
            
            # Perform the swap
            code[idx_remove] = candidate
            code_set.discard(w_remove)
            code_set.add(candidate)
            
            # Update blocked array
            # Unblock neighbors of w_remove
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w_remove ^ mask] = 0
            
            # Block neighbors of candidate
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[candidate ^ mask] = 1
            
            # Track best
            if len(code) > best_len:
                best_len = len(code)
                best_code = list(code)
        
        # Restore best found
        if best_len > len(code):
            code = best_code
    
    return code
