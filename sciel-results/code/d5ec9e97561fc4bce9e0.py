import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # The task specifies q=3, n=8, radius=1.
    N = q ** n
    powers = [q ** i for i in range(n)]
    
    # Helper to decode an integer to a list of digits
    def decode(x):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    # Precompute coverage bitsets using numpy for speed
    # coverage[c] is an integer where bit i is 1 if point i is covered by codeword c
    # For q=3, n=8, N=6561. We can represent coverage as a list of Python ints.
    coverage = [0] * N
    
    # Precompute neighbors for each point
    # Instead of iterating all N points and all neighbors in Python loops (slow),
    # we can precompute the neighbor list for each point more efficiently or
    # just compute the mask on the fly if it's faster? No, precomputation is better.
    
    # Optimization: Precompute the list of neighbor indices for each point?
    # Or just build the bitmask.
    # Let's try to speed up the bitmask construction.
    # For each c, neighbors are c itself and c +/- powers[i] (with wrap-around logic for q-ary).
    # Actually, for q-ary, changing digit i from v to v+1 (mod q) is not simply adding power[i]
    # unless we handle the carry/borrow.
    
    # Let's stick to the Python loop for precomputation but optimize it slightly.
    # Or use a lookup table for the neighbors?
    # There are N=6561 points. For each, 1 + n*(q-1) = 1 + 8*2 = 17 neighbors.
    # Total operations ~ 6561 * 17 ~ 111k. This is fast.
    
    for c in range(N):
        bits = 1 << c
        for i in range(n):
            base = powers[i]
            val = (c // base) % q
            for new_val in range(q):
                if new_val == val:
                    continue
                neighbor = c + (new_val - val) * base
                bits |= (1 << neighbor)
        coverage[c] = bits

    full_mask = (1 << N) - 1
    
    best_code = None
    best_size = N + 1
    
    # Strategy:
    # 1. Greedy Set Cover to get an initial valid code.
    # 2. Local Search to minimize size.
    #    - Try to remove a codeword if it's redundant.
    #    - Try to swap a codeword for another that allows removing a different one?
    #      Or just simple removal is often sufficient after good greedy.
    
    num_iterations = 20 # Reduced to allow more time per local search
    
    for _ in range(num_iterations):
        if time.time() > deadline - 10:  # Leave time for final check/return
            break
            
        uncovered = full_mask
        selected = []
        
        while uncovered:
            best_c = -1
            best_gain = 0
            # To speed up, we can sample a subset if N is large, but N=6561 is small enough.
            # Random tie-breaking helps escape local minima.
            for c in range(N):
                gain = (uncovered & coverage[c]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                elif gain == best_gain and best_gain > 0:
                    if random.random() < 0.5:
                        best_c = c
            
            if best_gain == 0:
                break
                
            selected.append(best_c)
            uncovered &= ~coverage[best_c]
            
        if uncovered != 0:
            continue
            
        # Local Search: Redundancy Removal
        # We maintain a count of how many times each point is covered by the selected set.
        # This makes the "is it redundant?" check O(1) instead of O(K).
        
        # Count coverage
        # We can't easily do this with bitsets in Python without iterating bits.
        # Alternative: Use a numpy array of counts?
        # N=6561. 
        # Let's try the O(K^2) union check first, it might be fast enough for K~500.
        # 500 * 500 = 250k big-int ORs. Big-ints are 6561 bits ~ 1KB.
        # 250k * 1KB ops ~ 250MB of data movement per check? No, it's bitwise OR.
        # It should be quite fast in C under the hood.
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(selected):
                    continue
                c = selected[i]
                
                # Check if removing c leaves all points covered
                union_others = 0
                for j, other_c in enumerate(selected):
                    if j != i:
                        union_others |= coverage[other_c]
                
                if (union_others & full_mask) == full_mask:
                    selected.pop(i)
                    changed = True
                    break # Restart
            
        # Additional Local Search: Swap
        # Try to replace a codeword with another to allow further removal?
        # This is more complex. Let's stick to removal for now.
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
            
    # If we still have time, do a final polish on the best code found
    if best_code is not None and time.time() < deadline - 2:
        selected = best_code[:]
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(selected):
                    continue
                c = selected[i]
                union_others = 0
                for j, other_c in enumerate(selected):
                    if j != i:
                        union_others |= coverage[other_c]
                if (union_others & full_mask) == full_mask:
                    selected.pop(i)
                    changed = True
                    break
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]

    # Fallback
    if best_code is None:
        best_code = list(range(N)) # Should not happen

    result = []
    for c in best_code:
        result.append(decode(c))
        
    return result
