import sys
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s buffer
    
    fixed = [1, 2, 5, 11]
    max_val = 284
    
    def get_sums(S):
        """Generate all pairwise sums a+b where a<=b from set S."""
        sums = []
        S = list(S)
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.append(S[i] + S[j])
        return sums

    def count_conflicts(S):
        """Return the number of duplicate sums. 0 means valid Sidon set."""
        sums = get_sums(S)
        # Use a set to track seen sums
        seen = set()
        conflicts = 0
        for s in sums:
            if s in seen:
                conflicts += 1
            else:
                seen.add(s)
        return conflicts

    def solve_for_size(N, time_limit):
        """Try to find a Sidon set of size N containing fixed elements."""
        if N < len(fixed):
            return None
        
        # Initialize with fixed elements
        S = set(fixed)
        
        # Add random elements to reach size N
        # We need N - len(fixed) more elements
        candidates = [x for x in range(1, max_val + 1) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates[:N - len(S)]:
            S.add(x)
            
        S = list(S)
        
        # Simulated Annealing
        current_energy = count_conflicts(S)
        if current_energy == 0:
            return S
            
        T = 10.0  # Initial temperature
        cool_rate = 0.995
        
        best_S = S
        best_energy = current_energy
        
        end_time = start_time + time_limit
        
        while time.time() < end_time:
            # Pick a random index to swap out
            i = random.randint(0, len(S) - 1)
            # Pick a random candidate to swap in
            # Ensure we don't pick an element already in S
            new_val = random.randint(1, max_val)
            if new_val in S:
                continue
                
            # Swap S[i] with new_val
            old_val = S[i]
            S[i] = new_val
            
            new_energy = count_conflicts(S)
            
            # Accept if better, or with probability exp(-delta/T)
            delta = new_energy - current_energy
            if delta < 0 or (random.random() < np.exp(-delta / T) if T > 0 else False):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_S = list(S)
                    if best_energy == 0:
                        return best_S
            else:
                # Revert
                S[i] = old_val
                
            T *= cool_rate
            if T < 1e-4:
                break
                
        return best_S if best_energy == 0 else None

    # Try to find size 20, then 19, etc.
    for N in range(20, len(fixed), -1):
        # Allocate time based on N, give more time to larger N
        # We have ~115s total.
        # Let's give 30s to N=20, 20s to N=19, etc.
        time_alloc = min(30, 115 - (time.time() - start_time))
        if time_alloc <= 1:
            break
            
        result = solve_for_size(N, time_alloc)
        if result:
            return result
            
    # If we couldn't find a perfect one, return the best partial set we can construct greedily
    # Fallback: Greedy construction
    S = set(fixed)
    sums = set(get_sums(S))
    
    for x in range(1, max_val + 1):
        if x in S:
            continue
        # Check if x can be added
        valid = True
        for y in S:
            if x + y in sums:
                valid = False
                break
        if valid and 2*x in sums:
            valid = False
            
        if valid:
            S.add(x)
            # Update sums
            for y in list(S): # list(S) to avoid runtime error if S changes? No, S is not changed inside loop except add at end.
                 # Actually, we need to add sums of x with all existing y, and 2x.
                 pass
            # Efficient update:
            new_sums = set()
            for y in S:
                new_sums.add(x + y)
            new_sums.add(2*x)
            sums.update(new_sums)
            
    return list(S)

