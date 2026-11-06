import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    # Precompute: for each x, list of (y, z) such that {x,y,z} is a 3-AP
    # We'll use this for fast conflict checking
    # neighbors[x] = list of (y, z) where x, y, z form a 3-AP (x is one of the three)
    neighbors = {i: [] for i in range(1, n+1)}
    for a in range(1, n+1):
        for d in range(1, n+1):
            b = a + d
            c = a + 2*d
            if b <= n and c <= n:
                # (a, b, c) is a 3-AP
                neighbors[a].append((b, c))
                neighbors[b].append((a, c))
                neighbors[c].append((a, b))
            # Also (c, b, a) is covered by same
    
    # For SA: we need to quickly check if adding x to set S creates a 3-AP
    # x creates a 3-AP with S if there exist y, z in S such that {x,y,z} is a 3-AP
    # i.e., for some (y,z) in neighbors[x], both y and z are in S
    
    # Precompute inv
    inv = [0.0] + [1.0/i for i in range(1, n+1)]
    
    def can_add_fast(x, mask):
        """Check if x can be added to mask without creating 3-AP."""
        for y, z in neighbors[x]:
            if (mask & (1 << (y-1))) and (mask & (1 << (z-1))):
                return False
        return True
    
    def get_elements(mask):
        return [i for i in range(1, n+1) if mask & (1 << (i-1))]
    
    def sum_recip(mask):
        s = 0.0
        for i in range(1, n+1):
            if mask & (1 << (i-1)):
                s += inv[i]
        return s
    
    def is_valid(mask):
        """Verify no 3-AP exists."""
        elems = get_elements(mask)
        elem_set = set(elems)
        for a in elems:
            for b in elems:
                if a == b:
                    continue
                c = 2*b - a
                if c in elem_set and c != a and c != b:
                    return False
        return True
    
    # Simulated Annealing
    # State: bitmask representing the set
    # Move: add a random element not in set, or remove a random element in set
    # Objective: maximize sum of reciprocals, subject to no 3-AP
    
    def simulated_annealing(initial_mask, time_limit):
        mask = initial_mask
        current_energy = sum_recip(mask)  # We maximize this, so "energy" to minimize is -sum
        # Actually let's just track the sum and accept if new_sum > old_sum or with prob exp((new-old)/T)
        
        T = 0.05  # Initial temperature
        T_min = 0.0001
        # We'll use a simple cooling schedule based on iteration count
        # Estimate: each move takes ~50-100 microseconds for the conflict check
        # So we can do ~100k-200k moves in 10 seconds
        # Let's target ~500k total moves across all SA runs
        
        best_mask = mask
        best_score = current_energy
        best_time = time.time()
        
        # Precompute for speed: for each x, the list of (y,z) pairs
        # We already have neighbors[x]
        
        rng = random.Random(42)
        
        iteration = 0
        total_iterations = 800000  # Adjust based on time
        
        while time.time() < time_limit:
            iteration += 1
            # Cooling: T decreases geometrically
            T = max(T_min, T * 0.99995)
            
            # Random move: add or remove a random element
            if rng.random() < 0.5:
                # Try to add a random element
                x = rng.randint(1, n)
                if mask & (1 << (x-1)):
                    continue  # Already in set, skip
                if can_add_fast(x, mask):
                    new_mask = mask | (1 << (x-1))
                    delta = inv[x]
                    # Always accept addition (it increases the sum)
                    mask = new_mask
                    current_energy += delta
                    if current_energy > best_score:
                        best_score = current_energy
                        best_mask = mask
                else:
                    # Can't add, try removing something else
                    pass
            else:
                # Try to remove a random element
                # Pick a random element in the set
                elems = get_elements(mask)
                if not elems:
                    continue
                x = rng.choice(elems)
                new_mask = mask & ~(1 << (x-1))
                delta = -inv[x]
                # Removing always decreases the sum, so only accept with probability
                new_energy = current_energy + delta
                if new_energy > current_energy:
                    # Shouldn't happen for removal, but just in case
                    mask = new_mask
                    current_energy = new_energy
                else:
                    # Accept with probability exp(delta/T)
                    if rng.random() < np.exp(delta / T):
                        mask = new_mask
                        current_energy = new_energy
            
            # Occasionally try a swap: remove one, add another
            if rng.random() < 0.3:
                # Pick a random element to remove
                elems = get_elements(mask)
                if elems:
                    x = rng.choice(elems)
                    # Pick a random element to add
                    y = rng.randint(1, n)
                    if not (mask & (1 << (y-1))):
                        mask_without_x = mask & ~(1 << (x-1))
                        if can_add_fast(y, mask_without_x):
                            delta = inv[y] - inv[x]
                            new_energy = current_energy + delta
                            if new_energy > current_energy:
                                mask = mask_without_x | (1 << (y-1))
                                current_energy = new_energy
                                if current_energy > best_score:
                                    best_score = current_energy
                                    best_mask = mask
                            else:
                                if rng.random() < np.exp(delta / T):
                                    mask = mask_without_x | (1 << (y-1))
                                    current_energy = new_energy
        
        return best_mask, best_score
    
    # Phase 1: Multiple greedy initializations
    def greedy_construct(order):
        mask = 0
        for x in order:
            if can_add_fast(x, mask):
                mask |= (1 << (x-1))
        return mask
    
    best_mask = 0
    best_score = 0.0
    
    # Start 1: Standard greedy (ascending order)
    M1 = greedy_construct(list(range(1, n+1)))
    s1 = sum_recip(M1)
    best_mask, best_score = M1, s1
    
    # Run SA from this start
    remaining = deadline - time.time()
    if remaining > 5.0:
        M1, s1 = simulated_annealing(M1, time.time() + min(remaining - 2.0, 30.0))
        if s1 > best_score:
            best_score = s1
            best_mask = M1
    
    # Start 2: Descending greedy (different structure)
    remaining = deadline - time.time()
    if remaining > 5.0:
        M2 = greedy_construct(list(range(n, 0, -1)))
        M2, s2 = simulated_annealing(M2, time.time() + min(remaining - 2.0, 20.0))
        if s2 > best_score:
            best_score = s2
            best_mask = M2
    
    # Start 3: Randomized greedy
    remaining = deadline - time.time()
    if remaining > 5.0:
        for trial in range(3):
            if time.time() > deadline - 5.0:
                break
            order = list(range(1, n+1))
            random.shuffle(order)
            M3 = greedy_construct(order)
            M3, s3 = simulated_annealing(M3, time.time() + min(remaining - 3.0, 10.0))
            if s3 > best_score:
                best_score = s3
                best_mask = M3
    
    # Start 4: Sparse initialization (only odds, or only numbers coprime to 6, etc.)
    remaining = deadline - time.time()
    if remaining > 5.0:
        # Numbers coprime to 6: 1,5,7,11,13,17,...
        sparse = [x for x in range(1, n+1) if x % 2 != 0 and x % 3 != 0]
        M4 = greedy_construct(sparse)
        # Then fill in the rest
        for x in range(1, n+1):
            if x not in sparse and can_add_fast(x, M4):
                M4 |= (1 << (x-1))
        M4, s4 = simulated_annealing(M4, time.time() + min(remaining - 2.0, 15.0))
        if s4 > best_score:
            best_score = s4
            best_mask = M4
    
    # Final verification
    if not is_valid(best_mask):
        # Fallback: use a known valid set
        # Simple construction: all numbers with only odd powers of 2 in their factorization
        # Or just use the greedy result which should be valid
        # Let's rebuild with greedy
        best_mask = greedy_construct(list(range(1, n+1)))
        best_score = sum_recip(best_mask)
    
    result = get_elements(best_mask)
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
