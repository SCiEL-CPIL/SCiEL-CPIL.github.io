import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 82.0
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)

    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    # Helper: popcount
    def popcount(x):
        return bin(x).count('1')
    
    # Start with all odds
    current_mask = 0
    for i in odds_indices:
        current_mask |= (1 << i)
    
    best_mask = current_mask
    best_size = popcount(current_mask)
    
    # Phase 1: Greedy - add elements that have 0 conflicts, then try to add with minimal conflicts
    changed = True
    while changed and time.time() < DEADLINE - 60:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 60:
                break
            if (current_mask >> e) & 1:
                continue
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
                best_mask = current_mask
                best_size = popcount(current_mask)
                changed = True
    
    # Phase 2: Simulated Annealing with 1-add and 2-swap moves
    T = 3.0
    cooling = 0.9997
    max_iters = 300000
    no_improve = 0
    NO_IMPROVE_LIMIT = 5000
    
    current_size = best_size
    
    for i_iter in range(max_iters):
        if time.time() > DEADLINE - 3:
            break
        
        r = random.random()
        
        if r < 0.70:
            # Move 1: Add one element (remove its conflicts)
            # Pick from evens preferentially
            if random.random() < 0.85:
                e_candidates = [idx for idx in evens_indices if not ((current_mask >> idx) & 1)]
            else:
                e_candidates = [idx for idx in odds_indices if not ((current_mask >> idx) & 1)]
            
            if not e_candidates:
                continue
            e = random.choice(e_candidates)
            
            conflicts = conflict_mask[e] & current_mask
            num_conflicts = popcount(conflicts)
            new_size = current_size - num_conflicts + 1
            delta = new_size - current_size
            
            if delta >= 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                current_mask = (current_mask & ~conflicts) | (1 << e)
                current_size = new_size
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
                    no_improve = 0
                else:
                    no_improve += 1
                    
        elif r < 0.90:
            # Move 2: 2-swap - remove 2 random elements, add 1 random element
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if len(bits) < 2:
                continue
            
            # Pick 2 elements to remove
            r1 = random.choice(bits)
            remaining = [i for i in bits if i != r1]
            r2 = random.choice(remaining)
            
            # Pick element to add (not in set, not r1 or r2)
            not_in = [i for i in range(N) if not ((current_mask >> i) & 1)]
            if not not_in:
                continue
            e = random.choice(not_in)
            
            # Remove r1, r2, add e (and remove e's conflicts)
            removed_mask = (1 << r1) | (1 << r2)
            conflicts = conflict_mask[e] & current_mask & ~removed_mask
            num_conflicts = popcount(conflicts)
            
            new_size = current_size - 2 - num_conflicts + 1
            delta = new_size - current_size
            
            if delta >= 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                current_mask = (current_mask & ~removed_mask & ~conflicts) | (1 << e)
                current_size = new_size
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
                    no_improve = 0
                else:
                    no_improve += 1
        else:
            # Move 3: Random removal to escape local minima
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if bits:
                r_idx = random.choice(bits)
                current_mask &= ~(1 << r_idx)
                current_size -= 1
                no_improve += 1
        
        T *= cooling
        if T < 0.05:
            T = 0.05
        
        if no_improve > NO_IMPROVE_LIMIT:
            # Reset temperature to explore more
            T = max(T, 1.0)
            no_improve = 0

    # Final polish: try to add any element with 0 conflicts
    for e in evens_indices + odds_indices:
        if time.time() > DEADLINE - 1:
            break
        if (best_mask >> e) & 1:
            continue
        if (conflict_mask[e] & best_mask) == 0:
            best_mask |= (1 << e)
            best_size = popcount(best_mask)

    # Convert back to numbers
    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    
    return result
