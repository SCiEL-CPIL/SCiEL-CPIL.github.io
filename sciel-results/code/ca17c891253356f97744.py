import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute for each x, the list of pairs (y, z) such that {x, y, z} forms a 3-AP
    # x can be first, middle, or last term
    # If x is first: x, y, z with y = x+d, z = x+2d => 2y = x+z
    # If x is middle: y, x, z with y = x-d, z = x+d => 2x = y+z
    # If x is last: y, z, x with z = y+d, x = y+2d => 2z = y+x
    
    # For fast checking with bitmask:
    # When adding x to set S (represented as bitmask), we need to check:
    # For each pair (y, z) in conflict_pairs[x], not (both y and z are in S)
    
    conflict_pairs = {i: [] for i in range(1, N+1)}
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            c = 2*b - a
            if c <= N and c > b:
                # AP: a, b, c
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))
    
    def can_add(mask, x):
        """Check if x can be added to the set represented by mask without creating a 3-AP."""
        for (y, z) in conflict_pairs[x]:
            if ((mask >> (y-1)) & 1) and ((mask >> (z-1)) & 1):
                return False
        return True
    
    def greedy_construct(order):
        mask = 0
        for x in order:
            if can_add(mask, x):
                mask |= (1 << (x-1))
        return mask
    
    def mask_to_set(mask):
        return [i+1 for i in range(N) if (mask >> i) & 1]
    
    def set_score(mask):
        s = 0.0
        for i in range(N):
            if (mask >> i) & 1:
                s += 1.0 / (i+1)
        return s
    
    def hill_climb(mask):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add elements (smallest first for max reciprocal gain)
            for x in range(1, N+1):
                if (mask >> (x-1)) & 1:
                    continue
                if can_add(mask, x):
                    new_mask = mask | (1 << (x-1))
                    mask = new_mask
                    improved = True
            
            # Try swaps: remove x_out, add x_in where x_in < x_out
            S = mask_to_set(mask)
            S_set = set(S)
            x_outs = sorted(S, reverse=True)
            x_ins = sorted([x for x in range(1, N+1) if x not in S_set])
            
            for x_out in x_outs:
                if time.time() > deadline:
                    break
                mask_without = mask & ~(1 << (x_out-1))
                for x_in in x_ins:
                    if x_in >= x_out:
                        break
                    if can_add(mask_without, x_in):
                        mask = mask_without | (1 << (x_in-1))
                        improved = True
                        break
                if improved:
                    break
        
        return mask
    
    # Phase 1: Greedy in ascending order (best for reciprocal sum)
    mask = greedy_construct(range(1, N+1))
    mask = hill_climb(mask)
    
    best_mask = mask
    best_score = set_score(mask)
    
    # Phase 2: Random restarts with biased ordering
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 20:
        # Create a biased random order: small numbers more likely to come first
        order = list(range(1, N+1))
        # Shuffle but with bias towards small numbers
        # Use exponential weighting: weight = 1/x
        weights = [1.0/x for x in order]
        # Partial shuffle: swap adjacent elements with probability based on weight
        for i in range(len(order)-1, 0, -1):
            # With some probability, keep the smaller number earlier
            if random.random() < 0.7:
                # Keep order (smaller tends to stay earlier)
                pass
            else:
                order[i], order[i-1] = order[i-1], order[i]
        
        mask_rand = greedy_construct(order)
        mask_rand = hill_climb(mask_rand)
        score_rand = set_score(mask_rand)
        
        if score_rand > best_score:
            best_score = score_rand
            best_mask = mask_rand
        attempts += 1
    
    return mask_to_set(best_mask)
