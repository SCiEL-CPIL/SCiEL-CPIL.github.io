"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Precompute all bad triples: (a, b, c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.add((a, b, c))
    
    # For each element x, precompute list of (y, z) pairs such that {x,y,z} is bad
    # Store as bitmask pairs for fast checking
    conflict_pairs_by_elem = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflict_pairs_by_elem[a].append((b, c))
        conflict_pairs_by_elem[b].append((a, c))
        conflict_pairs_by_elem[c].append((a, b))
    
    # Also precompute bitmask for each element's direct conflict partners
    conflict_mask = {x: 0 for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflict_mask[a] |= (1 << b) | (1 << c)
        conflict_mask[b] |= (1 << a) | (1 << c)
        conflict_mask[c] |= (1 << a) | (1 << b)
    
    def can_add_bitmask(x, mask):
        """Check if x can be added to set represented by bitmask."""
        for (y, z) in conflict_pairs_by_elem[x]:
            if (mask >> y) & 1 and (mask >> z) & 1:
                return False
        return True
    
    def popcount(mask):
        return bin(mask).count('1')
    
    def fill_set_bitmask(mask):
        """Greedily add all possible elements to the bitmask set."""
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if not (mask >> x) & 1 and can_add_bitmask(x, mask):
                    mask |= (1 << x)
                    changed = True
        return mask
    
    def local_search_bitmask(mask, max_rounds=20):
        """Try 1-out 1-in swap moves to improve the set."""
        mask = int(mask)
        for _ in range(max_rounds):
            if time.time() > deadline - 3:
                break
            improved = False
            # Get elements in the set
            in_set = [x for x in range(1, N + 1) if (mask >> x) & 1]
            random.shuffle(in_set)
            
            for to_remove in in_set:
                if time.time() > deadline - 3:
                    break
                new_mask = mask & ~(1 << to_remove)
                best_new_mask = new_mask
                best_size = popcount(new_mask)
                
                # Try adding each candidate
                for x in range(1, N + 1):
                    if x != to_remove and not (new_mask >> x) & 1:
                        if can_add_bitmask(x, new_mask):
                            trial = new_mask | (1 << x)
                            # Try to fill
                            trial_filled = fill_set_bitmask(trial)
                            if popcount(trial_filled) > best_size:
                                best_size = popcount(trial_filled)
                                best_new_mask = trial_filled
                
                if best_size > popcount(mask):
                    mask = best_new_mask
                    improved = True
                    break
            
            if not improved:
                break
        
        return mask
    
    def local_search_2swap(mask, max_rounds=10):
        """Try 2-out 2-in swap moves to escape deeper local minima."""
        mask = int(mask)
        for _ in range(max_rounds):
            if time.time() > deadline - 3:
                break
            improved = False
            in_set = [x for x in range(1, N + 1) if (mask >> x) & 1]
            if len(in_set) < 2:
                break
            
            # Try random pairs to remove
            random.shuffle(in_set)
            num_pairs = min(5, len(in_set) * (len(in_set) - 1) // 2)
            for pi in range(num_pairs):
                if time.time() > deadline - 3:
                    break
                i, j = pi % len(in_set), (pi + 1) % len(in_set)
                if i == j:
                    continue
                a, b = in_set[i], in_set[j]
                if a == b:
                    continue
                new_mask = mask & ~(1 << a) & ~(1 << b)
                best_new_mask = new_mask
                best_size = popcount(new_mask)
                
                # Try adding pairs of candidates
                candidates = [x for x in range(1, N + 1) if not (new_mask >> x) & 1 and x != a and x != b]
                random.shuffle(candidates)
                candidates = candidates[:30]  # Limit for speed
                
                for ci in range(len(candidates)):
                    if time.time() > deadline - 3:
                        break
                    x = candidates[ci]
                    if not can_add_bitmask(x, new_mask):
                        continue
                    trial1 = new_mask | (1 << x)
                    for cj in range(ci + 1, len(candidates)):
                        y = candidates[cj]
                        if not can_add_bitmask(y, trial1):
                            continue
                        trial2 = trial1 | (1 << y)
                        trial_filled = fill_set_bitmask(trial2)
                        if popcount(trial_filled) > best_size:
                            best_size = popcount(trial_filled)
                            best_new_mask = trial_filled
                
                if best_size > popcount(mask):
                    mask = best_new_mask
                    improved = True
                    break
            
            if not improved:
                break
        
        return mask
    
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Start from odd numbers
    odds_mask = 0
    for x in range(1, N + 1):
        if x % 2 == 1:
            odds_mask |= (1 << x)
    odds_mask = fill_set_bitmask(odds_mask)
    odds_mask = local_search_bitmask(odds_mask, max_rounds=15)
    if popcount(odds_mask) > best_size:
        best_size = popcount(odds_mask)
        best_mask = odds_mask
    
    # Strategy 2: Start from elements with fewest conflicts
    sorted_by_conflicts = sorted(range(1, N + 1), key=lambda x: bin(conflict_mask[x]).count('1'))
    S2_mask = 0
    for x in sorted_by_conflicts:
        if can_add_bitmask(x, S2_mask):
            S2_mask |= (1 << x)
    S2_mask = fill_set_bitmask(S2_mask)
    S2_mask = local_search_bitmask(S2_mask, max_rounds=15)
    if popcount(S2_mask) > best_size:
        best_size = popcount(S2_mask)
        best_mask = S2_mask
    
    # Multiple randomized greedy passes with local search
    num_starts = 500
    for start in range(num_starts):
        if time.time() > deadline - 15:
            break
        
        # Randomize the order with bias toward low-conflict elements
        order = elements[:]
        random.shuffle(order)
        
        S_mask = 0
        for x in order:
            if can_add_bitmask(x, S_mask):
                S_mask |= (1 << x)
        
        # Fill
        S_mask = fill_set_bitmask(S_mask)
        
        # Local search: try to improve
        if time.time() > deadline - 8:
            if popcount(S_mask) > best_size:
                best_size = popcount(S_mask)
                best_mask = S_mask
            break
        
        S_mask = local_search_bitmask(S_mask, max_rounds=10)
        
        # Occasionally try 2-swap
        if start % 5 == 0 and time.time() < deadline - 5:
            S_mask = local_search_2swap(S_mask, max_rounds=5)
        
        if popcount(S_mask) > best_size:
            best_size = popcount(S_mask)
            best_mask = S_mask
    
    # Final: try to add any remaining elements
    if time.time() < deadline - 2:
        for x in range(1, N + 1):
            if not (best_mask >> x) & 1 and can_add_bitmask(x, best_mask):
                best_mask |= (1 << x)
    
    if best_mask == 0:
        best_mask = 1 << 1
    
    return [x for x in range(1, N + 1) if (best_mask >> x) & 1]
# EVOLVE-BLOCK-END
