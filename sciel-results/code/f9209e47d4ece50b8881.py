"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def construct():
    """Return a nonempty subset of [1,22] with all distinct reciprocal subset sums, maximizing cardinality."""
    candidates = list(range(1, 23))
    # Sort candidates by value descending (1, 2, 3...) to potentially reduce collisions early?
    # Actually, let's just use the natural order 1..22.
    # v[1] is largest.
    
    vals = {a: _L // a for a in candidates}
    
    best_subset = [1]
    best_size = 1
    
    # Order candidates. 
    # Trying larger v first might be better?
    # Let's stick to 1..22 for now.
    
    def backtrack(current_subset, current_sums, remaining_indices):
        nonlocal best_subset, best_size
        
        # Pruning: if we can't beat best_size, return
        if len(current_subset) + len(remaining_indices) <= best_size:
            return
            
        if len(current_subset) > best_size:
            best_size = len(current_subset)
            best_subset = current_subset[:]
            
        if not remaining_indices:
            return
            
        for i, idx in enumerate(remaining_indices):
            cand = candidates[idx]
            v = vals[cand]
            
            # Check collision
            # Is there s in current_sums such that s + v in current_sums?
            collision = False
            # Optimization: 
            # If min(current_sums) + v > max(current_sums), no collision?
            # No, s1 + v = s2. s2 > s1.
            # So we need s2 - s1 = v.
            # We can just iterate.
            
            # To speed up, we can check if v is in {s2 - s1 | s1, s2 in current_sums}
            # But iterating current_sums is O(N).
            
            for s in current_sums:
                if s + v in current_sums:
                    collision = True
                    break
                    
            if collision:
                continue
                
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + v)
                
            backtrack(current_subset + [cand], new_sums, remaining_indices[i+1:])
            
    backtrack([], {0}, list(range(len(candidates))))
    return best_subset
# EVOLVE-BLOCK-END
