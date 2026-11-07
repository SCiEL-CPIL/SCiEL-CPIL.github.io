import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 128
    
    # Precompute all forbidden triples: sets of 3 distinct numbers {a,b,c}
    # where a+b, a+c, b+c are all in 1..128
    # We'll use bitmasks for efficiency
    
    # For each pair (u,v) with u<v, compute which x values would create a violation
    # if x, u, v are all in the set AND x+u, x+v, u+v are all in the set
    
    # Key insight: A triple {a,b,c} is forbidden if a+b, a+c, b+c are all selected.
    # We want to avoid having any such triple where all three pair sums are also selected.
    
    # Strategy: Use bitmask representation and precompute conflict relationships
    # Then use greedy minimum-degree heuristic
    
    # Bit i represents number i+1 (so bit 0 = 1, bit 127 = 128)
    
    def mask_to_set(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]
    
    def set_to_mask(S):
        mask = 0
        for x in S:
            mask |= (1 << (x - 1))
        return mask
    
    # Precompute for each number x, the set of pairs (u,v) such that 
    # {x,u,v} would be a forbidden triple if x+u, x+v, u+v are all present
    # Actually, let's think differently.
    
    # For a given set S, adding x is invalid if there exist u,v in S (u<v)
    # such that x+u, x+v, u+v are all in S ∪ {x}
    
    # Let's precompute all possible "conflict patterns"
    # For each triple (a,b,c) with a<b<c, check if a+b, a+c, b+c are all ≤ 128
    # If so, this triple creates a constraint: we cannot have all of {a,b,c,a+b,a+c,b+c}
    
    # Actually, the constraint is: if a,b,c are selected, then at least one of 
    # {a+b, a+c, b+c} must NOT be selected
    
    # This is a hypergraph problem. Let's use a different approach:
    # Greedy with lookahead using bitmask operations
    
    # Start with a large valid set: {64, 65, ..., 128} (size 65)
    # All pair sums ≥ 129, so no pair sum is in the set. Valid!
    
    base_mask = 0
    for i in range(64, 129):
        base_mask |= (1 << (i - 1))
    
    def can_add_to_mask(mask, x):
        """Check if x can be added to the set represented by mask."""
        idx = x - 1
        if (mask >> idx) & 1:
            return False  # Already in set
        
        new_mask = mask | (1 << idx)
        
        # Check all pairs (u,v) in the set where u < v
        # For efficiency, iterate over u in 1..x-1 and v in x+1..128
        # But we need u,v both in the set
        
        # Optimization: only need to check pairs where at least one sum ≤ 128
        # If u,v ≥ 64, then u+v ≥ 129, so u+v not in new_mask. Skip such pairs.
        
        # So at least one of u,v must be < 64
        # Since x < 64 (we're only adding small numbers), and u,v are in S,
        # we need to check pairs where min(u,v) < 64
        
        # Iterate over all u in 1..127 that are in mask
        # For each u, iterate over v in u+1..128 that are in mask
        # Check if x+u ≤ 128, x+v ≤ 128, u+v ≤ 128
        # And if all three sums are in new_mask
        
        # For speed, use bit operations where possible
        
        # Collect all numbers in the set
        # This is O(N) per check, but N=128, so manageable
        
        # Actually, let's precompute the list of elements for faster iteration
        # But that requires maintaining state. For now, extract from mask.
        
        # Extract elements
        elems = []
        m = mask
        i = 0
        while m:
            if m & 1:
                elems.append(i + 1)
            m >>= 1
            i += 1
        
        # Add x to consider
        elems_new = elems + [x]
        elems_set = set(elems_new)
        
        # Check all pairs in elems (not including x yet, since we're checking if adding x creates violation)
        # Actually, we need to check pairs (u,v) where u,v are in the ORIGINAL set (mask),
        # and see if {x, u, v} forms a forbidden triple
        
        n = len(elems)
        for i in range(n):
            u = elems[i]
            for j in range(i+1, n):
                v = elems[j]
                # Check if x+u, x+v, u+v are all in new_mask
                # And all ≤ 128
                su = x + u
                sv = x + v
                uv = u + v
                if su > 128 or sv > 128 or uv > 128:
                    continue
                # Check if all three are in new_mask
                if ((new_mask >> (su-1)) & 1) and ((new_mask >> (sv-1)) & 1) and ((new_mask >> (uv-1)) & 1):
                    return False
        
        return True
    
    # Greedy: try to add numbers 1..63 in order of "least restrictive"
    # A number x is less restrictive if it conflicts with fewer potential additions
    
    # For now, try in increasing order, then random restarts
    
    best_mask = base_mask
    best_size = 65
    
    def greedy_from_order(order, initial_mask):
        """Try to add numbers in the given order."""
        mask = initial_mask
        for x in order:
            if time.time() > deadline - 5:
                break
            if can_add_to_mask(mask, x):
                mask |= (1 << (x - 1))
        return mask
    
    # Try increasing order first
    smalls = list(range(1, 64))
    mask_inc = greedy_from_order(smalls, base_mask)
    size_inc = bin(mask_inc).count('1')
    if size_inc > best_size:
        best_size = size_inc
        best_mask = mask_inc
    
    # Random restarts with shuffled orders
    random.seed(42)
    for restart in range(100):
        if time.time() > deadline - 10:
            break
        random.shuffle(smalls)
        mask_rand = greedy_from_order(smalls, base_mask)
        size_rand = bin(mask_rand).count('1')
        if size_rand > best_size:
            best_size = size_rand
            best_mask = mask_rand
    
    # Local search: try to add remaining numbers in multiple passes
    def local_search(mask, max_iter=10):
        """Try to add more numbers via iterative improvement."""
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for x in range(1, 129):
                if (mask >> (x-1)) & 1:
                    continue
                if can_add_to_mask(mask, x):
                    mask |= (1 << (x-1))
                    improved = True
            if not improved:
                break
        return mask
    
    best_mask = local_search(best_mask)
    best_size = bin(best_mask).count('1')
    
    # Verify the result
    result_set = set(mask_to_set(best_mask))
    valid = True
    elems = sorted(result_set)
    for i in range(len(elems)):
        for j in range(i+1, len(elems)):
            for k in range(j+1, len(elems)):
                a, b, c = elems[i], elems[j], elems[k]
                if (a+b) in result_set and (a+c) in result_set and (b+c) in result_set:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
        best_mask = base_mask
    
    result = [int(x) for x in mask_to_set(best_mask)]
    return result
