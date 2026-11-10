import math
from math import gcd
import time

def construct():
    """
    Finds the maximum subset S ⊆ [1, 255] containing 255 such that gcd(a,b) > 1 
    for all distinct pairs a,b in S.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # 1. Generate candidates C
    # x must be coprime to 255? No, gcd(x, 255) > 1.
    # 255 = 3 * 5 * 17.
    # x is divisible by 3, 5, or 17.
    C = [x for x in range(1, 256) if gcd(x, 255) > 1]
    
    # We must include 255.
    # The problem reduces to finding the maximum clique in the graph induced by C \ {255},
    # then adding 255.
    # Note: 255 is connected to all nodes in C by definition.
    
    V = [x for x in C if x != 255]
    n = len(V)
    
    if n == 0:
        return [255]

    # Map values to indices
    val_to_idx = {v: i for i, v in enumerate(V)}
    
    # Build adjacency bitmasks
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    for i in range(n):
        vi = V[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            if gcd(vi, V[j]) > 1:
                mask |= (1 << j)
        adj[i] = mask

    # Order vertices by degree (descending) to improve B&B performance
    # We need to remap adjacencies if we reorder.
    degrees = [adj[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    
    # Create new adjacency list based on new order
    # new_idx maps old_idx to new_idx
    new_idx = [0] * n
    for new_i, old_i in enumerate(order):
        new_idx[old_i] = new_i
        
    new_adj = [0] * n
    for new_i, old_i in enumerate(order):
        old_mask = adj[old_i]
        new_mask = 0
        # Convert bits from old indices to new indices
        m = old_mask
        while m:
            lsb = m & -m
            old_j = lsb.bit_length() - 1
            new_j = new_idx[old_j]
            new_mask |= (1 << new_j)
            m ^= lsb
        new_adj[new_i] = new_mask

    # Update V to match new order
    V_ordered = [V[old_i] for old_i in order]

    # Greedy initialization to get a good lower bound
    best_size = 0
    best_clique = []
    
    # Simple greedy: pick vertex with max degree, add to clique, remove it and its non-neighbors?
    # Or just a random/greedy heuristic.
    # Let's try a simple greedy: repeatedly pick the vertex with highest degree in remaining candidates.
    curr_clique = []
    curr_cand = (1 << n) - 1
    while curr_cand:
        # Find vertex with max degree in curr_cand
        max_deg = -1
        best_v = -1
        m = curr_cand
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (new_adj[v] & curr_cand).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            m ^= lsb
        if best_v == -1:
            break
        curr_clique.append(best_v)
        curr_cand = curr_cand & new_adj[best_v]
        curr_cand &= ~(1 << best_v)
        
    if len(curr_clique) > best_size:
        best_size = len(curr_clique)
        best_clique = curr_clique[:]

    # Tomita's Maximum Clique Algorithm with Pivot Selection
    # We use a stack or recursion. Recursion is fine for n ~ 100-200 if depth is controlled, 
    # but n here is roughly:
    # Multiples of 3: 85
    # Multiples of 5: 51
    # Multiples of 17: 15
    # Union: 85 + 51 - 28 (mult 15) + 15 - 5 (mult 51) - 3 (mult 17*3=51? no, 17*3=51 is in 3 and 17)
    # Let's calculate C size properly.
    # |A| = 85, |B| = 51, |D| = 15 (for 17)
    # |A U B U D| = |A| + |B| + |D| - |AB| - |AD| - |BD| + |ABD|
    # AB (15): 17
    # AD (51): 5
    # BD (85): 3 (85, 170, 255) -> 255 is removed, so 2 in V? 
    # Wait, 255 is removed from V.
    # Multiples of 85 in [1, 255]: 85, 170, 255.
    # In V (excluding 255): 85, 170.
    # ABD (425): None in [1, 255].
    # So |C| = 85 + 51 + 15 - 17 - 5 - 3 + 0 = 126.
    # |V| = 125.
    # n = 125. Recursion depth max 125 is safe.

    def expand(clique, candidates):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return

        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return

        # Pruning
        if len(clique) + candidates.bit_count() <= best_size:
            return

        # Pivot selection
        # Choose v in candidates with max degree in candidates
        pivot_v = -1
        max_common_neighbors = -1
        
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            common = (new_adj[v] & candidates).bit_count()
            if common > max_common_neighbors:
                max_common_neighbors = common
                pivot_v = v
            c ^= lsb

        # Branch on candidates \ N(pivot_v)
        # Note: N(pivot_v) in the context of the subgraph induced by candidates
        candidates_to_branch = candidates & ~new_adj[pivot_v]
        
        # Iterate through bits in candidates_to_branch
        c = candidates_to_branch
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            
            new_clique = clique + [v]
            new_candidates = candidates & new_adj[v]
            
            expand(new_clique, new_candidates)
            
            # Remove v from candidates
            candidates ^= lsb
            
            # Pruning check
            if len(clique) + candidates.bit_count() <= best_size:
                break
            c ^= lsb

    # Start with empty clique and all V as candidates
    initial_candidates = (1 << n) - 1
    expand([], initial_candidates)

    # Construct final set
    result_values = [V_ordered[i] for i in best_clique]
    result_values.append(255)
    
    # Verification
    s = set(result_values)
    assert 255 in s
    assert len(s) == len(result_values)
    for a in s:
        assert 1 <= a <= 255
    for i in range(len(result_values)):
        for j in range(i + 1, len(result_values)):
            assert gcd(result_values[i], result_values[j]) > 1
            
    return result_values

# EVOLVE-BLOCK-END
