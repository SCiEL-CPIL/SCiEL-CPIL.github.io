"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import sys

def construct():
    n = 128
    
    # 1. Precompute LCMs
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    lcm_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            lcm_table[i][j] = lcm(i + 1, j + 1)
            lcm_table[j][i] = lcm_table[i][j]
            
    # 2. Find all forbidden triples
    # A triple (a, b, c) is forbidden if lcm(a,b) == lcm(a,c) == lcm(b,c)
    forbidden_triples = []
    for a in range(n):
        for b in range(a + 1, n):
            lab = lcm_table[a][b]
            for c in range(b + 1, n):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    forbidden_triples.append((a, b, c))
    
    # 3. Build structures for backtracking
    # For each vertex v, store the list of pairs (a, b) such that {a, b, v} is a forbidden triple.
    # We use indices 0..127.
    v_to_pairs = [[] for _ in range(n)]
    for a, b, c in forbidden_triples:
        # For vertex a, the other two are b, c
        v_to_pairs[a].append((b, c))
        # For vertex b, the other two are a, c
        v_to_pairs[b].append((a, c))
        # For vertex c, the other two are a, b
        v_to_pairs[c].append((a, b))
    
    # Calculate degree for ordering
    degrees = [len(pairs) for pairs in v_to_pairs]
    
    # Order vertices by degree descending (most constrained first)
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Map original index to position in order (not strictly needed if we iterate order, but good for bounds)
    # We just iterate through `order` list in backtracking.
    
    # 4. Backtracking with Branch and Bound
    # Use bitmask for selected set for fast intersection checks?
    # Since n=128, we can use Python ints as bitmasks.
    # selected_mask: bit i is 1 if vertex i is selected.
    
    best_size = 0
    best_mask = 0
    
    # To speed up "can add v", we check if any pair in v_to_pairs[v] is fully contained in selected_mask.
    # precompute bitmask for each pair? No, just check bits.
    
    # We can optimize the check:
    # For each v, precompute a list of (mask_a, mask_b) where mask_a = 1<<a, mask_b = 1<<b.
    # Then check: (selected_mask & mask_a) and (selected_mask & mask_b)
    
    v_to_pair_masks = [[] for _ in range(n)]
    for v in range(n):
        for a, b in v_to_pairs[v]:
            v_to_pair_masks[v].append((1 << a, 1 << b))
    
    start_time = time.time()
    timeout = 55  # Leave 5s margin
    
    def backtrack(index, selected_mask, current_size):
        nonlocal best_size, best_mask, start_time, timeout
        
        # Timeout check
        if time.time() - start_time > timeout:
            return
        
        # Pruning: Upper bound
        # Max possible size = current_size + (n - index)
        if current_size + (n - index) <= best_size:
            return
            
        if index == n:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
        
        v = order[index]
        
        # Check if v can be added
        can_add = True
        for mask_a, mask_b in v_to_pair_masks[v]:
            if (selected_mask & mask_a) and (selected_mask & mask_b):
                can_add = False
                break
        
        # Branch 1: Include v (if possible)
        if can_add:
            backtrack(index + 1, selected_mask | (1 << v), current_size + 1)
            
        # Branch 2: Exclude v
        backtrack(index + 1, selected_mask, current_size)
        
    backtrack(0, 0, 0)
    
    # Extract result from best_mask
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
