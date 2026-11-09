"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute: for each subset (as bitmask), whether it's a clique or independent set
    # We'll use bitmasks for subsets of vertices
    # n=14, so 2^14 = 16384 subsets, manageable
    
    def is_clique(mask):
        """Check if subset represented by mask is a clique."""
        vertices = [i for i in range(n) if mask & (1 << i)]
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if vertices[j] not in adj[vertices[i]]:
                    return False
        return True
    
    def is_independent(mask):
        """Check if subset represented by mask is an independent set."""
        vertices = [i for i in range(n) if mask & (1 << i)]
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if vertices[j] in adj[vertices[i]]:
                    return False
        return True
    
    # Precompute validity for all subsets
    valid_subset = {}
    for mask in range(1, 1 << n):
        if is_clique(mask):
            valid_subset[mask] = 'clique'
        elif is_independent(mask):
            valid_subset[mask] = 'independent'
        else:
            valid_subset[mask] = None
    
    # Find all valid subsets (cliques or independent sets)
    valid_masks = [mask for mask in range(1, 1 << n) if valid_subset[mask] is not None]
    
    # Sort by size descending to prefer larger parts
    valid_masks.sort(key=lambda m: -bin(m).count('1'))
    
    # We want to find a partition of all vertices into minimum number of valid subsets
    # This is a set cover / exact cover problem.
    # Use backtracking with branch and bound.
    
    full_mask = (1 << n) - 1
    
    best_parts = None
    best_count = n + 1
    
    def solve(remaining_mask, current_parts, start_idx):
        nonlocal best_parts, best_count
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if len(current_parts) < best_count:
                best_count = len(current_parts)
                best_parts = [list(current_parts)]
            return
        
        # Pruning: if current_parts already >= best_count, no point continuing
        if len(current_parts) >= best_count:
            return
        
        # Try adding valid subsets that cover at least one vertex in remaining_mask
        for i in range(start_idx, len(valid_masks)):
            mask = valid_masks[i]
            # Must be subset of remaining_mask
            if mask & ~remaining_mask:
                continue
            # Must intersect remaining_mask
            if mask & remaining_mask == 0:
                continue
            
            current_parts.append(mask)
            solve(remaining_mask & ~mask, current_parts, i + 1)
            current_parts.pop()
            
            if time.time() > deadline:
                return
    
    # Start with full_mask
    solve(full_mask, [], 0)
    
    if best_parts is not None:
        # Convert bitmask parts to vertex lists
        result = []
        for part_mask in best_parts[0]:
            vertices = [i for i in range(n) if part_mask & (1 << i)]
            result.append(vertices)
        return result
    
    # Fallback: greedy construction if search didn't find anything
    parts = []
    used = [False] * n
    for v in range(n):
        if used[v]:
            continue
        # Find largest valid subset containing v
        best_mask = 1 << v
        for mask in valid_masks:
            if mask & (1 << v) and (mask & ~full_mask) == 0:
                # Check if all vertices in mask are unused
                if all(not used[i] for i in range(n) if mask & (1 << i)):
                    if bin(mask).count('1') > bin(best_mask).count('1'):
                        best_mask = mask
        for i in range(n):
            if best_mask & (1 << i):
                used[i] = True
        parts.append([i for i in range(n) if best_mask & (1 << i)])
    return parts

# EVOLVE-BLOCK-END
