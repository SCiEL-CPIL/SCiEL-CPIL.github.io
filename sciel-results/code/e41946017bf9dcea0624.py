"""Cochromatic partition of a 14-vertex graph - optimized search."""
import time
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute: for each subset, whether it's a clique or independent set
    # Represent subsets as bitmasks
    def is_clique(mask):
        # Check if all pairs in the subset are connected
        verts = [i for i in range(n) if (mask >> i) & 1]
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                if verts[j] not in adj[verts[i]]:
                    return False
        return True
    
    def is_independent(mask):
        # Check if no pairs in the subset are connected
        verts = [i for i in range(n) if (mask >> i) & 1]
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                if verts[j] in adj[verts[i]]:
                    return False
        return True
    
    # Precompute valid subsets (clique or independent)
    valid_subsets = {}
    for mask in range(1, 1 << n):
        if is_clique(mask) or is_independent(mask):
            valid_subsets[mask] = True
    
    # Get all valid subsets as list of vertex tuples
    valid_parts = []
    for mask in valid_subsets:
        verts = tuple(i for i in range(n) if (mask >> i) & 1)
        valid_parts.append((mask, verts))
    
    # Backtracking search for minimum cochromatic number
    start_time = time.time()
    deadline = 55.0  # seconds, with safety margin
    
    best_solution = None
    best_count = n  # worst case: each vertex its own part
    
    # Order vertices by degree (higher degree first) for better pruning
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    order_index = {v: i for i, v in enumerate(order)}
    
    def backtrack(unassigned_mask, current_parts, part_count):
        nonlocal best_solution, best_count, deadline
        
        if time.time() > deadline:
            return False
        
        if unassigned_mask == 0:
            if part_count < best_count:
                best_count = part_count
                best_solution = [list(p) for p in current_parts]
                return True
            return False
        
        # Lower bound: at least 1 more part needed
        if part_count + 1 >= best_count:
            return False
        
        # Pick the first unassigned vertex (in our ordering)
        first_unassigned = None
        for v in order:
            if (unassigned_mask >> v) & 1:
                first_unassigned = v
                break
        
        if first_unassigned is None:
            return False
        
        # Try to add first_unassigned to existing parts
        for idx, part_mask in enumerate(current_parts):
            new_mask = part_mask | (1 << first_unassigned)
            if new_mask in valid_subsets:
                current_parts[idx] = new_mask
                if backtrack(unassigned_mask & ~(1 << first_unassigned), current_parts, part_count):
                    return True
                current_parts[idx] = part_mask
        
        # Try creating a new part with first_unassigned
        new_part_mask = 1 << first_unassigned
        if new_part_mask in valid_subsets:
            current_parts.append(new_part_mask)
            if backtrack(unassigned_mask & ~(1 << first_unassigned), current_parts, part_count + 1):
                return True
            current_parts.pop()
        
        return False
    
    # Start backtracking
    backtrack((1 << n) - 1, [], 0)
    
    # If no solution found within deadline, use greedy as fallback
    if best_solution is None:
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial_mask = 0
                for u in part:
                    trial_mask |= (1 << u)
                trial_mask |= (1 << v)
                if trial_mask in valid_subsets:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        best_solution = parts
    
    # Convert to proper format: list of lists of ints
    result = []
    for part in best_solution:
        result.append([int(v) for v in part])
    
    return result
# EVOLVE-BLOCK-END
