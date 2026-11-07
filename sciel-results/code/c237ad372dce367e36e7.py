import sys
import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute all valid parts (cliques or independent sets)
    # A set S is a clique if for all u,v in S, edge (u,v) exists
    # A set S is independent if for all u,v in S, no edge (u,v) exists
    
    all_valid_parts = []
    
    # Enumerate all subsets (2^14 = 16384, very manageable)
    for mask in range(1, 1 << n):
        # Check if clique
        is_clique = True
        # Check if independent
        is_indep = True
        
        # Extract vertices
        verts = []
        for i in range(n):
            if mask & (1 << i):
                verts.append(i)
        
        if len(verts) <= 1:
            # Single vertex is both clique and independent
            all_valid_parts.append(mask)
            continue
        
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if adj[u] & (1 << v):
                    # Edge exists: not independent
                    is_indep = False
                else:
                    # No edge: not clique
                    is_clique = False
            
            if not is_clique and not is_indep:
                break
        
        if is_clique or is_indep:
            all_valid_parts.append(mask)
    
    # Remove duplicates (shouldn't be any since each mask is unique)
    # Sort by size descending for better pruning
    all_valid_parts.sort(key=lambda m: bin(m).count('1'), reverse=True)
    
    # Full set of all vertices
    full_mask = (1 << n) - 1
    
    # Branch and bound: try to cover all vertices with k parts
    # We try k = 1, 2, 3, ... and stop at first success
    
    start_time = time.time()
    time_limit = 80.0  # Safety margin for 90s limit
    
    def can_cover(remaining_mask, k, parts_used, depth, deadline):
        """
        Try to cover remaining_mask with at most k parts from all_valid_parts.
        Returns a list of part masks if successful, else None.
        """
        if time.time() > deadline:
            return None
        
        if remaining_mask == 0:
            return parts_used
        
        if len(parts_used) >= k:
            return None
        
        # Lower bound: if remaining has more bits than k * max_part_size, impossible
        # But we can use a tighter bound: count how many distinct vertices remain
        rem_count = bin(remaining_mask).count('1')
        if rem_count == 0:
            return parts_used
        
        # Try each valid part that intersects with remaining_mask
        # Use a heuristic: try larger parts first (already sorted)
        for p in all_valid_parts:
            # Part must intersect with remaining
            if not (p & remaining_mask):
                continue
            
            new_remaining = remaining_mask & ~p
            new_parts = parts_used + [p]
            
            result = can_cover(new_remaining, k, new_parts, depth + 1, deadline)
            if result is not None:
                return result
        
        return None
    
    # Try increasing k
    for k in range(1, n + 1):
        deadline = start_time + time_limit
        result = can_cover(full_mask, k, [], 0, deadline)
        
        if result is not None:
            # Convert part masks to lists of vertex indices
            final_result = []
            for p_mask in result:
                part_vertices = []
                for i in range(n):
                    if p_mask & (1 << i):
                        part_vertices.append(i)
                final_result.append(part_vertices)
            
            # Verify each part is nonempty and covers all vertices exactly once
            seen = [0] * n
            for part in final_result:
                for v in part:
                    seen[v] += 1
            if all(c == 1 for c in seen) and all(len(p) > 0 for p in final_result):
                return final_result
    
    # Fallback: single vertices
    return [[i] for i in range(n)]
# EVOLVE-BLOCK-END
