"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = set()
    for e in P['edges']:
        edges.add((e[0], e[1]))
        edges.add((e[1], e[0]))
    
    # Precompute adjacency as bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute: for each subset (bitmask), is it a clique? is it an independent set?
    # We use a bytearray for memory efficiency and speed
    is_clique = bytearray(1 << n)
    is_indep = bytearray(1 << n)
    
    # Empty set is both
    is_clique[0] = 1
    is_indep[0] = 1
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        
        # Check clique: v must be adjacent to all in rest, and rest must be clique
        if is_clique[rest]:
            if (adj[v] & rest) == rest:
                is_clique[mask] = 1
        
        # Check independent set: v must NOT be adjacent to any in rest, and rest must be independent
        if is_indep[rest]:
            if (adj[v] & rest) == 0:
                is_indep[mask] = 1

    from functools import lru_cache
    
    full_mask = (1 << n) - 1
    
    @lru_cache(maxsize=None)
    def solve(remaining):
        if remaining == 0:
            return 0
        
        # Find the lowest set bit in remaining
        lb = remaining & (-remaining)
        v = lb.bit_length() - 1
        rest_mask = remaining ^ lb
        
        best = n # worst case
        
        # Iterate through all submasks of rest_mask
        # mask = sub | lb
        sub = rest_mask
        while True:
            mask = sub | lb
            if is_clique[mask] or is_indep[mask]:
                new_remaining = remaining ^ mask
                count = 1 + solve(new_remaining)
                if count < best:
                    best = count
                    if best == 1:
                        break
            if sub == 0:
                break
            sub = (sub - 1) & rest_mask
            
        return best

    min_parts = solve(full_mask)
    
    # Now reconstruct the actual partition
    def reconstruct(remaining, current_parts):
        if remaining == 0:
            return current_parts
        
        lb = remaining & (-remaining)
        v = lb.bit_length() - 1
        rest_mask = remaining ^ lb
        
        target_cost = solve(remaining)
        
        sub = rest_mask
        while True:
            mask = sub | lb
            if is_clique[mask] or is_indep[mask]:
                new_remaining = remaining ^ mask
                if 1 + solve(new_remaining) == target_cost:
                    # Extract vertices from mask
                    vertices = []
                    for i in range(n):
                        if mask & (1 << i):
                            vertices.append(i)
                    return reconstruct(new_remaining, current_parts + [vertices])
            if sub == 0:
                break
            sub = (sub - 1) & rest_mask
            
        return current_parts

    partition = reconstruct(full_mask, [])
    
    # Convert to list of lists of ints (ensure Python int, not numpy)
    result = []
    for part in partition:
        result.append([int(v) for v in part])
    
    return result

# EVOLVE-BLOCK-END
