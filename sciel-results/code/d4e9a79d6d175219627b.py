"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute all K4s and Independent 4s in the host graph
    # A K4 is a set of 4 vertices where all pairs are connected
    # An Independent 4 is a set of 4 vertices where no pairs are connected
    k4_list = []
    a4_list = []
    
    # Helper to check if a 4-tuple is a K4
    def is_k4_quad(q):
        a, b, c, d = q
        return (adj_mask[a] & (1 << b)) and (adj_mask[a] & (1 << c)) and (adj_mask[a] & (1 << d)) and \
               (adj_mask[b] & (1 << c)) and (adj_mask[b] & (1 << d)) and \
               (adj_mask[c] & (1 << d))
               
    def is_a4_quad(q):
        a, b, c, d = q
        return not (adj_mask[a] & (1 << b)) and not (adj_mask[a] & (1 << c)) and not (adj_mask[a] & (1 << d)) and \
               not (adj_mask[b] & (1 << c)) and not (adj_mask[b] & (1 << d)) and \
               not (adj_mask[c] & (1 << d))

    for quad in itertools.combinations(range(n), 4):
        mask = 0
        for v in quad:
            mask |= (1 << v)
        if is_k4_quad(quad):
            k4_list.append(mask)
        if is_a4_quad(quad):
            a4_list.append(mask)
            
    # If host is Ramsey-free, k4_list and a4_list are empty
    # We iterate over all subsets of size >= min_v
    best_distinct = 0
    best_vertices = None
    
    # Iterate over all masks
    # We can iterate by size to potentially prune or just iterate all
    # Total subsets 2^17 = 131072. This is small.
    for size in range(min_v, n + 1):
        for combo in itertools.combinations(range(n), size):
            mask = 0
            for v in combo:
                mask |= (1 << v)
                
            # Check validity
            valid = True
            for k4_mask in k4_list:
                if (mask & k4_mask) == k4_mask:
                    valid = False
                    break
            if not valid:
                continue
            for a4_mask in a4_list:
                if (mask & a4_mask) == a4_mask:
                    valid = False
                    break
            if not valid:
                continue
                
            # Calculate distinct degrees
            degrees = set()
            for v in combo:
                # Count neighbors of v within the subset
                # adj_mask[v] has 1s for all neighbors in G
                # We want count of 1s in adj_mask[v] & mask
                deg = bin(adj_mask[v] & mask).count('1')
                degrees.add(deg)
                
            if len(degrees) > best_distinct:
                best_distinct = len(degrees)
                best_vertices = list(combo)
                
    if best_vertices is None:
        # Fallback, should not happen if min_v <= n and graph is valid
        return list(range(min_v))
        
    return best_vertices
# EVOLVE-BLOCK-END
