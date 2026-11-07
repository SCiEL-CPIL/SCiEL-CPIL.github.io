"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency masks for fast degree computation
_n = P['n']
_adj_masks = [0] * _n
for _u, _v in P['edges']:
    _adj_masks[_u] |= (1 << _v)
    _adj_masks[_v] |= (1 << _u)

def _induced_degrees(mask):
    """Compute the multiset of degrees within the induced subgraph on vertices in mask."""
    degs = []
    for v in range(_n):
        if (mask >> v) & 1:
            # Count neighbors of v that are also in mask
            d = bin(_adj_masks[v] & mask).count('1')
            degs.append(d)
    return degs

def _distinct_degree_count(mask):
    """Number of distinct degree values in the induced subgraph."""
    if mask == 0:
        return 0
    degs = _induced_degrees(mask)
    return len(set(degs))

def construct():
    """
    Find a subset of at least 8 vertices that maximizes the number of distinct
    vertex degrees in the induced subgraph.
    Uses bitmask branch-and-bound search.
    """
    n = _n
    full_mask = (1 << n) - 1
    
    best_count = 0
    best_mask = 0
    best_vertices = []
    
    def dfs(idx, mask, size):
        nonlocal best_count, best_mask, best_vertices
        
        # If we have processed all vertices
        if idx == n:
            if size >= 8:
                dc = _distinct_degree_count(mask)
                if dc > best_count:
                    best_count = dc
                    best_mask = mask
                    best_vertices = [v for v in range(n) if (mask >> v) & 1]
            return
        
        # Upper bound on the number of distinct degrees
        # The final set will have size `size + k` where 0 <= k <= (n - idx)
        # Max distinct degrees <= max final size = size + (n - idx)
        upper = size + (n - idx)
        
        # Prune if we cannot possibly beat the current best
        if upper <= best_count:
            return
        
        # Option 1: Include vertex idx (try to build large sets first to get good bounds early?)
        # Actually, including vertices increases size, which increases the upper bound.
        # Excluding vertices decreases size, which decreases the upper bound.
        # To find a good solution quickly, we might want to explore the "Include" branch first
        # if we think larger sets have more distinct degrees.
        # However, distinct degrees are not monotonic.
        # Let's stick to standard order: Include then Exclude, or Exclude then Include.
        # If we Include first, we reach size 8 faster.
        
        # Include vertex idx
        dfs(idx + 1, mask | (1 << idx), size + 1)
        
        # Exclude vertex idx
        dfs(idx + 1, mask, size)

    # Start DFS
    dfs(0, 0, 0)
    
    if best_count == 0:
        # Fallback: return first 8 vertices
        return list(range(8))
    
    return list(best_vertices)

# EVOLVE-BLOCK-END
