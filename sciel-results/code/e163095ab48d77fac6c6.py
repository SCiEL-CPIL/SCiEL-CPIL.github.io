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
    Uses bitmask branch-and-bound search with incremental degree tracking and tight pruning.
    """
    n = _n
    
    best_count = 0
    best_mask = 0
    best_vertices = []
    
    # 1. Brute force all subsets of size 8 to get a strong initial lower bound
    from itertools import combinations
    for comb in combinations(range(n), 8):
        mask = 0
        for v in comb:
            mask |= (1 << v)
        dc = _distinct_degree_count(mask)
        if dc > best_count:
            best_count = dc
            best_mask = mask
            best_vertices = list(comb)
            
    # 2. Branch and Bound for subsets of size > 8
    # Sort vertices by degree descending to potentially find diverse structures earlier
    vertex_order = sorted(range(n), key=lambda x: bin(_adj_masks[x]).count('1'), reverse=True)
    
    def dfs(idx, mask, size, degs_list):
        nonlocal best_count, best_mask, best_vertices
        
        # Prune 1: Cannot reach minimum size 8
        if size + (n - idx) < 8:
            return
            
        # Prune 2: Cannot beat current best distinct degree count
        # Max possible distinct degrees <= Max possible final size
        if size + (n - idx) <= best_count:
            return
            
        if idx == n:
            if size >= 8:
                # Calculate distinct degrees from the maintained list
                distinct_degs = set(degs_list[v] for v in range(n) if (mask >> v) & 1)
                dc = len(distinct_degs)
                if dc > best_count:
                    best_count = dc
                    best_mask = mask
                    best_vertices = [v for v in range(n) if (mask >> v) & 1]
            return
        
        v = vertex_order[idx]
        
        # Option 1: Exclude v
        dfs(idx + 1, mask, size, degs_list)
        
        # Option 2: Include v
        new_mask = mask | (1 << v)
        
        # Update degrees incrementally
        # New degree of v is number of neighbors in current mask
        new_deg_v = bin(_adj_masks[v] & mask).count('1')
        
        # Copy degree list
        new_degs = degs_list[:]
        
        # Increment degrees of neighbors in mask
        neighbors_mask = _adj_masks[v] & mask
        u = 0
        while neighbors_mask:
            if neighbors_mask & 1:
                new_degs[u] += 1
            neighbors_mask >>= 1
            u += 1
            
        new_degs[v] = new_deg_v
        
        dfs(idx + 1, new_mask, size + 1, new_degs)
        
    # Initial state
    initial_degs = [0] * n
    dfs(0, 0, 0, initial_degs)
    
    if best_count == 0:
        return list(range(8))
        
    return list(best_vertices)

# EVOLVE-BLOCK-END
