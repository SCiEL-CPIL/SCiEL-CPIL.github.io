# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency masks for fast degree computation
_n = P['n']
_adj_masks = [0] * _n
_global_degrees = [0] * _n
for _u, _v in P['edges']:
    _adj_masks[_u] |= (1 << _v)
    _adj_masks[_v] |= (1 << _u)
    _global_degrees[_u] += 1
    _global_degrees[_v] += 1

# Order vertices by global degree (descending) to find high-variance subsets earlier
# We map original vertex indices to sorted positions
# sorted_indices[i] is the original vertex index at position i in the sorted order
_sorted_indices = sorted(range(_n), key=lambda v: _global_degrees[v], reverse=True)

# Create a mapping from sorted position to original vertex
# And precompute adjacency masks in the sorted order?
# No, it's easier to keep original indices and just iterate in sorted order.
# But for bitmask recursion, it's cleaner to remap.
# Let's remap.
# new_adj_masks[i] corresponds to the vertex at sorted_indices[i]
new_adj_masks = [0] * _n
for i in range(_n):
    orig_u = _sorted_indices[i]
    mask = _adj_masks[orig_u]
    new_mask = 0
    for j in range(_n):
        orig_v = _sorted_indices[j]
        if (mask >> orig_v) & 1:
            new_mask |= (1 << j)
    new_adj_masks[i] = new_mask

def _induced_degrees_sorted(mask):
    """Compute the multiset of degrees within the induced subgraph on vertices in mask (sorted order)."""
    degs = []
    for v in range(_n):
        if (mask >> v) & 1:
            d = (new_adj_masks[v] & mask).bit_count()
            degs.append(d)
    return degs

def _distinct_degree_count_sorted(mask):
    """Number of distinct degree values in the induced subgraph (sorted order)."""
    if mask == 0:
        return 0
    degs = _induced_degrees_sorted(mask)
    return len(set(degs))

def construct():
    """
    Find a subset of at least 8 vertices that maximizes the number of distinct
    vertex degrees in the induced subgraph.
    Uses bitmask branch-and-bound search with vertex ordering by degree.
    """
    n = _n
    best_count = 0
    best_mask = 0
    best_vertices_sorted = [] # In sorted order

    # DFS over sorted vertices
    def dfs(idx, mask, size):
        nonlocal best_count, best_mask, best_vertices_sorted
        
        # Prune: if even adding all remaining vertices can't beat best
        # Max final size is size + (n - idx)
        # Max distinct degrees <= final size
        if size + (n - idx) <= best_count:
            return
        
        if idx == n:
            if size >= 8:
                dc = _distinct_degree_count_sorted(mask)
                if dc > best_count:
                    best_count = dc
                    best_mask = mask
                    best_vertices_sorted = [v for v in range(n) if (mask >> v) & 1]
            return
        
        # Option 1: include vertex idx
        new_mask = mask | (1 << idx)
        dfs(idx + 1, new_mask, size + 1)
        
        # Option 2: exclude vertex idx
        dfs(idx + 1, mask, size)
        
    dfs(0, 0, 0)
    
    if best_count == 0:
        # Fallback: return first 8 vertices (in original order)
        return list(range(8))
    
    # Map back to original vertex indices
    original_vertices = [_sorted_indices[v] for v in best_vertices_sorted]
    return original_vertices

# EVOLVE-BLOCK-END
