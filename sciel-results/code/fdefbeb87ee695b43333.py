<<<<<<< SEARCH
def _has_independent_set_k(adj_bits, n, k):
    """
    Fast exact check for independent set of size k.
    Uses bitset-based branch and bound with aggressive pruning.
    """
    if k <= 0:
        return True
    if n < k:
        return False
    
    # Precompute: for each vertex, its non-neighbors mask (including itself)
    # non_neighbors[v] = vertices that are NOT adjacent to v (v can be in independent set with them)
    # Actually for independent set: if we pick v, we can only pick vertices NOT adjacent to v
    # non_adj_mask[v] = ~(adj_bits[v] | (1 << v)) & ((1 << n) - 1)
    
    full_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = full_mask & ~(adj_bits[v] | (1 << v))
    
    # Memoization cache: (rem, size_needed) -> bool
    # But rem can be large, so we use a dict with limited size
    cache = {}
    
    def search(rem, size):
        if size >= k:
            return True
        if rem == 0:
            return False
        
        # Pruning: not enough vertices
        if size + _popcount(rem) < k:
            return False
            
        # Check cache
        key = (rem, size)
        if key in cache:
            return cache[key]
            
        # Additional pruning: if the maximum independent set of the remaining graph
        # is less than what we need, prune.
        # Quick upper bound: greedy coloring of complement? Too expensive.
        # Simpler: if rem has fewer than (k - size) vertices, already handled above.
        
        # Pick branching vertex: highest degree in induced subgraph on rem
        best_v = -1
        best_deg = -1
        temp = rem
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = _popcount(adj_bits[v] & rem)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            cache[key] = False
            return False
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v -> remove v and all its neighbors
        new_rem_include = rem & non_adj[best_v]
        if search(new_rem_include, size + 1):
            cache[key] = True
            return True
            
        # Branch 2: Exclude best_v
        new_rem_exclude = rem & ~v_bit
        if search(new_rem_exclude, size):
            cache[key] = True
            return True
            
        cache[key] = False
        return False
        
    return search(full_mask, 0)
=======
def _has_independent_set_k(adj_bits, n, k):
    """
    Fast exact check for independent set of size k.
    Uses bitset-based branch and bound with aggressive pruning.
    """
    if k <= 0:
        return True
    if n < k:
        return False
    
    full_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = full_mask & ~(adj_bits[v] | (1 << v))
    
    cache = {}
    cache_hits = [0]
    cache_misses = [0]
    
    def search(rem, size):
        if size >= k:
            return True
        if rem == 0:
            return False
        
        if size + _popcount(rem) < k:
            return False
            
        key = (rem, size)
        if key in cache:
            cache_hits[0] += 1
            return cache[key]
        cache_misses[0] += 1
            
        # Pick branching vertex: highest degree in induced subgraph on rem
        best_v = -1
        best_deg = -1
        temp = rem
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = _popcount(adj_bits[v] & rem)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            cache[key] = False
            return False
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v -> remove v and all its neighbors
        new_rem_include = rem & non_adj[best_v]
        if search(new_rem_include, size + 1):
            cache[key] = True
            return True
            
        # Branch 2: Exclude best_v
        new_rem_exclude = rem & ~v_bit
        if search(new_rem_exclude, size):
            cache[key] = True
            return True
            
        cache[key] = False
        return False
        
    result = search(full_mask, 0)
    return result
>>>>>>> REPLACE

