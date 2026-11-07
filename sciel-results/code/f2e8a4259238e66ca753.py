"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute clique and independent set checks for all masks using DP
    full_mask = (1 << n) - 1
    
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        low = mask & (-mask)
        v = low.bit_length() - 1
        rest = mask ^ low
        
        if is_clique[rest] and (rest & ~adj_mask[v]) == 0:
            is_clique[mask] = True
        
        if is_indep[rest] and (rest & adj_mask[v]) == 0:
            is_indep[mask] = True
    
    # A mask is a valid part if it's a clique or independent set
    is_valid = [False] * (1 << n)
    for mask in range(1 << n):
        is_valid[mask] = is_clique[mask] or is_indep[mask]
    
    # Greedy upper bound: assign vertices one by one to existing parts or create new
    def greedy_upper_bound():
        parts = []  # list of (type, mask) where type is 'clique' or 'indep'
        for v in range(n):
            placed = False
            for i, (ptype, pmask) in enumerate(parts):
                trial = pmask | (1 << v)
                if ptype == 'clique' and is_clique[trial]:
                    parts[i] = ('clique', trial)
                    placed = True
                    break
                elif ptype == 'indep' and is_indep[trial]:
                    parts[i] = ('indep', trial)
                    placed = True
                    break
            if not placed:
                # Try to find best type for new part
                # Check if v can start a clique or indep with remaining
                parts.append(('clique', 1 << v))
        return len(parts)
    
    ub = greedy_upper_bound()
    
    # DP with memoization
    dp = [999] * (1 << n)
    dp[0] = 0
    best_sub = [0] * (1 << n)
    
    deadline = time.time() + 85  # Safety margin before 90s limit
    timeout = [False]
    
    def solve(mask):
        if timeout[0]:
            return 999
        if time.time() > deadline:
            timeout[0] = True
            return 999
        if dp[mask] != 999:
            return dp[mask]
        
        best = ub + 1  # Start with a value worse than our upper bound
        best_s = 0
        
        # Pick lowest set bit to break symmetry
        low = mask & (-mask)
        
        # Enumerate submasks containing 'low'
        sub = mask
        while sub:
            if timeout[0]:
                return 999
            if sub & low:
                if is_valid[sub]:
                    rest = mask ^ sub
                    # Prune: if rest needs at least best-1 parts, skip
                    # Use a quick lower bound: at least 1 part needed if rest != 0
                    if rest == 0:
                        val = 1
                    else:
                        # Lower bound: need at least 1 more part
                        # Better: use dp[rest] if already computed
                        if dp[rest] != 999:
                            val = 1 + dp[rest]
                        else:
                            # Quick estimate: popcount(rest) / max_part_size
                            # For now, just recurse
                            val = 1 + solve(rest)
                    if val < best:
                        best = val
                        best_s = sub
                        if best == 1:
                            break  # Can't do better than 1
            sub = (sub - 1) & mask
        
        dp[mask] = best
        best_sub[mask] = best_s
        return best
    
    # Solve for the full mask
    full = (1 << n) - 1
    result_count = solve(full)
    
    # Fallback to greedy if timeout or no solution found
    if result_count >= 999:
        parts = []
        part_masks = []
        part_types = []
        for v in range(n):
            placed = False
            for i, pm in enumerate(part_masks):
                trial = pm | (1 << v)
                if part_types[i] == 'clique' and is_clique[trial]:
                    part_masks[i] = trial
                    placed = True
                    break
                elif part_types[i] == 'indep' and is_indep[trial]:
                    part_masks[i] = trial
                    placed = True
                    break
            if not placed:
                # Determine type: try clique first
                part_masks.append(1 << v)
                part_types.append('clique')
        
        result = []
        for pm in part_masks:
            verts = []
            m = pm
            while m:
                low = m & (-m)
                v = low.bit_length() - 1
                verts.append(int(v))
                m ^= low
            if verts:
                result.append(verts)
        return result
    
    # Reconstruct the partition
    partition_masks = []
    mask = full
    while mask:
        sub = best_sub[mask]
        if sub == 0:
            break
        partition_masks.append(sub)
        mask ^= sub
    
    # Convert masks to lists of vertices
    result = []
    for pm in partition_masks:
        verts = []
        m = pm
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            verts.append(int(v))
            m ^= low
        if verts:
            result.append(verts)
    
    # Verify
    all_vertices = []
    for part in result:
        all_vertices.extend(part)
    if sorted(all_vertices) != list(range(n)) or len(result) == 0:
        # Fallback to greedy
        parts = []
        part_masks = []
        part_types = []
        for v in range(n):
            placed = False
            for i, pm in enumerate(part_masks):
                trial = pm | (1 << v)
                if part_types[i] == 'clique' and is_clique[trial]:
                    part_masks[i] = trial
                    placed = True
                    break
                elif part_types[i] == 'indep' and is_indep[trial]:
                    part_masks[i] = trial
                    placed = True
                    break
            if not placed:
                part_masks.append(1 << v)
                part_types.append('clique')
        
        result = []
        for pm in part_masks:
            verts = []
            m = pm
            while m:
                low = m & (-m)
                v = low.bit_length() - 1
                verts.append(int(v))
                m ^= low
            if verts:
                result.append(verts)
        return result
    
    return result
# EVOLVE-BLOCK-END
