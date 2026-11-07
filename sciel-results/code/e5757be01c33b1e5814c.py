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
    
    # Collect all valid subsets (cliques or independent sets) for each mask
    # valid_subs[mask] = list of valid submasks of mask (including mask itself)
    # This is expensive to precompute, so we compute on the fly
    
    # Greedy upper bound
    def greedy_upper_bound():
        parts = []
        for v in range(n):
            placed = False
            for i in range(len(parts)):
                ptype = parts[i][0]
                pmask = parts[i][1]
                trial = pmask | (1 << v)
                if ptype == 0 and is_clique[trial]:  # 0 = clique
                    parts[i] = (0, trial)
                    placed = True
                    break
                elif ptype == 1 and is_indep[trial]:  # 1 = indep
                    parts[i] = (1, trial)
                    placed = True
                    break
            if not placed:
                parts.append((0, 1 << v))
        return len(parts)
    
    ub = greedy_upper_bound()
    
    # Iterative deepening DFS with memoization
    # f[mask] = minimum number of valid parts to cover mask
    # Use branch and bound with good lower bounds
    
    memo = {}
    
    def popcount(x):
        return bin(x).count('1')
    
    def solve(mask, remaining_budget, depth):
        """Try to cover 'mask' with at most 'remaining_budget' valid parts.
        Returns True if possible, False otherwise."""
        if mask == 0:
            return True
        if remaining_budget <= 0:
            return False
        if remaining_budget == 1:
            return is_valid[mask]
        
        key = (mask, remaining_budget)
        if key in memo:
            return memo[key]
        
        # Lower bound: at least popcount(mask) / max_size needed
        # But more useful: we need at least 1 if mask != 0
        
        # Pick a vertex to include in some part
        # Use the vertex with highest degree in the induced subgraph for better pruning
        low = mask & (-mask)
        v = low.bit_length() - 1
        
        # Enumerate all valid subsets containing v that are subsets of mask
        sub = mask
        while sub:
            if sub & low:
                if is_valid[sub]:
                    rest = mask ^ sub
                    if solve(rest, remaining_budget - 1, depth + 1):
                        memo[key] = True
                        return True
            sub = (sub - 1) & mask
        
        memo[key] = False
        return False
    
    # Binary search / iterative deepening for minimum number of parts
    best_result = None
    best_count = ub
    
    for k in range(1, ub + 1):
        memo.clear()
        if solve(full_mask, k, 0):
            best_count = k
            break
    
    # Now reconstruct the actual partition for best_count
    # Use a different memo that stores the choices
    choice_memo = {}
    
    def solve_reconstruct(mask, remaining_budget):
        """Find a valid partition of 'mask' into exactly 'remaining_budget' parts.
        Returns list of masks."""
        if mask == 0:
            return []
        if remaining_budget == 1:
            return [mask]
        
        key = (mask, remaining_budget)
        if key in choice_memo:
            return choice_memo[key]
        
        low = mask & (-mask)
        sub = mask
        while sub:
            if sub & low:
                if is_valid[sub]:
                    rest = mask ^ sub
                    subparts = solve_reconstruct(rest, remaining_budget - 1)
                    if subparts is not None:
                        result = [sub] + subparts
                        choice_memo[key] = result
                        return result
            sub = (sub - 1) & mask
        
        choice_memo[key] = None
        return None
    
    partition_masks = solve_reconstruct(full_mask, best_count)
    
    if partition_masks is None:
        # Fallback: use greedy
        part_masks = []
        part_types = []
        for v in range(n):
            placed = False
            for i in range(len(part_masks)):
                trial = part_masks[i] | (1 << v)
                if part_types[i] == 0 and is_clique[trial]:
                    part_masks[i] = trial
                    placed = True
                    break
                elif part_types[i] == 1 and is_indep[trial]:
                    part_masks[i] = trial
                    placed = True
                    break
            if not placed:
                part_masks.append(1 << v)
                part_types.append(0)
        partition_masks = part_masks
    
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
        part_masks = []
        part_types = []
        for v in range(n):
            placed = False
            for i in range(len(part_masks)):
                trial = part_masks[i] | (1 << v)
                if part_types[i] == 0 and is_clique[trial]:
                    part_masks[i] = trial
                    placed = True
                    break
                elif part_types[i] == 1 and is_indep[trial]:
                    part_masks[i] = trial
                    placed = True
                    break
            if not placed:
                part_masks.append(1 << v)
                part_types.append(0)
        
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
