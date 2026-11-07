"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    es = set()
    for e in P['edges']:
        es.add((e[0], e[1]))
        es.add((e[1], e[0]))
    
    # Build adjacency bitmasks for fast clique/independent checks
    adj_mask = [0] * n
    for u, v in es:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute all maximal cliques and maximal independent sets
    all_cliques = set()
    all_independent = set()
    
    def is_clique_mask(mask):
        """Check if vertices in mask form a clique using bitmasks."""
        m = mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            others = mask ^ low
            if others & ~adj_mask[v]:
                return False
            m ^= low
        return True
    
    def is_independent_mask(mask):
        """Check if vertices in mask form an independent set using bitmasks."""
        m = mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            others = mask ^ low
            if others & adj_mask[v]:
                return False
            m ^= low
        return True
    
    # Enumerate all subsets to find valid parts (cliques or independent sets)
    # For n=14, 2^14 = 16384 subsets, which is very manageable
    valid_part_masks = set()
    for mask in range(1, 1 << n):
        if is_clique_mask(mask) or is_independent_mask(mask):
            valid_part_masks.add(mask)
    
    # Memoization for the partition problem
    memo = {}
    deadline = time.time() + 75  # Safety margin
    
    def min_parts(remaining_mask):
        """Find minimum number of parts to cover remaining_mask."""
        if time.time() > deadline:
            return None
        if remaining_mask == 0:
            return 0
        if remaining_mask in memo:
            return memo[remaining_mask]
        
        best = None
        # Try each valid part that is a subset of remaining_mask
        # Optimization: only try parts that contain the lowest set bit
        low_bit = remaining_mask & (-remaining_mask)
        
        # Iterate through valid parts that contain the low bit
        for part_mask in valid_part_masks:
            if (part_mask & remaining_mask) == part_mask and (part_mask & low_bit):
                sub = min_parts(remaining_mask ^ part_mask)
                if sub is not None:
                    cand = 1 + sub
                    if best is None or cand < best:
                        best = cand
                    if best == 1:
                        break
        
        if best is not None:
            memo[remaining_mask] = best
        return best
    
    # Try to find the optimal partition
    full_mask = (1 << n) - 1
    optimal_count = min_parts(full_mask)
    
    if optimal_count is not None:
        # Now reconstruct the actual partition
        def reconstruct(remaining_mask, target_count):
            """Reconstruct a partition of remaining_mask into target_count parts."""
            if remaining_mask == 0:
                return []
            
            low_bit = remaining_mask & (-remaining_mask)
            
            for part_mask in valid_part_masks:
                if (part_mask & remaining_mask) == part_mask and (part_mask & low_bit):
                    sub = reconstruct(remaining_mask ^ part_mask, target_count - 1)
                    if sub is not None:
                        # Convert part_mask to list of vertices
                        part_list = [i for i in range(n) if (part_mask >> i) & 1]
                        return [part_list] + sub
            return None
        
        partition = reconstruct(full_mask, optimal_count)
        if partition is not None:
            # Verify the partition is valid
            all_vertices = []
            for part in partition:
                all_vertices.extend(part)
            if sorted(all_vertices) == list(range(n)):
                # Ensure all parts are nonempty lists of ints
                result = []
                for part in partition:
                    if part:
                        result.append([int(v) for v in part])
                return result
    
    # Fallback: greedy approach
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            # Check if adding v keeps part as clique or independent set
            part_mask = 0
            for u in part:
                part_mask |= (1 << u)
            
            # Check clique
            clique_ok = True
            for u in part:
                if not (adj_mask[u] >> v) & 1:
                    clique_ok = False
                    break
            if clique_ok and is_clique_mask(part_mask):
                part.append(v)
                placed = True
                break
            
            # Check independent
            indep_ok = True
            for u in part:
                if (adj_mask[u] >> v) & 1:
                    indep_ok = False
                    break
            if indep_ok and is_independent_mask(part_mask):
                part.append(v)
                placed = True
                break
        
        if not placed:
            parts.append([v])
    
    return parts
# EVOLVE-BLOCK-END
