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
    
    adj_mask = [0] * n
    for u, v in es:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute: for each vertex, which vertices it's adjacent to
    # and which it's NOT adjacent to (among all vertices)
    non_adj_mask = [0] * n
    for v in range(n):
        non_adj_mask[v] = full_mask & ~adj_mask[v] & ~(1 << v)
    
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    def is_clique(mask):
        """Check if vertices in mask form a clique."""
        m = mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            others = mask ^ low
            if others & ~adj_mask[v]:
                return False
            m ^= low
        return True
    
    def is_independent(mask):
        """Check if vertices in mask form an independent set."""
        m = mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            others = mask ^ low
            if others & adj_mask[v]:
                return False
            m ^= low
        return True
    
    def can_add_to_clique(part_mask, v):
        """Check if v can be added to a clique part."""
        if part_mask == 0:
            return True
        # All vertices in part must be adjacent to v
        return (part_mask & ~adj_mask[v]) == 0
    
    def can_add_to_independent(part_mask, v):
        """Check if v can be added to an independent set part."""
        if part_mask == 0:
            return True
        # No vertex in part should be adjacent to v
        return (part_mask & adj_mask[v]) == 0
    
    # Strategy: branch and bound
    # State: tuple of part masks (sorted for canonical form), remaining vertices mask
    # We try to assign vertices one by one, choosing which part to put them in
    
    # For efficiency, we'll use a different approach:
    # Enumerate all valid subsets (cliques and independent sets)
    # Then do set cover / partition search
    
    # Precompute all maximal cliques and maximal independent sets
    # Actually, let's just precompute all valid subsets (cliques or independent sets)
    # There are at most 2^14 = 16384 subsets, so this is feasible
    
    valid_subsets = []  # list of (mask, type) where type is 'clique' or 'independent'
    for mask in range(1, 1 << n):
        if is_clique(mask):
            valid_subsets.append((mask, 'clique'))
        elif is_independent(mask):
            valid_subsets.append((mask, 'independent'))
    
    # For the search, we'll use branch and bound:
    # At each step, pick the smallest remaining vertex (by index)
    # Try to add it to each existing part, or create a new part
    
    # Better approach: use the fact that we need to partition into minimum parts
    # This is equivalent to finding the minimum number of cliques/independent sets
    # that cover all vertices (with each vertex in exactly one part)
    
    # Let's use a recursive search with memoization
    # State: (remaining_mask, tuple_of_part_masks)
    # But this state space might be too large.
    
    # Alternative: since n=14 is small, let's try a more direct approach
    # We'll use the following insight:
    # A part can be either a clique or an independent set.
    # For each vertex, it can join a clique part if it's adjacent to all vertices in that part,
    # or join an independent set part if it's not adjacent to any vertex in that part.
    
    # Let's use iterative deepening: try k=1, k=2, ..., k=n
    # For each k, do a backtracking search
    
    def search_k(k, order):
        """Try to partition vertices in 'order' into k parts, each a clique or independent set."""
        parts_mask = [0] * k
        parts_type = [None] * k  # 'clique', 'independent', or None (empty)
        
        # Memoization: key is (idx, tuple of part masks with types)
        # But types matter, so we need to include them
        # Actually, the type of a part is determined by its contents (it's either a clique or independent set)
        # So we just need the masks, but we need to know if they're cliques or independent sets
        # to determine if we can add more vertices
        
        # For efficiency, let's not memoize on the full state, but use pruning
        
        def bt(idx):
            if time.time() > deadline:
                return None
            if idx == len(order):
                # All vertices assigned, check all parts are non-empty
                for i in range(k):
                    if parts_mask[i] == 0:
                        return None
                # Reconstruct result
                result = [[] for _ in range(k)]
                for i in range(k):
                    m = parts_mask[i]
                    while m:
                        low = m & (-m)
                        v = low.bit_length() - 1
                        result[i].append(v)
                        m ^= low
                return result
            
            v = order[idx]
            v_bit = 1 << v
            seen_empty = False
            
            for i in range(k):
                if parts_mask[i] == 0:
                    if seen_empty:
                        continue
                    seen_empty = True
                    # Try making this part a clique
                    parts_mask[i] = v_bit
                    parts_type[i] = 'clique'
                    result = bt(idx + 1)
                    if result is not None:
                        return result
                    # Try making this part an independent set
                    parts_mask[i] = v_bit
                    parts_type[i] = 'independent'
                    result = bt(idx + 1)
                    if result is not None:
                        return result
                    parts_mask[i] = 0
                    parts_type[i] = None
                else:
                    # Part is non-empty, check if v can be added
                    if parts_type[i] == 'clique':
                        if can_add_to_clique(parts_mask[i], v):
                            parts_mask[i] |= v_bit
                            result = bt(idx + 1)
                            if result is not None:
                                return result
                            parts_mask[i] ^= v_bit
                    elif parts_type[i] == 'independent':
                        if can_add_to_independent(parts_mask[i], v):
                            parts_mask[i] |= v_bit
                            result = bt(idx + 1)
                            if result is not None:
                                return result
                            parts_mask[i] ^= v_bit
            return None
        
        return bt(0)
    
    # Try different vertex orderings
    orderings = [
        list(range(n)),
        sorted(range(n), key=lambda x: -bin(adj_mask[x]).count('1')),  # decreasing degree
        sorted(range(n), key=lambda x: bin(adj_mask[x]).count('1')),   # increasing degree
    ]
    
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        for order in orderings:
            if time.time() > deadline:
                break
            result = search_k(k, order)
            if result is not None:
                # Convert to proper list of lists with int types
                final_result = []
                for part in result:
                    final_result.append([int(v) for v in part])
                return final_result
    
    # Fallback: greedy
    parts = []
    for v in range(n):
        placed = False
        for i, part in enumerate(parts):
            part_mask = 0
            for u in part:
                part_mask |= (1 << u)
            # Check if part is a clique and v can be added
            if is_clique(part_mask) and can_add_to_clique(part_mask, v):
                part.append(v)
                placed = True
                break
            # Check if part is an independent set and v can be added
            if is_independent(part_mask) and can_add_to_independent(part_mask, v):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    
    return [[int(v) for v in part] for part in parts]

# EVOLVE-BLOCK-END
