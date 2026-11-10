import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Adjacency matrix as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_independent for all masks
    # is_clique[mask]: True if all vertices in mask form a clique
    # is_independent[mask]: True if all vertices in mask form an independent set
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    
    is_clique[0] = True
    is_independent[0] = True
    
    # Iterate through all masks
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = (lsb.bit_length() - 1)
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be a clique
        # v is connected to all in rest if (rest & ~adj[v]) == 0
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        else:
            is_clique[mask] = False
            
        # Check independent: v must NOT be connected to any in rest, and rest must be independent
        # v is not connected to any in rest if (rest & adj[v]) == 0
        if is_independent[rest] and (rest & adj[v]) == 0:
            is_independent[mask] = True
        else:
            is_independent[mask] = False
            
    is_valid_part = [is_clique[m] or is_independent[m] for m in range(1 << n)]
    
    # Backtracking search
    # We want to find min k such that V can be partitioned into k valid parts.
    # Symmetry breaking: The next part must contain the smallest vertex in remaining_mask.
    
    def search(remaining_mask, current_partition, k_limit):
        if remaining_mask == 0:
            return True
        
        # If we have used k_limit parts, we must have finished
        if len(current_partition) == k_limit:
            return False
            
        # Pick the smallest vertex in remaining_mask
        lsb = remaining_mask & -remaining_mask
        v = lsb.bit_length() - 1
        
        # Iterate over all subsets S of remaining_mask that contain v
        # S must be a valid part (clique or independent)
        # To ensure we don't miss any, we iterate submasks of (remaining_mask ^ lsb)
        # and add lsb to them.
        
        # Optimization: Order subsets by size? Larger parts reduce depth.
        # But generating all subsets is 2^(popcount(remaining)-1).
        # For n=14, max 2^13 = 8192. This is small.
        
        rest = remaining_mask ^ lsb
        sub = rest
        while True:
            S = sub | lsb
            if is_valid_part[S]:
                current_partition.append(S)
                if search(remaining_mask ^ S, current_partition, k_limit):
                    return True
                current_partition.pop()
            if sub == 0:
                break
            sub = (sub - 1) & rest
            
        return False

    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    # Try k = 1, 2, ...
    # Lower bound for k: ceil(n / max(alpha, omega))
    # We can just start from 1.
    
    for k in range(1, n + 1):
        if time.time() - start_time > deadline:
            # Fallback to a greedy solution if exact search takes too long
            # This should not happen for n=14
            return _greedy_construct()
        
        partition = []
        if search(full_mask, partition, k):
            # Convert masks to lists of vertices
            result = []
            for mask in partition:
                verts = []
                m = mask
                while m:
                    lsb = m & -m
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                result.append(verts)
            
            # Ensure types are correct (list of lists of ints)
            # The problem requires list or tuple. We return list of lists.
            # Integers must be Python int.
            
            # Verify validity one last time (optional but safe)
            # ...
            
            return result

    # If no solution found (should not happen), return greedy
    return _greedy_construct()

def _greedy_construct():
    """Fallback greedy constructor if exact search fails or times out."""
    n = P['n']
    es = {tuple(e) for e in P['edges']}
    es.add(tuple(reversed(e)) for e in P['edges']) # Make symmetric for quick check? No, just check both ways or use set of frozensets
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add(frozenset((u, v)))
        
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            # Check if adding v keeps part as clique or independent
            is_clique_so_far = True
            is_independent_so_far = True
            
            # Check against existing vertices
            for u in part:
                if frozenset((u, v)) in edge_set:
                    is_independent_so_far = False
                else:
                    is_clique_so_far = False
            
            if is_clique_so_far or is_independent_so_far:
                part.append(v)
                placed = True
                break
        
        if not placed:
            parts.append([v])
            
    return parts

# EVOLVE-BLOCK-END
