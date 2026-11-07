import math

def construct():
    P_data = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    n = P_data['n']
    edges = P_data['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_ind
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    # Precompute max_clique_size and max_ind_size
    max_clique_size = [0] * (1 << n)
    max_ind_size = [0] * (1 << n)
    
    for m in range(1, 1 << n):
        # Get lowest set bit
        lsb = m & -m
        v_idx = lsb.bit_length() - 1
        rest = m ^ lsb
        
        is_clique[m] = is_clique[rest] and (rest & ~adj[v_idx] == 0)
        is_ind[m] = is_ind[rest] and (rest & adj[v_idx] == 0)
        
        # Max clique size DP
        # Option 1: exclude v
        # Option 2: include v, then rest must be clique in neighbors of v
        max_clique_size[m] = max(max_clique_size[rest], 1 + max_clique_size[rest & adj[v_idx]])
        
        # Max independent set size DP
        # Option 1: exclude v
        # Option 2: include v, then rest must be independent in non-neighbors of v
        non_adj = full_mask ^ adj[v_idx] ^ lsb # non-neighbors excluding v itself? 
        # Actually, rest is subset of m\{v}. We need rest to be subset of non-neighbors of v.
        # So valid part for v is rest & non_adj_v
        # non_adj_v = full_mask & ~adj[v_idx] & ~lsb
        non_adj_v = full_mask & ~adj[v_idx] & ~lsb
        max_ind_size[m] = max(max_ind_size[rest], 1 + max_ind_size[rest & non_adj_v])

    # Greedy initial solution
    def greedy():
        parts = []
        covered = 0
        while covered != full_mask:
            rem = full_mask ^ covered
            # Find largest clique or independent set in rem
            best_size = 0
            best_set = 0
            # Iterate all subsets of rem? 2^k. k up to 14.
            # We can just iterate all masks subset of rem
            s = rem
            while s:
                if (is_clique[s] or is_ind[s]) and s.bit_count() > best_size:
                    best_size = s.bit_count()
                    best_set = s
                s = (s - 1) & rem
            if best_set == 0:
                # Should not happen, at least single vertex is both
                # Find any vertex
                lsb = rem & -rem
                best_set = lsb
            parts.append(best_set)
            covered |= best_set
        return parts

    init_parts = greedy()
    best_count = len(init_parts)
    best_partition = init_parts
    
    # DFS
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(covered, parts):
        nonlocal best_count, best_partition
        if covered == full_mask:
            if len(parts) < best_count:
                best_count = len(parts)
                best_partition = parts[:]
            return
            
        rem = full_mask ^ covered
        # Lower bound
        # Max size of a valid set in rem
        max_sz = max(max_clique_size[rem], max_ind_size[rem])
        if max_sz == 0:
            return
        lb = math.ceil(rem.bit_count() / max_sz)
        if len(parts) + lb >= best_count:
            return
            
        # Find smallest uncovered vertex
        lsb = rem & -rem
        v_idx = lsb.bit_length() - 1
        
        # Generate all valid subsets S of rem containing lsb
        # S = s | lsb, where s is subset of (rem ^ lsb)
        rest = rem ^ lsb
        candidates = []
        s = rest
        while True:
            S = s | lsb
            if is_clique[S] or is_ind[S]:
                candidates.append(S)
            if s == 0:
                break
            s = (s - 1) & rest
        
        # Sort by size descending to find good solutions early
        candidates.sort(key=lambda x: x.bit_count(), reverse=True)
        
        for S in candidates:
            new_rem = rem ^ S
            # Prune based on new lower bound
            max_sz_new = max(max_clique_size[new_rem], max_ind_size[new_rem]) if new_rem else 0
            lb_new = math.ceil(new_rem.bit_count() / max_sz_new) if max_sz_new > 0 else 0
            if len(parts) + 1 + lb_new >= best_count:
                continue
            dfs(covered | S, parts + [S])

    dfs(0, [])
    
    # Convert masks back to lists of vertices
    result = []
    for mask in best_partition:
        part = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            part.append(idx)
            m ^= lsb
        result.append(part)
        
    # Ensure types are correct
    # "sequences must be list or tuple"
    # "integers must be Python int"
    return result
