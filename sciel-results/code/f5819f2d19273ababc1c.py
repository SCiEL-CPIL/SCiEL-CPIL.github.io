import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Precompute is_clique and is_independent for all masks
    # mask is an integer from 0 to 2^n - 1
    num_masks = 1 << n
    is_clique = [False] * num_masks
    is_indep = [False] * num_masks
    alpha = [0] * num_masks
    omega = [0] * num_masks
    
    # 0 is both clique and independent
    is_clique[0] = True
    is_indep[0] = True
    
    valid_sets = []
    
    for mask in range(1, num_masks):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        # A set is a clique if rest is a clique and v is adjacent to all in rest
        if is_clique[rest]:
            # Check if v is adjacent to all in rest
            is_c = True
            r = rest
            while r:
                lsb_r = r & -r
                u = lsb_r.bit_length() - 1
                if not adj[v][u]:
                    is_c = False
                    break
                r ^= lsb_r
            if is_c:
                is_clique[mask] = True
                omega[mask] = omega[rest] + 1
            else:
                omega[mask] = omega[rest] # Not strictly correct for non-clique, but we only care about max clique in subset?
                # Actually, for LB we need max clique size IN the mask.
                # If mask is not a clique, omega[mask] should be max(omega[rest], max clique in rest not containing v?)
                # This precomputation of omega for all masks is tricky if we just do DP on subsets.
                # Let's compute omega and alpha separately or use a simpler LB.
                pass
        
        # Check independent
        # A set is independent if rest is independent and v is NOT adjacent to all in rest
        if is_indep[rest]:
            is_i = True
            r = rest
            while r:
                lsb_r = r & -r
                u = lsb_r.bit_length() - 1
                if adj[v][u]:
                    is_i = False
                    break
                r ^= lsb_r
            if is_i:
                is_indep[mask] = True
                alpha[mask] = alpha[rest] + 1
            else:
                pass

        if is_clique[mask] or is_indep[mask]:
            valid_sets.append(mask)

    # We need accurate alpha[mask] and omega[mask] for the LB.
    # Let's compute them properly.
    # alpha[mask] = size of max independent set in G[mask]
    # omega[mask] = size of max clique in G[mask]
    
    # This DP approach above only calculates alpha/omega if the mask ITSELF is an indep/clique.
    # We need the max size.
    # Given n=14, we can compute alpha/omega for all masks using standard DP:
    # alpha[mask] = max(alpha[mask without v], 1 + alpha[mask without (v and neighbors of v)])
    # This is complex to implement quickly.
    
    # Simpler LB: 
    # LB(mask) = ceil(popcount(mask) / max_alpha_omega(mask))
    # Where max_alpha_omega(mask) is the max size of an independent set or clique in G[mask].
    # We can approximate this by precomputing for each mask the max size of a clique/subset that is a clique?
    # Actually, the largest clique in G[mask] is the max omega[sub] for sub subset of mask where is_clique[sub].
    # Same for alpha.
    
    # Since 2^14 is small, let's just compute max_clique_size[mask] and max_ind_set_size[mask]
    # by iterating all submasks? No, that's 3^n.
    
    # Alternative: 
    # The LB `ceil(|mask| / max(omega(mask), alpha(mask)))` is valid if we know the true max clique/ind set size.
    # Let's compute them properly.
    
    max_clique_size = [0] * num_masks
    max_ind_set_size = [0] * num_masks
    
    for mask in range(1, num_masks):
        # We can use the property:
        # max_clique_size[mask] = max(max_clique_size[mask ^ (1<<v)], max_clique_size[sub] where sub is clique containing v?)
        # Standard DP for Max Clique is exponential.
        
        # Given the small size, maybe we can just use a weaker LB?
        # LB = ceil(popcount(mask) / 7) ? No.
        
        # Let's compute max_clique_size and max_ind_set_size using a simple recursion or iterative DP.
        # For n=14, we can afford a slightly more expensive precomputation.
        
        # Let's just store the valid sets and use a simpler LB during search:
        # If remaining vertices R cannot be covered by remaining parts k, prune.
        # A part can cover at most 7 vertices? (Max clique/ind set size in this graph).
        # Let's find the global max clique and max ind set size.
        
        global_max_clique = 0
        global_max_ind = 0
        for mask in valid_sets:
            cnt = mask.bit_count()
            if is_clique[mask]:
                if cnt > global_max_clique: global_max_clique = cnt
            if is_indep[mask]:
                if cnt > global_max_ind: global_max_ind = cnt
            
    max_part_size = max(global_max_clique, global_max_ind)
    
    def min_parts_lb(mask):
        if mask == 0: return 0
        # Weak LB: ceil(popcount / max_part_size)
        return (mask.bit_count() + max_part_size - 1) // max_part_size

    # Stronger LB: 
    # We can precompute for each mask the max size of a clique/subset in it.
    # Let's do it.
    # max_clique_in[mask]
    max_clique_in = [0] * num_masks
    max_ind_in = [0] * num_masks
    
    for mask in range(1, num_masks):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        # Option 1: v not in max clique
        # Option 2: v in max clique -> 1 + max clique in (rest & neighbors(v))
        # But we don't have neighbors(v) mask precomputed? We can.
        
    # Let's precompute neighbor masks
    neigh_mask = [0] * n
    for u in range(n):
        m = 0
        for v in range(n):
            if u != v and adj[u][v]:
                m |= (1 << v)
        neigh_mask[u] = m
        
    for mask in range(1, num_masks):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Max clique in mask is max( max_clique_in[rest], 1 + max_clique_in[rest & neigh_mask[v]] )
        # But this assumes we know max_clique_in for smaller masks.
        # This DP works if we iterate masks in increasing order of bit count?
        # Yes, rest and rest & neigh_mask[v] are strictly smaller than mask.
        
        max_c1 = max_clique_in[rest]
        max_c2 = 1 + max_clique_in[rest & neigh_mask[v]]
        max_clique_in[mask] = max(max_c1, max_c2)
        
        # Max independent set in mask
        # v in max ind set -> 1 + max ind set in (rest & ~neigh_mask[v])
        non_neigh = rest & (~neigh_mask[v] & (num_masks - 1)) # Mask off higher bits
        max_i1 = max_ind_in[rest]
        max_i2 = 1 + max_ind_in[non_neigh]
        max_ind_in[mask] = max(max_i1, max_i2)
        
    def strong_lb(mask):
        if mask == 0: return 0
        max_size = max(max_clique_in[mask], max_ind_in[mask])
        if max_size == 0: return 0
        return (mask.bit_count() + max_size - 1) // max_size

    # Sort valid_sets to try larger sets first? Or sets containing the first vertex?
    # We will filter in the loop.
    
    # To speed up, we can group valid_sets by the smallest vertex they contain?
    # Or just iterate.
    
    # We want to find the minimum k.
    # Start with LB of the whole graph.
    full_mask = (1 << n) - 1
    start_k = strong_lb(full_mask)
    
    # Upper bound: n (each vertex is a part)
    # Or greedy result?
    # Let's just iterate k from start_k to n.
    
    for k in range(start_k, n + 1):
        # DFS
        # State: uncovered_mask, parts_remaining
        # We pick the smallest vertex in uncovered_mask.
        
        # To avoid recursion depth issues (max depth n=14), recursion is fine.
        # We need to track the actual partition to return it.
        
        best_partition = None
        
        def dfs(uncovered_mask, parts_left, current_partition):
            nonlocal best_partition
            if best_partition is not None:
                return True # Found a solution
            
            if uncovered_mask == 0:
                if parts_left == 0:
                    best_partition = current_partition
                    return True
                else:
                    return False # Should not happen if we stop at 0
            
            if parts_left == 0:
                return False
            
            # LB Pruning
            if strong_lb(uncovered_mask) > parts_left:
                return False
            
            # Pick smallest vertex
            lsb = uncovered_mask & -uncovered_mask
            v = lsb.bit_length() - 1
            
            # Try all valid sets that contain v and are subset of uncovered_mask
            # Optimization: Precompute list of valid sets containing each vertex?
            # valid_sets_by_vertex[v]
            
            for mask in valid_sets:
                if (mask & (1 << v)) == 0:
                    continue
                if (mask & uncovered_mask) != mask:
                    continue
                
                # Try this set
                new_uncovered = uncovered_mask ^ mask
                if dfs(new_uncovered, parts_left - 1, current_partition + [mask]):
                    return True
            
            return False
        
        # Precompute valid_sets_by_vertex for speed
        valid_sets_by_vertex = [[] for _ in range(n)]
        for mask in valid_sets:
            # Find all vertices in mask
            m = mask
            while m:
                lsb_m = m & -m
                v_m = lsb_m.bit_length() - 1
                valid_sets_by_vertex[v_m].append(mask)
                m ^= lsb_m
        
        # Re-define dfs to use the precomputed lists
        def dfs_opt(uncovered_mask, parts_left, current_partition):
            nonlocal best_partition
            if best_partition is not None:
                return True
            
            if uncovered_mask == 0:
                if parts_left == 0:
                    best_partition = current_partition
                    return True
                return False
            
            if parts_left == 0:
                return False
            
            if strong_lb(uncovered_mask) > parts_left:
                return False
            
            lsb = uncovered_mask & -uncovered_mask
            v = lsb.bit_length() - 1
            
            # Only try sets that contain v
            for mask in valid_sets_by_vertex[v]:
                if (mask & uncovered_mask) != mask:
                    continue
                
                new_uncovered = uncovered_mask ^ mask
                if dfs_opt(new_uncovered, parts_left - 1, current_partition + [mask]):
                    return True
            
            return False

        if dfs_opt(full_mask, k, []):
            # Convert masks to lists of vertices
            result = []
            for mask in best_partition:
                verts = []
                m = mask
                while m:
                    lsb_m = m & -m
                    v_m = lsb_m.bit_length() - 1
                    verts.append(v_m)
                    m ^= lsb_m
                result.append(verts)
            return result

    # Fallback: return greedy or something valid
    # Should not reach here
    return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
