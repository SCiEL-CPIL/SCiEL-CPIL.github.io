import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    n = 18
    best_score = 18  # Worst case
    best_edges = None
    
    # Helper to check K4-free
    def is_k4_free(adj):
        # adj is list of bitmasks
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    continue
                # Check common neighbors of i and j
                common = adj[i] & adj[j]
                # For any k in common, check if k and l are connected for l in common?
                # Actually, K4 is {i, j, k, l} all connected.
                # If i, j connected, we need to check if there are k, l in common(i,j) such that k~l.
                # Iterate k in common
                k_bits = common
                while k_bits:
                    lsb = k_bits & -k_bits
                    k = lsb.bit_length() - 1
                    k_bits ^= lsb
                    # Check if k is connected to any other vertex in common (excluding k itself, but k is in common so adj[k] has k? No, no self loops)
                    # We need l in common such that l != k and adj[k] & (1<<l)
                    # common has i and j? No, adj[i] doesn't have i. So common is neighbors of i AND neighbors of j.
                    # Does common contain i? No. j? No.
                    # So we just check if adj[k] intersects (common & ~(1<<k))
                    # Actually, if k and l are both in common, and k~l, then {i,j,k,l} is K4.
                    # So if adj[k] & (common & ~(1<<k)) is not 0, we have a K4.
                    if adj[k] & (common & ~(1 << k)):
                        return False
        return True

    # Helper to find max triangle-free induced subgraph size
    def max_tf_size(adj):
        # B&B
        best = 0
        # Precompute adj as bitmasks
        # State: S (current set), C (candidates)
        # Initial C = all vertices
        
        # Optimization: Order vertices by degree? 
        # Let's use a simple recursive B&B.
        
        def dfs(S, C):
            nonlocal best
            if len(S) + len(C) <= best:
                return
            if not C:
                if len(S) > best:
                    best = len(S)
                return
            
            # Heuristic: pick vertex with highest degree in induced subgraph on C? 
            # Or just pick first.
            # Let's pick the vertex in C that has the most neighbors in C (to reduce branching?)
            # Actually, for triangle-free, we want to be careful.
            # Let's just pick the first vertex in C for simplicity, n=18 is small.
            v = C[0]
            C_rest = C[1:]
            
            # Branch 1: Include v
            # Check if v can be added to S
            # v is in C, so it was already checked to be compatible with S?
            # Wait, C is defined as vertices that *can* be added to S.
            # So we don't need to re-check compatibility with S.
            # But we need to update C for the new S' = S U {v}.
            
            # New candidates C' from C_rest:
            # u in C_rest can be added to S' if u does not form a triangle with v and any w in S.
            # Triangle w-v-u: w in S, v in S', u in C'.
            # Condition: NOT (w~v and v~u and w~u)
            
            # Precompute neighbors of v
            v_adj = adj[v]
            
            new_C = []
            for u in C_rest:
                # Check if u is compatible with v and S
                # If u~v, we must ensure no w in S is connected to both u and v.
                if v_adj & (1 << u):
                    # u is connected to v.
                    # Check if there is w in S such that w~u and w~v.
                    # w must be in S, adj[u], and adj[v].
                    # common_uv = adj[u] & adj[v]
                    # if common_uv & S_mask: conflict
                    # We need S as bitmask for this.
                    pass # Need bitmask S
                
            # Let's refactor to use bitmasks for S and C.
            pass
        
        # Bitmask implementation
        def dfs_mask(S_mask, C_mask):
            nonlocal best
            # Popcount
            s_count = bin(S_mask).count('1')
            c_count = bin(C_mask).count('1')
            if s_count + c_count <= best:
                return
            if C_mask == 0:
                if s_count > best:
                    best = s_count
                return
            
            # Pick a vertex v from C_mask
            # Heuristic: pick vertex with max degree in C?
            # For now, pick lowest index
            lsb = C_mask & -C_mask
            v = lsb.bit_length() - 1
            C_mask_rest = C_mask ^ lsb
            
            # Branch 1: Include v
            # Check compatibility of v with S_mask
            # v is in C_mask, so it is compatible with S_mask by definition of C_mask generation?
            # Yes, if we maintain C_mask correctly.
            
            # New S_mask' = S_mask | lsb
            S_mask_prime = S_mask | lsb
            
            # Update C_mask
            # For each u in C_mask_rest, check if u is compatible with S_mask_prime.
            # u is compatible with S_mask_prime if u is compatible with S_mask (already true) 
            # AND u is compatible with v (i.e., no triangle w-v-u with w in S_mask).
            
            # Condition for conflict: exists w in S_mask such that w~v, v~u, w~u.
            # v~u is (adj[v] >> u) & 1
            # w~v and w~u means w in adj[v] & adj[u].
            # So if (adj[v] & adj[u] & S_mask) != 0, then conflict IF v~u.
            # If v not~ u, no triangle v-u-w.
            
            new_C_mask = 0
            u_mask = C_mask_rest
            while u_mask:
                u_lsb = u_mask & -u_mask
                u = u_lsb.bit_length() - 1
                u_mask ^= u_lsb
                
                # Check if u can be added
                if adj[v] & u_lsb:
                    # v and u are connected
                    # Check if any w in S_mask is connected to both
                    if adj[v] & adj[u] & S_mask:
                        # Conflict
                        continue
                # If v and u not connected, no triangle involving v and u.
                # But wait, does u need to be compatible with other vertices in S?
                # Yes, but u was in C_mask, so it was compatible with S_mask.
                # Adding v to S only creates new triangles involving v.
                # So the check above is sufficient.
                
                new_C_mask |= u_lsb
            
            dfs_mask(S_mask_prime, new_C_mask)
            
            # Branch 2: Exclude v
            dfs_mask(S_mask, C_mask_rest)
        
        # Initial C_mask is all vertices
        # But we must ensure no vertex in C is incompatible with initial S (which is empty).
        # Empty S is compatible with everyone.
        initial_C = (1 << n) - 1
        dfs_mask(0, initial_C)
        return best

    # Search Loop
    # Generate random K4-free graphs
    # Strategy: Generate random graph, remove edges from K4s until K4-free.
    
    while time.time() < deadline:
        # Generate random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Ensure K4-free
        # Find a K4 and remove an edge
        # Repeat until no K4
        # To speed up, we can just check and if not, discard?
        # Discarding might be slow if p=0.5.
        # Let's try p=0.3 or 0.4.
        
        # Better: Start with empty graph, add edges randomly, if K4 formed, reject edge?
        # This ensures K4-free.
        
        adj = [0] * n
        # Random order of edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for i, j in edges:
            # Try adding edge (i,j)
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
            # Check if this creates a K4
            # Common neighbors of i and j
            common = adj[i] & adj[j]
            # If there are two common neighbors k, l that are connected, we have K4.
            # Check if induced subgraph on common contains an edge.
            # Iterate k in common
            k_bits = common
            has_k4 = False
            while k_bits:
                lsb = k_bits & -k_bits
                k = lsb.bit_length() - 1
                k_bits ^= lsb
                # Check if k is connected to any other vertex in common
                # Other vertices in common: common & ~(1<<k)
                if adj[k] & (common & ~(1 << k)):
                    has_k4 = True
                    break
            
            if has_k4:
                # Remove edge
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
        
        # Evaluate
        score = max_tf_size(adj)
        
        if score < best_score:
            best_score = score
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
            
            # If we found a very good score, we can try to improve it locally?
            # Or just keep searching.
            if best_score <= 10:
                # We can break or continue to find even better?
                # 10 is likely the minimum.
                pass

    if best_edges is None:
        # Fallback to seed
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
        
    return best_edges

# Test locally?
# No, just output.
