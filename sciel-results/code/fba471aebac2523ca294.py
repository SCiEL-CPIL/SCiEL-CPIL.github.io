import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0
    n = 20
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Helper to get neighbors from bitmask
    def get_neighbors(mask):
        neighbors = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            neighbors.append(idx)
            mask ^= lsb
        return neighbors

    # Main search loop
    best_edges = 0
    best_adj = [0] * n
    best_common = [[0]*n for _ in range(n)]
    
    rng = random.Random(42)
    
    for attempt in range(100000): # High limit, will be cut by time
        if time.time() > deadline:
            break
            
        # Initialize state
        adj = [0] * n
        common = [[0]*n for _ in range(n)]
        
        # Randomize edge order for greedy construction
        edges = all_edges[:]
        rng.shuffle(edges)
        
        # Greedy Phase: Try to add edges in random order
        for u, v in edges:
            if (adj[u] >> v) & 1:
                continue # Already present
            
            # Check if adding (u, v) is valid
            # 1. common[u][v] must be <= 2 (It is, if we maintain validity, but check just in case)
            if common[u][v] > 2:
                continue
                
            valid = True
            # 2. For all w in N(v), common[u][w] must be < 2
            #    Because adding (u,v) makes v a common neighbor of u and w
            #    So common[u][w] increases by 1. If it was 2, it becomes 3 (invalid).
            mask_v = adj[v]
            while mask_v:
                lsb = mask_v & -mask_v
                w = lsb.bit_length() - 1
                if common[u][w] >= 2:
                    valid = False
                    break
                mask_v ^= lsb
            if not valid:
                continue
                
            # 3. For all w in N(u), common[v][w] must be < 2
            mask_u = adj[u]
            while mask_u:
                lsb = mask_u & -mask_u
                w = lsb.bit_length() - 1
                if common[v][w] >= 2:
                    valid = False
                    break
                mask_u ^= lsb
            if not valid:
                continue
            
            # Add edge (u, v)
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            
            # Update common counts
            # For w in N(v): common[u][w] += 1
            mask_v = adj[v] & ~(1 << u) # Exclude u itself? No, u is not in N(v) before add, but now it is?
            # Wait, adj[v] was updated. We need the neighbors BEFORE adding (u,v) for the update logic?
            # Let's re-verify the update logic.
            # Before add: N(v) = V_old.
            # After add: N(v) = V_old U {u}.
            # For w in V_old:
            #   common[u][w] increases by 1 (because v is now in N(u) and v was in N(w)).
            #   common[v][w] increases by 1 (because u is now in N(v) and u was in N(w)).
            # So we should iterate over the OLD neighbors.
            
            # Let's fetch old neighbors before modifying adj?
            # I modified adj already. Let's undo or use the masks we already iterated?
            # In the check above, I iterated `mask_v` (old adj[v]) and `mask_u` (old adj[u]).
            # I can reuse those iterations or just re-extract.
            # Since N=20, re-extracting is cheap.
            
            # Re-extract old neighbors
            # Actually, I can just use the fact that I didn't store them.
            # Let's do the update carefully.
            
            # The neighbors of v BEFORE adding u are the bits in adj[v] before OR.
            # I already did adj[v] |= 1<<u.
            # So old_adj_v = adj[v] & ~(1<<u)
            # old_adj_u = adj[u] & ~(1<<v)
            
            old_adj_v = adj[v] & ~(1 << u)
            old_adj_u = adj[u] & ~(1 << v)
            
            m = old_adj_v
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                common[u][w] += 1
                common[w][u] += 1
                m ^= lsb
                
            m = old_adj_u
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                common[v][w] += 1
                common[w][v] += 1
                m ^= lsb
                # Note: common[u][v] does NOT change.
                # Wait, does common[u][v] change?
                # N(u) gained v. N(v) gained u.
                # Common neighbors of u and v are X such that X in N(u) and X in N(v).
                # X cannot be u or v.
                # The set of X is unchanged. So common[u][v] is invariant. Correct.
            
        # Local Search Phase: Try to add any remaining edge
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if common[u][v] > 2:
                    continue
                valid = True
                m = adj[v]
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if w == u: continue
                    if common[u][w] >= 2:
                        valid = False
                        break
                    m ^= lsb
                if not valid: continue
                
                m = adj[u]
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if w == v: continue
                    if common[v][w] >= 2:
                        valid = False
                        break
                    m ^= lsb
                if not valid: continue
                
                # Add edge
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                old_adj_v = adj[v] & ~(1 << u)
                old_adj_u = adj[u] & ~(1 << v)
                
                m = old_adj_v
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    common[u][w] += 1
                    common[w][u] += 1
                    m ^= lsb
                    
                m = old_adj_u
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    common[v][w] += 1
                    common[w][v] += 1
                    m ^= lsb
                improved = True
        
        # Calculate score
        edge_count = sum(bin(a).count('1') for a in adj) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj[:]
            best_common = [row[:] for row in common]
            
    # Construct result list
    res = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                res.append([u, v])
    return res
