import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    n = 20
    
    def popcount(x):
        return x.bit_count()
        
    def can_add(adj, common, u, v):
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Pair (u, v) common neighbors must be <= 2
        if common[u][v] > 2:
            return False
            
        # 2. For all w in N(v), pair (u, w) gains v as common neighbor
        #    So we need: for each w in N(v) \ {u}, common[u][w] <= 1
        nv_mask = adj[v]
        # Iterate bits in nv_mask
        w = 0
        while w < n:
            if (nv_mask >> w) & 1:
                if w != u and common[u][w] > 1:
                    return False
            w += 1
            
        # 3. For all w in N(u), pair (v, w) gains u as common neighbor
        #    So we need: for each w in N(u) \ {v}, common[v][w] <= 1
        nu_mask = adj[u]
        w = 0
        while w < n:
            if (nu_mask >> w) & 1:
                if w != v and common[v][w] > 1:
                    return False
            w += 1
            
        return True

    def add_edge(adj, common, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update common counts
        # For w in N(u), common[v][w] increases by 1 (u is now common neighbor of v and w? No, u is neighbor of v and w? Yes, if w in N(u) and u in N(v). Wait.
        # Common neighbors of (v, w) are N(v) & N(w).
        # u is in N(v) (just added). Is u in N(w)?
        # If w is in N(u), then u is in N(w). So u is a common neighbor of (v, w).
        # So for all w in N(u), common[v][w] += 1.
        nu_mask = adj[u] & ~(1 << v) # Exclude v itself? N(u) doesn't contain u.
        w = 0
        while w < n:
            if (nu_mask >> w) & 1:
                common[v][w] += 1
                common[w][v] += 1
            w += 1
            
        # For w in N(v), common[u][w] increases by 1.
        nv_mask = adj[v] & ~(1 << u)
        w = 0
        while w < n:
            if (nv_mask >> w) & 1:
                common[u][w] += 1
                common[w][u] += 1
            w += 1

    def remove_edge(adj, common, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Update common counts
        nu_mask = adj[u]
        w = 0
        while w < n:
            if (nu_mask >> w) & 1:
                common[v][w] -= 1
                common[w][v] -= 1
            w += 1
            
        nv_mask = adj[v]
        w = 0
        while w < n:
            if (nv_mask >> w) & 1:
                common[u][w] -= 1
                common[w][u] -= 1
            w += 1

    def get_edge_count(adj):
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    def is_valid(common):
        for i in range(n):
            for j in range(i + 1, n):
                if common[i][j] > 2:
                    return False
        return True

    best_edges = 0
    best_adj = None
    best_common = None
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Initialize
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        
        # Random permutation for greedy order
        order = list(range(n))
        random.shuffle(order)
        
        # Greedy construction
        # Try to add all edges in random order
        # We can just iterate all pairs
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5: # Randomize order slightly or just shuffle pairs
                    pass
        
        # Better: Shuffle all pairs
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if can_add(adj, common, u, v):
                add_edge(adj, common, u, v)
                
        current_edges = get_edge_count(adj)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            best_common = [row[:] for row in common]
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding edges
            # Find all missing edges
            missing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((adj[i] >> j) & 1):
                        missing.append((i, j))
                        
            random.shuffle(missing)
            
            added_any = False
            for u, v in missing:
                if can_add(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    current_edges += 1
                    improved = True
                    added_any = True
                    # Restart missing list? No, just continue.
                    # But adding an edge might make other additions invalid.
                    # We check validity on the fly in can_add.
                    
            # If no edges added, try swaps
            if not added_any:
                # Try 1-swap: remove (u,v), add (x,y)
                # This is complex to implement efficiently.
                # Let's try random edge removal and re-addition
                existing = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            existing.append((i, j))
                            
                if existing:
                    rem_u, rem_v = random.choice(existing)
                    remove_edge(adj, common, rem_u, rem_v)
                    current_edges -= 1
                    
                    # Try to add back or add new
                    # Try adding (rem_u, rem_v) back
                    # Or try adding a random missing edge
                    # Let's just try to fill up again greedily
                    # This is effectively a restart of the greedy phase from current state
                    # To avoid O(n^3) per step, we just try a few random additions
                    for _ in range(50):
                         if time.time() > deadline: break
                         u = random.randint(0, n-1)
                         v = random.randint(0, n-1)
                         if u == v: continue
                         if u > v: u, v = v, u
                         if can_add(adj, common, u, v):
                             add_edge(adj, common, u, v)
                             current_edges += 1
                             
                    # Check if we improved
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        best_common = [row[:] for row in common]
                        
                    # If we didn't improve or get stuck, revert?
                    # For now, keep the state if it's valid.
                    # We need to ensure validity.
                    # The can_add function ensures validity.
        
    # Extract edges from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
