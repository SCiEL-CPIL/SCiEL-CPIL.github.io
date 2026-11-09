import time
import random

def construct():
    n = 20
    adj = [0] * n
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
        
    def can_add(u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
            
        # Check neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if popcount(adj[w] & adj[v]) >= 2:
                return False
            mask_u ^= lsb
            
        # Check neighbors of v
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            x = lsb.bit_length() - 1
            if popcount(adj[u] & adj[x]) >= 2:
                return False
            mask_v ^= lsb
            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def count_edges():
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2

    def get_edges_list():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def find_violation():
        for u in range(n):
            for v in range(u+1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return u, v
        return None

    def clean_graph():
        # Remove edges to resolve K2,3 violations
        # We want to be efficient.
        # A violation is (u, v) with >= 3 common neighbors.
        # We need to break the commonality.
        # Removing an edge (u, w) or (v, w) for w in common(u, v) helps.
        # Actually, removing (u, w) reduces common(u, v) by 1 if w was common.
        # It might also affect other pairs.
        
        # Strategy: Find a violation, pick a random common neighbor w,
        # and remove either (u, w) or (v, w).
        # Repeat until no violations.
        
        while True:
            viol = find_violation()
            if not viol:
                break
            u, v = viol
            common = adj[u] & adj[v]
            ws = []
            mask = common
            while mask:
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                ws.append(w)
                mask ^= lsb
            
            if not ws:
                break # Should not happen if violation found
            
            w = random.choice(ws)
            if random.random() < 0.5:
                remove_edge(u, w)
            else:
                remove_edge(v, w)

    def greedy_fill():
        # Try to add all possible missing edges
        added_any = True
        while added_any:
            added_any = False
            # Iterate pairs in random order? Or fixed?
            # Fixed order might be faster but deterministic.
            # Random order helps escape symmetry?
            # Since we are just filling, order doesn't matter for final count IF we check all.
            # But adding one might block another? No, adding edges only restricts.
            # Wait, if I add (u,v), it might make (a,b) invalid?
            # Yes. So the set of edges I can add depends on the order.
            # Greedy: pick any valid edge, add it, repeat.
            
            # To be fast, we can just iterate through all pairs and add if valid.
            # But if we add (1,2), then (3,4) becomes invalid, we skip it.
            # Next pass, maybe (3,4) is still invalid.
            # This is standard greedy maximal independent set style.
            
            # Optimization: Only check missing edges.
            for i in range(n):
                for j in range(i+1, n):
                    if not ((adj[i] >> j) & 1):
                        if can_add(i, j):
                            add_edge(i, j)
                            added_any = True

    start_time = time.time()
    deadline = start_time + 115.0 # 5s buffer
    
    best_adj = [0] * n
    best_count = 0
    
    # Multiple restarts to explore different local optima
    # The "clean_graph" approach from inspiration 1 allows starting from a dense random graph
    # and pruning it, which might lead to better structures than building up from empty.
    # Building up from empty is often stuck in "empty-ish" states or very regular structures.
    # Starting dense and pruning can find high-density K2,3-free subgraphs.
    
    while time.time() < deadline:
        # Reset graph
        for i in range(n):
            adj[i] = 0
            
        # Strategy A: Random dense start + Clean
        # Strategy B: Greedy build up
        
        # Let's mix. 50/50?
        # Or just use the Dense Start + Clean + Greedy Fill combo which seems robust.
        
        p = random.choice([0.5, 0.6, 0.7, 0.8])
        
        # Random initialization
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    add_edge(i, j)
        
        # Clean up violations
        clean_graph()
        
        # Greedy fill: add any remaining valid edges
        greedy_fill()
        
        current_count = count_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        # Local Search / Perturbation
        # If we have time, do some local moves.
        # Move: Remove a random edge, then Greedy Fill.
        # This allows escaping local optima where no single edge addition is possible,
        # but a swap (remove one, add many) might be.
        
        # We can do this a few times per restart if time permits.
        
        # Estimate time per iteration.
        # clean_graph is O(E * V^2) worst case? No, O(V^4) checks but breaks early.
        # greedy_fill is O(E * V^2).
        # With n=20, these are fast.
        
        # Let's do a few perturbation steps
        perturb_steps = 5
        for _ in range(perturb_steps):
            if time.time() > deadline:
                break
                
            # Pick a random edge to remove
            existing_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        existing_edges.append((i, j))
                    mask ^= lsb
                    
            if not existing_edges:
                break
                
            u, v = random.choice(existing_edges)
            remove_edge(u, v)
            
            # Greedy Fill again
            greedy_fill()
            
            current_count = count_edges()
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
                
            # If we didn't recover the edge or add better, maybe keep the removal?
            # The greedy fill will try to add it back if it's valid.
            # If it's not valid, we stay in the new state.
            
            # Check if we are at a local optimum (no edges can be added)
            # greedy_fill ensures this.
            
    # Return the best graph found
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges
