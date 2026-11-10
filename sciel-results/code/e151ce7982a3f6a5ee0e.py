import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m = len(all_edges) # 120
    
    # Precompute Q3 conflicts for each edge
    # conflict_mask[i] is a bitmask of edges j such that adding edge j to a graph 
    # containing edge i creates a Q3.
    # This is not quite right. Conflict depends on the context (other edges present).
    # However, we can precompute which sets of 8 vertices form a Q3.
    # A Q3 is a specific subgraph. If the graph contains all edges of a Q3, it's invalid.
    # So we are looking for a set of edges that does NOT contain the edge set of any Q3.
    # This is the Independent Set problem in a hypergraph, or equivalently,
    # we can view it as finding a set of edges with no "forbidden" subset.
    
    # Let's stick to the greedy + local search but optimize the check.
    # The current `creates_q3` is O(k^4) or worse.
    # We can precompute for each pair of edges (u,v) and (a,b) if they can be part of a Q3?
    # Actually, the most expensive part is the search for e and f.
    
    # Let's try a different approach:
    # 1. Generate a random initial graph (greedy).
    # 2. Use simulated annealing or tabu search to improve it.
    # Moves: Add edge, Remove edge, Swap edge.
    # Objective: Maximize number of edges subject to no Q3.
    
    # To speed up `creates_q3`, we can precompute common neighbors?
    # No, common neighbors change.
    
    # Optimization of creates_q3:
    # Instead of iterating all a, b, c, d, we can iterate over potential Q3s.
    # How many Q3s are there in K16?
    # A Q3 is determined by 8 vertices. C(16,8) = 12870.
    # For each set of 8 vertices, how many ways to form a Q3?
    # The Q3 is the cube graph. It has 12 edges.
    # So for each set of 8 vertices, there are multiple ways to label them as a cube.
    # The number of isomorphic copies of Q3 in K8 is 120 (automorphisms of Q3? No, embeddings).
    # Actually, it's easier to just check if a specific set of 12 edges exists.
    
    # Let's precompute all Q3 edge sets.
    # There are 12870 sets of 8 vertices.
    # For each set, we can check if it contains a Q3.
    # But we need to know WHICH edges form the Q3 to mark them as conflicting.
    
    # Alternative:
    # Precompute for every triple of edges if they form a "corner" of a Q3?
    
    # Let's optimize the current `creates_q3` first.
    # The inner loops are the bottleneck.
    
    def creates_q3(u, v, adj):
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        
        # Get list of neighbors
        # If degree is low, this is fast.
        
        # Iterate a in mask_u
        mask_a = mask_u
        while mask_a:
            lsb_a = mask_a & -mask_a
            a = lsb_a.bit_length() - 1
            mask_a ^= lsb_a
            
            # Iterate b in mask_u excluding a
            mask_b = mask_u & ~lsb_a
            while mask_b:
                lsb_b = mask_b & -mask_b
                b = lsb_b.bit_length() - 1
                mask_b ^= lsb_b
                
                # c in N(v) \ {u} AND N(a)
                mask_c = mask_v & adj[a]
                # d in N(v) \ {u} AND N(b)
                mask_d = mask_v & adj[b]
                
                # We need (a,c) and (b,d) to be edges.
                # And (a,e), (b,e) and (c,f), (d,f) and (e,f).
                
                # Optimization:
                # If mask_c is empty or mask_d is empty, skip.
                if not mask_c or not mask_d:
                    continue
                
                # Iterate c in mask_c
                mask_c_iter = mask_c
                while mask_c_iter:
                    lsb_c = mask_c_iter & -mask_c_iter
                    c = lsb_c.bit_length() - 1
                    mask_c_iter ^= lsb_c
                    
                    if c == a or c == b:
                        continue
                        
                    # d in mask_d, d != c
                    mask_d_iter = mask_d & ~lsb_c
                    while mask_d_iter:
                        lsb_d = mask_d_iter & -mask_d_iter
                        d = lsb_d.bit_length() - 1
                        mask_d_iter ^= lsb_d
                        
                        if d == a or d == b or d == c:
                            continue
                        
                        # Check e, f
                        # e in N(a) \cap N(b) \ {u,v,a,b,c,d}
                        # f in N(c) \cap N(d) \ {u,v,a,b,c,d}
                        # e != f, (e,f) edge.
                        
                        # Precompute intersection?
                        # adj[a] & adj[b] is fast.
                        mask_e = (adj[a] & adj[b]) & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d))
                        if mask_e == 0:
                            continue
                            
                        mask_f = (adj[c] & adj[d]) & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d))
                        if mask_f == 0:
                            continue
                        
                        # Check if there is e in mask_e and f in mask_f such that e!=f and (e,f) edge.
                        # This is equivalent to:
                        # Does mask_e intersect with (union of N(f) for f in mask_f)?
                        # Or simpler: iterate e, check if mask_f & adj[e] & ~lsb_e is non-empty.
                        
                        # Optimization:
                        # If mask_e has only 1 bit, check that one.
                        # If mask_f has only 1 bit, check that one.
                        
                        # General case:
                        # We want to know if there exists e in mask_e, f in mask_f s.t. e!=f and adj[e] has bit f.
                        
                        # Let's try to iterate the smaller set.
                        if mask_e.bit_count() < mask_f.bit_count():
                            mask_e_iter = mask_e
                            while mask_e_iter:
                                lsb_e = mask_e_iter & -mask_e_iter
                                e = lsb_e.bit_length() - 1
                                mask_e_iter ^= lsb_e
                                if mask_f & adj[e] & ~lsb_e:
                                    return True
                        else:
                            mask_f_iter = mask_f
                            while mask_f_iter:
                                lsb_f = mask_f_iter & -mask_f_iter
                                f = lsb_f.bit_length() - 1
                                mask_f_iter ^= lsb_f
                                if mask_e & adj[f] & ~lsb_f:
                                    return True
        return False

    best_edges = []
    best_count = 0
    
    # Greedy construction
    def greedy_construct():
        adj = [0] * n
        edges = []
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        return adj, edges

    # Local Search
    def local_search(adj, edges):
        improved = True
        while improved:
            improved = False
            # Try to add edges
            missing_edges = [e for e in all_edges if not ((adj[e[0]] >> e[1]) & 1)]
            random.shuffle(missing_edges)
            
            for u, v in missing_edges:
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
                    improved = True
            # Try to swap edges?
            # Remove an edge, try to add 2?
            # For now, just adding is safe.
        return adj, edges

    # Simulated Annealing / Tabu
    def tabu_search(adj, edges):
        current_adj = adj[:]
        current_edges = edges[:]
        current_score = len(current_edges)
        best_adj = current_adj[:]
        best_edges = current_edges[:]
        best_score = current_score
        
        tabu_list = []
        tabu_tenure = 10
        
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Generate move: Remove a random edge from current_edges
            if not current_edges:
                break
            idx = random.randint(0, len(current_edges) - 1)
            u_rem, v_rem = current_edges[idx]
            
            # Check tabu
            if (u_rem, v_rem) in tabu_list:
                continue
                
            # Remove
            current_adj[u_rem] &= ~(1 << v_rem)
            current_adj[v_rem] &= ~(1 << u_rem)
            current_edges.pop(idx)
            
            # Try to add as many edges as possible
            missing = [e for e in all_edges if not ((current_adj[e[0]] >> e[1]) & 1)]
            random.shuffle(missing)
            
            added = []
            for u, v in missing:
                if not creates_q3(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.append((u, v))
                    added.append((u, v))
            
            new_score = len(current_edges)
            
            # Accept if better or random (annealing)
            # Simple hill climbing for now
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_adj = current_adj[:]
                    best_edges = current_edges[:]
            else:
                # Revert
                # This is messy. Let's just restart if we get stuck?
                # Or simple random restarts.
                pass
            
            # Update tabu
            tabu_list.append((u_rem, v_rem))
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
                
        return best_adj, best_edges

    while time.time() < deadline:
        adj, edges = greedy_construct()
        adj, edges = local_search(adj, edges)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
        # Run a short tabu search
        if time.time() < deadline - 5:
            _, edges_tabu = tabu_search(adj, edges)
            if len(edges_tabu) > best_count:
                best_count = len(edges_tabu)
                best_edges = edges_tabu[:]

    return [[u, v] for u, v in best_edges]
