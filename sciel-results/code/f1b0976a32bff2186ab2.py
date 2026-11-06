import random
import time

def construct():
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_set = set(all_edges)
    
    def can_add(u, v, adj):
        mask_u = adj[u]
        mask_v = adj[v]
        
        # Check if adding (u,v) creates a K2,3
        # A K2,3 is formed if there exist x, y such that 
        # N(x) and N(y) have >= 3 common neighbors.
        # Adding (u,v) increases N(u) by v and N(v) by u.
        
        # Pairs (u, w) for w in N(v): 
        # Common neighbors of u and w are (N(u) \ {v}) \cap N(w) plus v? No.
        # N(u) becomes N(u) U {v}.
        # N(w) is unchanged.
        # Common neighbors of u and w:
        # Previously: N(u) \cap N(w).
        # Now: (N(u) \cap N(w)) U ({v} if v in N(w) else empty).
        # So if v is in N(w), the count increases by 1.
        # We need to ensure count < 3.
        
        # So for each w in N(v):
        #   if v in N(w) (always true since graph is simple undirected, wait. w is in N(v) means (v,w) is an edge. So v is in N(w)).
        #   So the common neighbors of u and w increase by 1 if u and w already shared v? No.
        #   u and w share v as a neighbor ONLY IF v is connected to both u and w.
        #   We are adding (u,v). So u is connected to v.
        #   If w is connected to v (which it is, since w in N(v)), then v is a common neighbor of u and w.
        #   So for each w in N(v), the number of common neighbors between u and w increases by 1 (specifically, v becomes a common neighbor).
        #   Wait, is v already a common neighbor of u and w?
        #   Before adding (u,v), u was NOT connected to v. So v was NOT a neighbor of u.
        #   So v was NOT a common neighbor of u and w.
        #   So adding (u,v) adds exactly 1 common neighbor (v) to the pair (u, w) for all w in N(v).
        #   Similarly, for all w in N(u), the pair (v, w) gains 1 common neighbor (u).
        
        # So we must check:
        # 1. For all w in N(v): (adj[u] & adj[w]).bit_count() + 1 < 3  => (adj[u] & adj[w]).bit_count() < 2
        #    Wait, adj[u] does not contain v yet.
        #    Current common neighbors of u and w: adj[u] & adj[w].
        #    New common neighbors: (adj[u] & adj[w]) U {v}.
        #    Since v is not in adj[u], v is not in (adj[u] & adj[w]).
        #    So new count = old count + 1.
        #    We need new count <= 2.
        #    So old count <= 1.
        #    Condition: for all w in N(v), (adj[u] & adj[w]).bit_count() <= 1.
        
        # 2. For all w in N(u): (adj[v] & adj[w]).bit_count() <= 1.
        
        # 3. What about pairs (x, y) where x != u, y != v?
        #    Their common neighbors don't change.
        
        # 4. What about the pair (u, v) itself?
        #    They don't have common neighbors in the sense of K2,3? 
        #    K2,3 requires two vertices with 3 common neighbors.
        #    u and v are the two vertices.
        #    Common neighbors of u and v: N(u) \cap N(v).
        #    This doesn't change when we add (u,v) because (u,v) is not a common neighbor of u and v (it's the edge between them).
        #    Wait, N(u) is the set of neighbors.
        #    If we add (u,v), v enters N(u) and u enters N(v).
        #    Does v enter N(u) \cap N(v)? No, v is not in N(v).
        #    Does u enter N(u) \cap N(v)? No, u is not in N(u).
        #    So the set of common neighbors of u and v is unchanged.
        
        # So the check is:
        # For all w in N(v): (adj[u] & adj[w]).bit_count() <= 1
        # For all w in N(u): (adj[v] & adj[w]).bit_count() <= 1
        
        w = adj[v]
        while w:
            lsb = w & -w
            w ^= lsb
            idx = lsb.bit_length() - 1
            if (adj[u] & adj[idx]).bit_count() > 1:
                return False
        
        w = adj[u]
        while w:
            lsb = w & -w
            w ^= lsb
            idx = lsb.bit_length() - 1
            if (adj[v] & adj[idx]).bit_count() > 1:
                return False
        
        return True

    def greedy_construction(order):
        adj = [0] * n
        edges = []
        for u, v in order:
            if can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        return edges, adj

    def local_search(edges, adj):
        current_edges = list(edges)
        current_adj = adj[:]
        current_count = len(current_edges)
        edge_set = set(current_edges)
        
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any missing edge
            missing = [e for e in all_edges if e not in edge_set]
            random.shuffle(missing)
            for u, v in missing:
                if can_add(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.append((u, v))
                    edge_set.add((u, v))
                    current_count += 1
                    improved = True
                    break
            
            if not improved:
                # 2. Try swaps: remove one, try to add others
                # We try to remove each existing edge and see if we can add more than 1
                # To speed up, we only try a subset or limit iterations
                swapped = False
                # Shuffle current edges to randomize which one we try to remove
                trial_edges = current_edges[:]
                random.shuffle(trial_edges)
                
                for u, v in trial_edges:
                    # Remove (u,v)
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                    current_edges.remove((u, v))
                    edge_set.remove((u, v))
                    
                    # Try to add as many as possible
                    added = 0
                    new_edges = []
                    missing_after_removal = [e for e in all_edges if e not in edge_set]
                    random.shuffle(missing_after_removal)
                    
                    for cu, cv in missing_after_removal:
                        if can_add(cu, cv, current_adj):
                            current_adj[cu] |= (1 << cv)
                            current_adj[cv] |= (1 << cu)
                            current_edges.append((cu, cv))
                            edge_set.add((cu, cv))
                            new_edges.append((cu, cv))
                            added += 1
                            if added > 1: # If we added more than we removed, it's a good swap
                                break # Stop adding for this swap attempt? Or keep going?
                                # Keeping going might find even more, but let's be greedy
                                # Actually, let's try to add ALL possible ones
                                # But that's expensive. Let's break if we already gained.
                                # Wait, if we break, we might miss a third edge.
                                # Let's just add all possible.
                            
                    # Check if total edges increased
                    if len(current_edges) > current_count:
                        current_count = len(current_edges)
                        improved = True
                        swapped = True
                        break # Restart the while loop with the new state
                    else:
                        # Revert
                        # Remove new_edges
                        for cu, cv in new_edges:
                            current_adj[cu] ^= (1 << cv)
                            current_adj[cv] ^= (1 << cu)
                            current_edges.remove((cu, cv))
                            edge_set.remove((cu, cv))
                        
                        # Re-add (u,v)
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        current_edges.append((u, v))
                        edge_set.add((u, v))
                        
                if swapped:
                    continue # Continue the while loop
                else:
                    # No swap improved the count
                    break

        return current_edges

    best_edges = []
    best_count = 0
    start_time = time.time()
    time_limit = 110.0
    
    rng = random.Random(42)
    
    # Run multiple greedy + local search
    for i in range(500):
        if time.time() - start_time > time_limit:
            break
            
        order = all_edges[:]
        rng.shuffle(order)
        
        edges, adj = greedy_construction(order)
        
        # Local search
        edges = local_search(edges, adj)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
    # Verification
    def is_valid(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() >= 3:
                    return False
        return True
        
    if not is_valid(best_edges):
        # Fallback: Empty graph or simple valid graph
        # A complete bipartite graph K(10,10) has 100 edges.
        # Common neighbors of two vertices in the same partition: 10.
        # K2,3 requires 3 common neighbors. 10 >= 3, so K(10,10) is NOT K2,3-free.
        # We need a graph with low codegree.
        # Random graph with p=0.1?
        # Let's just return empty or a known small valid graph.
        # For n=20, a cycle is valid.
        return [[i, (i+1)%20] for i in range(20)]
        
    return [[u, v] for u, v in best_edges]
