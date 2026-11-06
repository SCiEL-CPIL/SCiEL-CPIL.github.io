import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    N = 20
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    M = len(all_edges)  # 190
    
    # For each edge (u,v), precompute the set of edges that conflict with it
    # Two edges e1=(a,b) and e2=(c,d) conflict if adding both creates a K2,3
    # We need to check if the set {e1, e2} is K2,3-free together with any other edges.
    # Actually, for the conflict graph formulation:
    # Node = potential edge (u,v)
    # Edge in conflict graph between (u,v) and (w,x) if { (u,v), (w,x) } cannot both be in a K2,3-free graph
    # This is NOT simply pairwise. The K2,3 constraint is a 3-edge constraint (two vertices with 3 common neighbors).
    # So the conflict is not pairwise between edges.
    
    # Let me rethink. The K2,3-free condition: no two vertices have 3+ common neighbors.
    # This is a property of the entire graph, not pairwise edge conflicts.
    # So the MCIS formulation on a conflict graph doesn't directly work because the constraint is not pairwise.
    
    # Alternative approach: Use local search with edge addition and removal.
    # Start with a dense seed, then try to add edges, and if stuck, remove edges to create space.
    
    # Better: Use a different structural approach.
    # The complement of a K2,3-free graph: if G is K2,3-free, then for any pair of non-adjacent vertices in G,
    # they have at most 2 common neighbors in G.
    
    # Let me use a simulated annealing / local search approach with edge swaps.
    
    def common_neighbors_count(adj, u, v):
        return len(adj[u] & adj[v])
    
    def is_k23_free(adj):
        for u in range(N):
            for v in range(u + 1, N):
                if len(adj[u] & adj[v]) >= 3:
                    return False
        return True
    
    def count_edges(adj):
        return sum(len(s) for s in adj) // 2
    
    # Initial seed: 5 disjoint K4s = 5 * 6 = 30 edges
    # Better seed: Try a different structure
    # Let's try: complete bipartite K_{10,10} has 100 edges but has many K2,3
    # K2,3-free means for any two vertices, at most 2 common neighbors.
    
    # Let me use a greedy approach with multiple restarts and local search.
    
    best_edges = set()
    best_count = 0
    
    def greedy_build(order):
        edges = set()
        adj = [set() for _ in range(N)]
        
        for u, v in order:
            # Check if adding (u,v) keeps K2,3-free
            # 1. (u,v) pair: common neighbors of u and v must be <= 2
            if len(adj[u] & adj[v]) >= 3:
                continue
            # 2. For each w in N(v), check (u,w): new common neighbors
            valid = True
            for w in adj[v]:
                if w == u:
                    continue
                # After adding (u,v), v is in N(u). 
                # Common neighbors of u and w: (N(u) ∪ {v}) ∩ N(w)
                # = (N(u) ∩ N(w)) ∪ ({v} ∩ N(w))
                # v is in N(w) since w in N(v) and graph is undirected
                # So count = |N(u) ∩ N(w)| + 1 (since v not in N(u) yet)
                if len(adj[u] & adj[w]) + 1 > 2:
                    valid = False
                    break
            if not valid:
                continue
            # 3. For each w in N(u), check (v,w): new common neighbors
            for w in adj[u]:
                if w == v:
                    continue
                # After adding (u,v), u is in N(v).
                # Common neighbors of v and w: (N(v) ∪ {u}) ∩ N(w)
                # = (N(v) ∩ N(w)) ∪ ({u} ∩ N(w))
                # u is in N(w) since w in N(u)
                # So count = |N(v) ∩ N(w)| + 1
                if len(adj[v] & adj[w]) + 1 > 2:
                    valid = False
                    break
            if not valid:
                continue
            
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
        
        return edges, adj
    
    def local_search(edges, adj, max_iter=5000):
        """Try to improve by adding edges, and if stuck, do edge swaps."""
        edges = set(edges)
        adj = [set(s) for s in adj]
        
        def try_add(u, v):
            nonlocal edges, adj
            if (u, v) in edges or (v, u) in edges:
                return False
            # Ensure u < v for consistency
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                return False
            
            # Check K2,3 constraint
            if len(adj[u] & adj[v]) >= 3:
                return False
            for w in adj[v]:
                if w == u:
                    continue
                if len(adj[u] & adj[w]) + 1 > 2:
                    return False
            for w in adj[u]:
                if w == v:
                    continue
                if len(adj[v] & adj[w]) + 1 > 2:
                    return False
            
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            return True
        
        def try_remove(u, v):
            nonlocal edges, adj
            if u > v:
                u, v = v, u
            if (u, v) not in edges:
                return False
            edges.remove((u, v))
            adj[u].remove(v)
            adj[v].remove(u)
            return True
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter:
            if time.time() > deadline:
                break
            improved = False
            iteration += 1
            
            # Try to add any valid edge
            random.shuffle(all_edges)
            for u, v in all_edges:
                if (u, v) in edges:
                    continue
                if try_add(u, v):
                    improved = True
                    break
            
            if not improved:
                # Try edge swaps: remove one edge, then add up to two edges
                # Remove a random edge and try to add two new ones
                if edges:
                    remove_candidates = list(edges)
                    random.shuffle(remove_candidates)
                    for ru, rv in remove_candidates[:20]:  # Try up to 20 removals
                        if time.time() > deadline:
                            break
                        # Save state
                        edges_backup = set(edges)
                        adj_backup = [set(s) for s in adj]
                        
                        try_remove(ru, rv)
                        
                        # Try to add two edges
                        added_count = 0
                        random.shuffle(all_edges)
                        for u, v in all_edges:
                            if (u, v) in edges:
                                continue
                            if try_add(u, v):
                                added_count += 1
                                if added_count >= 2:
                                    break
                        
                        if added_count >= 2:
                            # Net gain of 1 edge
                            improved = True
                            break
                        else:
                            # Restore
                            edges = edges_backup
                            adj = adj_backup
        
        return edges, adj
    
    # Multiple restarts with different greedy orders
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        order = all_edges[:]
        if restart % 3 == 0:
            random.shuffle(order)
        elif restart % 3 == 1:
            # Prefer edges between different K4 clusters
            order.sort(key=lambda e: (e[0] // 4) != (e[1] // 4), reverse=True)
            random.shuffle(order)
        else:
            # Random with bias toward high-degree potential
            random.shuffle(order)
        
        edges, adj = greedy_build(order)
        edges, adj = local_search(edges, adj, max_iter=3000)
        
        cur_count = len(edges)
        if cur_count > best_count:
            best_count = cur_count
            best_edges = set(edges)
    
    return [list(e) for e in best_edges]
