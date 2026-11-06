import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Target n
    target_n = 47
    min_deg_req = target_n - 39  # 8
    
    # We will try to construct a graph on target_n vertices
    # If it fails, we might try 46, but 47 is likely possible.
    
    def is_valid(n, adj):
        # Check min degree
        for i in range(n):
            if len(adj[i]) < min_deg_req:
                return False
        # Check C4-free: for all u != v, |N(u) cap N(v)| <= 1
        # Using bitmasks for speed? n=47 fits in 64 bits.
        # Let's use sets or lists.
        # For n=47, O(n^2 * deg) is fine.
        for u in range(n):
            for v in range(u + 1, n):
                # Count common neighbors
                common = 0
                # Iterate over smaller neighborhood
                if len(adj[u]) < len(adj[v]):
                    for w in adj[u]:
                        if w in adj[v]:
                            common += 1
                            if common > 1:
                                return False
                else:
                    for w in adj[v]:
                        if w in adj[u]:
                            common += 1
                            if common > 1:
                                return False
        return True

    def can_add_edge(n, adj, u, v):
        if u == v:
            return False
        if v in adj[u]:
            return False
        
        # Check 1: |N(u) cap N(v)| <= 1
        common_count = 0
        # We can just count, but we also need to check the "new common neighbor" logic for other pairs
        # Actually, the condition "for all x, if v in N(x) then N(x) cap N(u) is empty" is sufficient 
        # to ensure no new C4s are formed involving the new edge (u,v) as a side, 
        # but we must also ensure the pair (u,v) itself doesn't have >1 common neighbors.
        
        # Let's do a full check on the hypothetical graph.
        # Since n is small, we can simulate.
        
        # 1. Check pair (u, v)
        # Common neighbors of u and v in OLD graph:
        # If there are 2, adding edge doesn't change common neighbors count for pair (u,v) 
        # (as derived earlier, common neighbors remain N_old(u) cap N_old(v)).
        # Wait, if u and v are NOT connected, N(u) does not contain v.
        # After adding, N_new(u) contains v.
        # Common neighbors of u and v in NEW graph:
        # N_new(u) cap N_new(v) = (N_old(u) U {v}) cap (N_old(v) U {u})
        # = (N_old(u) cap N_old(v)) U ({v} cap N_old(v)) U (N_old(u) cap {u}) U ({v} cap {u})
        # = N_old(u) cap N_old(v).
        # So we just need |N_old(u) cap N_old(v)| <= 1.
        
        cnt = 0
        for w in adj[u]:
            if w in adj[v]:
                cnt += 1
                if cnt > 1:
                    return False
        
        # 2. Check pairs (x, u) for x in N(v)
        # If x in N(v), then v becomes a common neighbor of x and u in the new graph.
        # So we need N_old(x) cap N_old(u) to be empty.
        for x in adj[v]:
            if x == u: continue
            # Check if N(x) and N(u) have any common neighbor
            # If they do, say w, then w and v are two common neighbors of x and u.
            has_common = False
            for w in adj[x]:
                if w in adj[u]:
                    has_common = True
                    break
            if has_common:
                return False
                
        # 3. Check pairs (x, v) for x in N(u)
        # Symmetric
        for x in adj[u]:
            if x == v: continue
            has_common = False
            for w in adj[x]:
                if w in adj[v]:
                    has_common = True
                    break
            if has_common:
                return False

        return True

    def solve(n):
        # Random restarts
        for restart in range(200):
            if time.time() > deadline:
                break
                
            # Initialize empty graph
            adj = [set() for _ in range(n)]
            
            # Greedily add edges
            # Shuffle vertices to break symmetry
            vertices = list(range(n))
            random.shuffle(vertices)
            
            # We want min degree >= min_deg_req
            # We can try to add edges to satisfy this.
            
            # Strategy: 
            # 1. Try to add random edges that don't violate C4.
            # 2. If stuck, remove edges from C4s? 
            #    Actually, if we start empty, we never have C4s. 
            #    We just need to add enough edges.
            #    The constraint is "can_add_edge".
            #    If we can't add an edge to a vertex to increase its degree, we are stuck.
            
            # Let's try a greedy approach:
            # Repeat until min degree satisfied or stuck:
            #   Pick a vertex v with min degree.
            #   Try to connect v to a random other vertex u.
            #   If can_add_edge(v, u), add it.
            #   If no u works, fail this restart.
            
            max_iter = 1000
            for _ in range(max_iter):
                if time.time() > deadline:
                    return None
                    
                # Find vertex with min degree
                min_deg = min(len(adj[i]) for i in range(n))
                if min_deg >= min_deg_req:
                    return adj
                
                # Pick a random vertex with min degree
                candidates = [i for i in range(n) if len(adj[i]) == min_deg]
                v = random.choice(candidates)
                
                # Try to connect v to a random u
                # Sort potential neighbors by some heuristic? 
                # Random is fine for now.
                us = [u for u in range(n) if u != v and u not in adj[v]]
                random.shuffle(us)
                
                added = False
                for u in us:
                    if can_add_edge(n, adj, v, u):
                        adj[v].add(u)
                        adj[u].add(v)
                        added = True
                        break
                        
                if not added:
                    # Stuck. Break restart.
                    break
                    
        return None

    # Try n=47
    best_n = 0
    best_edges = []
    
    for n in [47, 46, 45]:
        if time.time() > deadline:
            break
            
        result_adj = solve(n)
        if result_adj is not None:
            # Verify
            if is_valid(n, result_adj):
                best_n = n
                best_edges = [[u, v] for u in range(n) for v in result_adj[u] if u < v]
                break
                
    # If no solution found, return the seed (n=39, empty)
    # But we should find one.
    
    if best_n == 0:
        return {'n': 39, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
