import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    def try_build(n, min_deg_req, seed, time_limit):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        # c[u][v] is the number of common neighbors of u and v
        c = [[0] * n for _ in range(n)]
        
        def add_edge(u, v):
            if u == v or v in adj[u]:
                return False
            if c[u][v] > 0:
                return False
            
            # Update adjacency
            adj[u].add(v)
            adj[v].add(u)
            
            # Update common neighbor counts
            nu = adj[u]
            nv = adj[v]
            
            # For x in N(u), c[x][v] increases by 1 (u is common neighbor of x and v)
            for x in nu:
                if x != v:
                    c[x][v] += 1
                    c[v][x] += 1
                    
            # For y in N(v), c[u][y] increases by 1 (v is common neighbor of u and y)
            for y in nv:
                if y != u:
                    c[u][y] += 1
                    c[y][u] += 1
                    
            # For x in N(u) and y in N(v), c[x][y] increases by 1 
            # (u is neighbor of x, v is neighbor of y? No.
            # Wait, the common neighbor of x and y is not necessarily u or v.
            # Let's re-verify the update rule.
            # c[x][y] counts z such that x~z and y~z.
            # If we add (u,v), does u or v become a common neighbor of x and y?
            # u is a common neighbor of x and y if x~u and y~u.
            # v is a common neighbor of x and y if x~v and y~v.
            # So if x is in N(u) and y is in N(u), then u is a common neighbor of x and y.
            # But we are adding (u,v).
            # The new edge is (u,v).
            # Does this change c[x][y] for x in N(u), y in N(v)?
            # No. The existence of edge (u,v) does not make u or v a common neighbor of x and y unless x and y are already connected to u and v respectively.
            # Actually, the standard update for c when adding (u,v) is:
            # 1. For all x in N(u), c[x][v] += 1.
            # 2. For all y in N(v), c[u][y] += 1.
            # 3. For all x in N(u) and y in N(v), c[x][y] += 1? 
            # Let's check. 
            # c[x][y] is # of common neighbors.
            # Before adding (u,v), u and v were not connected.
            # After adding (u,v), is there a new common neighbor for x and y?
            # No, the edge (u,v) itself is not a vertex.
            # The neighbors of x are N(x). The neighbors of y are N(y).
            # The intersection N(x) ∩ N(y) changes only if u or v enters the intersection.
            # u enters N(x) ∩ N(y) if x~u and y~u.
            # v enters N(x) ∩ N(y) if x~v and y~v.
            # So if x in N(u) and y in N(u), u is a common neighbor. This was true before.
            # The addition of (u,v) does NOT change the set of common neighbors of any pair (x,y) 
            # EXCEPT pairs involving u or v.
            # Let's verify.
            # Common neighbors of x and y: Z = {z | x~z and y~z}.
            # If we add (u,v), the edges are (u,v) and (v,u).
            # Does u become a common neighbor of x and y? Only if x~u and y~u. This condition does not depend on (u,v) existing.
            # Does v become a common neighbor of x and y? Only if x~v and y~v. This condition does not depend on (u,v) existing.
            # So c[x][y] for x,y not in {u,v} does NOT change.
            # 
            # So the update is ONLY:
            # 1. c[x][v] += 1 for all x in N(u) (x != v)
            # 2. c[u][y] += 1 for all y in N(v) (y != u)
            # 
            # Wait, what about c[u][v]? It stays 0.
            # 
            # So my previous code was correct, except I added a triple loop which was unnecessary and wrong.
            pass
            
            return True
            
        def min_degree():
            return min(len(adj[i]) for i in range(n))
            
        # Initial check
        if min_degree() >= min_deg_req:
            return True
            
        stuck_count = 0
        max_stuck = 50
        
        while time.time() - start_time < start_time + time_limit:
            md = min_degree()
            if md >= min_deg_req:
                return True
                
            # Find vertex with min degree
            target = min(range(n), key=lambda i: len(adj[i]))
            
            # Candidates: non-neighbors of target
            candidates = [v for v in range(n) if v != target and v not in adj[target]]
            
            if not candidates:
                # Complete graph on target's component? No, just no candidates.
                # This means target is connected to everyone.
                # If min degree is still low, we are stuck.
                break
            
            # Shuffle to add randomness
            rng.shuffle(candidates)
            
            # Sort candidates by their degree (ascending) to prefer connecting to low-degree nodes
            # This helps balance the degrees
            candidates.sort(key=lambda v: len(adj[v]))
            
            # Try to add an edge
            added = False
            for v in candidates:
                if c[target][v] == 0:
                    add_edge(target, v)
                    added = True
                    stuck_count = 0
                    break
            
            if not added:
                stuck_count += 1
                if stuck_count > max_stuck:
                    # Try to remove a random edge to unblock
                    # Find an edge not incident to target
                    removable_edges = [(u, v) for u in range(n) for v in adj[u] if u < v and u != target and v != target]
                    if removable_edges:
                        u, v = rng.choice(removable_edges)
                        # Remove edge (u,v)
                        adj[u].discard(v)
                        adj[v].discard(u)
                        
                        # Update c
                        nu = adj[u]
                        nv = adj[v]
                        
                        # Reverse of add_edge update
                        for x in nu:
                            if x != v:
                                c[x][v] -= 1
                                c[v][x] -= 1
                                
                        for y in nv:
                            if y != u:
                                c[u][y] -= 1
                                c[y][u] -= 1
                                
                        # c[u][v] is still 0? 
                        # Before removal, u and v were adjacent. c[u][v] was 0.
                        # After removal, they are not adjacent. c[u][v] should still be 0?
                        # Common neighbors of u and v: z such that u~z and v~z.
                        # The removal of (u,v) does not change N(u) \ {v} or N(v) \ {u}.
                        # So common neighbors don't change.
                        # So c[u][v] remains 0.
                        
                        stuck_count = 0
                    else:
                        break
                        
        # Final check
        return min_degree() >= min_deg_req

    best_result = None
    best_n = 0
    
    # Try n from 47 down to 40
    # We prioritize higher n.
    for n in range(47, 39, -1):
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        # Time budget for each n
        # If we succeed with n=47, we return immediately.
        # We allocate a reasonable time for each attempt.
        time_budget = 10.0
        
        # Multiple restarts
        attempts = 0
        while time.time() - start_time < start_time + time_budget:
            seed = 12345 + n * 1000 + attempts
            if try_build(n, min_deg_req, seed, 5.0):
                # Construct edge list
                adj = [set() for _ in range(n)]
                # Re-run to get the graph? No, try_build doesn't return the graph.
                # We need to modify try_build to return the graph or edges.
                # Let's refactor try_build to return edges if successful.
                pass
            attempts += 1
            
    # The above logic is flawed because try_build doesn't return the graph.
    # I will rewrite the main loop to use a function that returns the graph.
    
    def build_graph(n, min_deg_req, seed, time_limit):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        c = [[0] * n for _ in range(n)]
        
        def add_edge(u, v):
            if u == v or v in adj[u]:
                return False
            if c[u][v] > 0:
                return False
            
            adj[u].add(v)
            adj[v].add(u)
            
            for x in adj[u]:
                if x != v:
                    c[x][v] += 1
                    c[v][x] += 1
            for y in adj[v]:
                if y != u:
                    c[u][y] += 1
                    c[y][u] += 1
            return True
            
        def min_degree():
            return min(len(adj[i]) for i in range(n))
            
        if min_degree() >= min_deg_req:
            return adj
            
        stuck_count = 0
        max_stuck = 100
        
        while time.time() < time_limit:
            md = min_degree()
            if md >= min_deg_req:
                return adj
                
            target = min(range(n), key=lambda i: len(adj[i]))
            candidates = [v for v in range(n) if v != target and v not in adj[target]]
            
            if not candidates:
                break
                
            # Prefer low degree candidates
            candidates.sort(key=lambda v: len(adj[v]))
            
            added = False
            for v in candidates:
                if c[target][v] == 0:
                    add_edge(target, v)
                    added = True
                    stuck_count = 0
                    break
            
            if not added:
                stuck_count += 1
                if stuck_count > max_stuck:
                    removable_edges = [(u, v) for u in range(n) for v in adj[u] if u < v and u != target and v != target]
                    if removable_edges:
                        u, v = rng.choice(removable_edges)
                        adj[u].discard(v)
                        adj[v].discard(u)
                        
                        for x in adj[u]:
                            if x != v:
                                c[x][v] -= 1
                                c[v][x] -= 1
                        for y in adj[v]:
                            if y != u:
                                c[u][y] -= 1
                                c[y][u] -= 1
                                
                        stuck_count = 0
                    else:
                        break
                        
        if min_degree() >= min_deg_req:
            return adj
        return None

    for n in range(47, 39, -1):
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        # Allocate time based on n
        # 47 is hardest, give it more time? Or try all quickly?
        # Let's try each n with a few seconds.
        time_per_n = 10.0
        start_n = time.time()
        
        found = False
        seed = 0
        while time.time() - start_n < time_per_n and not found:
            adj = build_graph(n, min_deg_req, seed, time.time() - start_n + 1.0)
            if adj is not None:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            seed += 1
            found = True # Break out if time is up or we found it
            
    # Fallback
    return {'n': 3, 'edges': [[0,1],[1,2],[0,2]]}

# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    def build_graph(n, min_deg_req, seed, time_limit):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        c = [[0] * n for _ in range(n)]
        
        def add_edge(u, v):
            if u == v or v in adj[u]:
                return False
            if c[u][v] > 0:
                return False
            
            adj[u].add(v)
            adj[v].add(u)
            
            for x in adj[u]:
                if x != v:
                    c[x][v] += 1
                    c[v][x] += 1
            for y in adj[v]:
                if y != u:
                    c[u][y] += 1
                    c[y][u] += 1
            return True
            
        def min_degree():
            return min(len(adj[i]) for i in range(n))
            
        if min_degree() >= min_deg_req:
            return adj
            
        stuck_count = 0
        max_stuck = 100
        
        while time.time() < time_limit:
            md = min_degree()
            if md >= min_deg_req:
                return adj
                
            target = min(range(n), key=lambda i: len(adj[i]))
            candidates = [v for v in range(n) if v != target and v not in adj[target]]
            
            if not candidates:
                break
                
            # Prefer low degree candidates to balance
            candidates.sort(key=lambda v: len(adj[v]))
            
            added = False
            for v in candidates:
                if c[target][v] == 0:
                    add_edge(target, v)
                    added = True
                    stuck_count = 0
                    break
            
            if not added:
                stuck_count += 1
                if stuck_count > max_stuck:
                    removable_edges = [(u, v) for u in range(n) for v in adj[u] if u < v and u != target and v != target]
                    if removable_edges:
                        u, v = rng.choice(removable_edges)
                        adj[u].discard(v)
                        adj[v].discard(u)
                        
                        for x in adj[u]:
                            if x != v:
                                c[x][v] -= 1
                                c[v][x] -= 1
                        for y in adj[v]:
                            if y != u:
                                c[u][y] -= 1
                                c[y][u] -= 1
                                
                        stuck_count = 0
                    else:
                        break
                        
        if min_degree() >= min_deg_req:
            return adj
        return None

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        # Allocate time for this n
        # We have ~145s total.
        # Let's try 47 with 30s, 46 with 30s, etc.
        time_budget = 30.0
        start_n = time.time()
        
        seed = 0
        while time.time() - start_n < time_budget:
            if time.time() > deadline:
                break
            adj = build_graph(n, min_deg_req, seed, min(deadline, time.time() + 5.0))
            if adj is not None:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            seed += 1
            # If we've spent enough time on this n, move to next
            if time.time() - start_n > time_budget:
                break
                
    # Fallback: return a valid small graph
    # n=3, min_deg_req = 1. K3 is C4-free.
    return {'n': 3, 'edges': [[0,1],[1,2],[0,2]]}

# EVOLVE-BLOCK-END
