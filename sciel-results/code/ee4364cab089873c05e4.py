import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = 39
    best_edges = []
    
    def try_greedy(n, seed):
        random.seed(seed)
        order = list(range(n))
        random.shuffle(order)
        
        adj = [set() for _ in range(n)]
        # D[u] is the set of vertices at distance 2 from u (neighbors of neighbors)
        # Used for fast C4 check: adding (u, v) is safe if D[u] is disjoint from adj[v] 
        # AND D[v] is disjoint from adj[u]?
        # Let's re-verify.
        # Adding (u, v).
        # New C4s must use (u, v).
        # C4: u-x-y-v-u.
        # x is neighbor of u. y is neighbor of v. (x, y) is edge.
        # So x in adj[u], y in adj[v], x in adj[y].
        # This means x is in D(v)? No.
        # x is in adj[u]. y is in adj[v].
        # If x and y are connected, then x is in adj[y] and y is in adj[x].
        # So x is in D(v) (since y in adj[v] and x in adj[y]).
        # And y is in D(u) (since x in adj[u] and y in adj[x]).
        # So if D[u] intersects D[v]? No.
        # We need to check if there exists x in adj[u] and y in adj[v] such that (x, y) is an edge.
        # This is equivalent to: (adj[u] \cap D[v]) is not empty?
        # If x in adj[u] and x in D[v], then there exists y in adj[v] such that (x, y) is edge.
        # Yes.
        # So condition: (adj[u] \cap D[v]) == empty AND (adj[v] \cap D[u]) == empty.
        # Note: D[v] includes adj[v]? No, D[v] is distance 2.
        # If x in adj[u] and x in adj[v], they are common neighbors.
        # If (x, y) is edge, and x, y in adj[v], then x is in D[v].
        # So checking adj[u] \cap D[v] covers the case where x is a common neighbor?
        # If x in adj[u] and x in adj[v], is x in D[v]?
        # D[v] = { z | exists y in adj[v], (z, y) in E }.
        # If x in adj[v], and there is some y in adj[v] connected to x, then x in D[v].
        # But if x is isolated in adj[v]?
        # In a simple graph, if x in adj[v], it's distance 1.
        # D[v] is usually defined as distance exactly 2, or distance <= 2?
        # If D[v] is distance <= 2 (closed neighborhood of neighbors), then it includes adj[v].
        # Let's define D[u] = { w | dist(u, w) == 2 }.
        # Then check:
        # 1. adj[u] \cap D[v] == empty
        # 2. adj[v] \cap D[u] == empty
        # 3. adj[u] \cap adj[v] == empty?
        # If x in adj[u] and x in adj[v], then x is a common neighbor.
        # Does this create a C4?
        # u-x-v-?-u.
        # If we add (u, v), and x is common neighbor.
        # We need another path of length 3 between u and v.
        # u-x-v is length 2.
        # If there is another common neighbor y, then u-x-v-y-u is C4.
        # So if |adj[u] \cap adj[v]| >= 2, C4 exists.
        # If |adj[u] \cap adj[v]| == 1, say x.
        # Is there a C4?
        # u-x-v.
        # Need u-y-x-v? No.
        # u-y-z-v?
        # If we add (u, v), the edge (u, v) closes the loop.
        # A C4 using (u, v) is u-v-x-y-u.
        # v-x, x-y, y-u.
        # x in adj[v], y in adj[u], (x, y) in E.
        # So x in D[u] (since y in adj[u] and x in adj[y]).
        # So we need to check if adj[v] \cap D[u] is empty.
        # This covers the case where x is a common neighbor?
        # If x in adj[v] and x in adj[u].
        # Is x in D[u]?
        # x is in adj[u]. D[u] is distance 2.
        # So x is NOT in D[u] (unless there is a triangle u-x-w-u, then x is dist 1 and 2? No, dist is min).
        # So if x is a common neighbor, it is NOT in D[u] (by definition of dist 2).
        # So the check `adj[v] \cap D[u]` does NOT catch the case where x is a common neighbor and there is another path?
        # Wait. If x is a common neighbor, we don't have a C4 yet.
        # We have a path u-x-v of length 2.
        # Adding (u, v) creates a triangle u-x-v-u.
        # This is NOT a C4.
        # So common neighbors are OK.
        # The problem is if there is a path of length 3: u-y-z-v.
        # Then u-v-z-y-u is a C4.
        # y in adj[u], z in adj[v], (y, z) in E.
        # z in D[u].
        # So we check if adj[v] \cap D[u] is empty.
        # This is correct.
        
        # So the condition to add (u, v) is:
        # (adj[u] \cap D[v]) == empty AND (adj[v] \cap D[u]) == empty.
        
        D = [set() for _ in range(n)]
        
        def update_D(a, b):
            # Add edge (a, b)
            # D[a] gains b? No, b is dist 1.
            # D[a] gains all neighbors of b (dist 2 from a via b).
            # D[b] gains all neighbors of a.
            # For any c in adj[a] \ {b}:
            #   D[c] gains b and neighbors of b.
            # For any c in adj[b] \ {a}:
            #   D[c] gains a and neighbors of a.
            
            # Optimize:
            # New nodes added to D[a]: adj[b]
            # New nodes added to D[b]: adj[a]
            
            # Update neighbors
            for c in list(adj[a]):
                if c == b: continue
                # c is neighbor of a.
                # c is now dist 2 from b.
                # c is dist 2 from neighbors of b.
                D[c].add(b)
                D[c].update(adj[b])
                
            for c in list(adj[b]):
                if c == a: continue
                D[c].add(a)
                D[c].update(adj[a])
                
            # Update a and b themselves
            D[a].update(adj[b])
            D[b].update(adj[a])
            
        # Greedy construction
        for v in order:
            # Try to connect v to all previous vertices
            # Sort candidates by degree (descending) to get high degree?
            # Or just iterate.
            # We need to maintain N_v = adj[v]
            # And check compatibility.
            
            candidates = [u for u in order[:order.index(v)] if u not in adj[v]]
            # Sort by degree descending?
            candidates.sort(key=lambda u: len(adj[u]), reverse=True)
            
            for u in candidates:
                if time.time() > deadline:
                    return None, 0
                
                # Check compatibility
                # adj[u] \cap D[v] == empty
                # adj[v] \cap D[u] == empty
                
                if adj[u].intersection(D[v]):
                    continue
                if adj[v].intersection(D[u]):
                    continue
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                update_D(u, v)
                
        # Calculate min degree
        min_deg = min(len(adj[i]) for i in range(n))
        
        # Verify C4-free (just in case)
        # This is O(n^3) but n is small.
        # Actually, the greedy algorithm guarantees it if the logic is correct.
        # But let's double check one random instance?
        # No, trust the logic.
        
        if min_deg >= n - 39:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}, min_deg
        return None, min_deg

    # Try n=47, 46, ...
    for n in range(47, 39, -1):
        best_min_deg_for_n = 0
        best_res_for_n = None
        
        # Run multiple random permutations
        # Budget: ~140s.
        # Each run takes ~0.1s?
        # Let's try 500 runs.
        for seed in range(500):
            if time.time() > deadline:
                break
            res, md = try_greedy(n, seed)
            if res is not None:
                if md > best_min_deg_for_n:
                    best_min_deg_for_n = md
                    best_res_for_n = res
                # If we found a valid one, we can stop?
                # The task says "Maximize n".
                # If we find a valid graph for n, we should try to find one with higher min degree?
                # No, the score is just n.
                # "The evaluator measures the raw value (higher is better)."
                # Value is n.
                # So if we find ANY valid graph for n=47, we should return it immediately?
                # Yes.
                return res
        
        # If no valid graph found for n, continue to n-1
        # But we should keep the best found so far?
        # The loop goes 47 down to 40.
        # If 47 fails, try 46.
        
    return {'n': 39, 'edges': []}
