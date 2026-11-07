import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    t0 = time.time()
    deadline = t0 + 85
    n = P['n']
    
    # Base graph setup
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # Initialize distance matrix for the path
    # D[i][j] is distance between i and j in the current graph
    def get_initial_dist():
        dist = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist[i][j] = j - i
                dist[j][i] = j - i
        return dist

    # Generate candidate edges
    # Must be same parity and not in base
    cands = []
    for u in range(n):
        for v in range(u + 1, n):
            if (v - u) % 2 == 0 and (u, v) not in base_edges:
                # Distance must be >= 4 to avoid triangles with path segments of length 2
                # If dist is 2, triangle (u, u+1, u+2, u) is formed.
                # If dist is 4, cycle is length 5. Safe.
                if v - u >= 4:
                    cands.append((u, v))
                    
    def is_triangle_free(added_edges):
        # Check if the graph formed by base + added is triangle-free
        # We know base is triangle-free.
        # Added edges are same-parity.
        # Triangles can only be formed by:
        # 1. 3 added edges
        # 2. 2 added + 1 base (Impossible due to parity: Add edges connect same parity.
        #    Let added be (a,b) and (b,c). a~b, b~c => a~c.
        #    Base edge (x,y) has x!=y parity.
        #    Triangle a,b,c. Edges (a,b) add, (b,c) add, (c,a) base?
        #    c~a (add) implies c,a same parity.
        #    If (c,a) is base, c,a diff parity. Contradiction.
        #    So 2 add + 1 base is impossible.
        # 3. 1 add + 2 base.
        #    Add (u,v). Base (v,w), (w,u).
        #    v,w diff parity. w,u diff parity => v,u same parity.
        #    This is consistent.
        #    Triangle u,v,w.
        #    Path in base between u and v is u...v.
        #    If (u,v) is added, and (v,w), (w,u) are base edges.
        #    This implies w is adjacent to u and v in the base path.
        #    In a path, a vertex has max 2 neighbors.
        #    If u and v are both neighbors of w, then u and v are distance 2 apart in the path.
        #    e.g., w=1, u=0, v=2.
        #    Added edge (0,2).
        #    We filtered out distance 2 edges. So this case is prevented by candidate generation.
        
        # So we only need to check for triangles formed by 3 added edges.
        added_set = set()
        for u, v in added_edges:
            added_set.add((u, v))
            added_set.add((v, u))
            
        # Check all triples of added edges? Or just check adjacency.
        # Build adjacency for added edges only
        adj_added = {i: set() for i in range(n)}
        for u, v in added_edges:
            adj_added[u].add(v)
            adj_added[v].add(u)
            
        for u in range(n):
            for v in adj_added[u]:
                for w in adj_added[u]:
                    if v < w and w in adj_added[v]:
                        return False
        return True

    def compute_diameter(added_edges):
        dist = get_initial_dist()
        # Apply added edges
        # Floyd-Warshall style update or BFS from each node?
        # Since N=24, we can just run BFS from each node on the combined graph.
        # Or use the distance matrix update.
        # Let's use BFS for simplicity and robustness against implementation errors in matrix update.
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_d = 0
        for s in range(n):
            # BFS
            visited = [False] * n
            visited[s] = True
            queue = [(s, 0)]
            while queue:
                node, d = queue.pop(0)
                max_d = max(max_d, d)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        queue.append((nei, d + 1))
            if not all(visited):
                return float('inf') # Disconnected
        return max_d

    def is_valid(added_edges):
        if not is_triangle_free(added_edges):
            return False
        diam = compute_diameter(added_edges)
        return diam <= 4

    # Greedy Construction
    def greedy_construct(random_seed=0):
        random.seed(random_seed)
        added = []
        current_diam = compute_diameter(added)
        
        while current_diam > 4:
            best_e = None
            best_diam = float('inf')
            # Sort candidates randomly to break ties differently in different runs?
            # Or just iterate.
            cands_copy = cands[:]
            random.shuffle(cands_copy)
            
            for e in cands_copy:
                if e in added:
                    continue
                trial = added + [e]
                # Quick check: triangle free?
                if not is_triangle_free(trial):
                    continue
                d = compute_diameter(trial)
                if d < best_diam:
                    best_diam = d
                    best_e = e
                    # If we reached diameter 4, we can stop looking for *better* diameter,
                    # but we still want to minimize edges. 
                    # Actually, if d == 4, we should still pick the one that 
                    # allows for the most future removals? 
                    # For now, just pick the one with min diameter.
                    if d <= 4:
                        # We found a valid solution. 
                        # Should we keep adding to try to reduce count later? No, we are adding.
                        # We stop adding when diameter is <= 4.
                        pass
            
            if best_e is None:
                # No edge can reduce diameter? 
                # This shouldn't happen if we have enough candidates.
                break
                
            added.append(best_e)
            current_diam = best_diam
            
        # Local Search: Minimize edges
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(added)):
                trial = added[:i] + added[i+1:]
                if is_valid(trial):
                    added = trial
                    improved = True
                    break # Restart loop
                    
        return added

    # Try multiple random seeds to find a better (smaller) solution
    best_solution = None
    best_len = float('inf')
    
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        sol = greedy_construct(seed)
        if sol is not None and is_valid(sol):
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol[:]
                
    # If greedy failed or didn't find a good one, try a more exhaustive approach on a smaller set?
    # Or just return the best found.
    # Given the problem size, greedy should find a solution.
    # A path of 24 with diameter 4.
    # Approx edges needed: 24/4 = 6?
    # Let's ensure we have a fallback.
    
    if best_solution is None:
        # Fallback: Add all distance 4 edges?
        # This will be valid but likely not minimal.
        fallback = [(u, u+4) for u in range(n-4) if (u, u+4) in cands]
        if is_valid(fallback):
            best_solution = fallback
        else:
            # Should not happen
            best_solution = []

    return [[u, v] for u, v in best_solution]
