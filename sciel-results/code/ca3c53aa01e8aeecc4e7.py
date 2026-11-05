import time
import random

def has_c5_after_adding_edge(adj, u, v):
    # adj is a list of sets
    # Check if adding (u, v) creates a C5
    # Path u-a-b-c-v
    # a in N(u), c in N(v)
    # b in N(a) and b in N(c)
    # Distinctness: u, v, a, b, c distinct
    
    nu = adj[u]
    nv = adj[v]
    
    # a must be in nu, c must be in nv
    # b must be in N(a) and N(c)
    
    # Optimization:
    # For each a in nu:
    #   For each c in nv:
    #     If a == c: continue
    #     If a == v or c == u: continue (a is in nu so a!=u, c is in nv so c!=v. But a could be v? No, simple graph, no self loops. But v is not in nu unless (u,v) already exists. We are checking before adding.)
    #     Actually, if (u,v) is not an edge, v is not in nu.
    #     So a != v and c != u is guaranteed if graph is simple and (u,v) not yet added.
    #     However, a and c must be distinct from u and v.
    #     a in N(u) => a != u.
    #     c in N(v) => c != v.
    #     We need a != v and c != u.
    #     Since (u,v) is not an edge, v is not in N(u), so a != v.
    #     Since (u,v) is not an edge, u is not in N(v), so c != u.
    #     So we just need a != c.
    #     And b != u, b != v, b != a, b != c.
    
    for a in nu:
        if a == v: continue # Should not happen
        for c in nv:
            if a == c: continue
            if c == u: continue # Should not happen
            
            # Find b in N(a) intersect N(c)
            # N(a) is adj[a], N(c) is adj[c]
            # b must be in adj[a] and adj[c]
            # b must not be u, v, a, c
            
            # Intersection of adj[a] and adj[c]
            # Since n is small, we can iterate
            for b in adj[a]:
                if b in adj[c]:
                    if b == u or b == v or b == a or b == c:
                        continue
                    # Found a C5: u-a-b-c-v-u
                    return True
    return False

def calc_alpha(adj, n):
    # Simple backtracking or greedy upper bound?
    # For exact check, we need to know if alpha >= 9.
    # We can use a simple branch and bound or just a greedy heuristic to find an independent set.
    # If greedy finds 9, it's invalid.
    # If greedy finds < 9, it might still have 9.
    # But we want to ADD edges to REDUCE alpha.
    # So if we start with empty graph (alpha=n), and add edges, alpha decreases.
    # We stop adding edges if we create C5.
    # The resulting graph is maximal C5-free (with respect to the order).
    # We then check if alpha < 9.
    
    # To check alpha < 9 exactly is NP-hard, but for n=35 it's small.
    # We can use a simple recursive search to find an independent set of size 9.
    # If found, return False (invalid).
    # If not found, return True (valid).
    
    def find_independent_set(k):
        if k == 0:
            return []
        # Pick a vertex
        # Try to include it
        # This is a standard exact cover / independent set search
        
    # Given time constraints, a heuristic might be safer if exact is too slow.
    # But n=35 is small enough for a decent exact solver.
    
    # Let's use a simple recursive search with pruning.
    # Sort vertices by degree descending (low degree vertices are more likely to be in independent set? No, high degree are less likely).
    # Actually, for finding a LARGE independent set, we should pick vertices with LOW degree?
    # No, vertices with low degree have fewer conflicts.
    
    # Let's just do a randomized greedy several times. If we find 9, it's invalid.
    # If we don't find 9 in many tries, it's likely valid?
    # Risky.
    
    # Better: Use the fact that we are adding edges.
    # We can maintain the independence number approximately?
    
    # Let's implement a fast exact check for alpha >= 9.
    
    vertices = list(range(n))
    # Heuristic: order by degree
    degrees = [len(adj[i]) for i in vertices]
    vertices.sort(key=lambda x: degrees[x])
    
    # Backtracking
    # State: current_set, candidates
    # If len(current_set) + len(candidates) < 9: return False
    
    # This might be slow if done for every edge addition.
    # We should only check alpha at the end, or periodically.
    
    # Strategy:
    # 1. Build maximal C5-free graph.
    # 2. Check alpha exactly.
    # 3. If invalid, retry with different random seed.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Try n = 35, 34, 33
    for n in [35, 34, 33]:
        if time.time() > deadline:
            break
        success = False
        for attempt in range(100): # 100 attempts per n
            if time.time() > deadline:
                break
            random.seed(attempt)
            adj = [set() for _ in range(n)]
            
            # Generate random edge order
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges)
            
            for u, v in edges:
                if time.time() > deadline:
                    break
                if v in adj[u]:
                    continue
                if has_c5_after_adding_edge(adj, u, v):
                    continue
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
            
            # Check alpha
            if check_alpha_less_than_9(adj, n):
                # Convert to edge list
                edge_list = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edge_list.append([i, j])
                return {'n': n, 'edges': edge_list}
        
        if success:
            break
            
    # Fallback to seed
    n = 32
    edges = []
    for block_idx in range(8):
        start = block_idx * 4
        verts = [start, start + 1, start + 2, start + 3]
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                edges.append([verts[i], verts[j]])
    return {'n': n, 'edges': edges}

def check_alpha_less_than_9(adj, n):
    # Exact check: does an independent set of size 9 exist?
    # Use branch and bound.
    
    # Heuristic ordering: sort by degree
    vertices = list(range(n))
    vertices.sort(key=lambda x: len(adj[x]))
    
    # Map original index to sorted index?
    # Or just use the sorted list for search.
    
    # We need to check if there is a set of 9 vertices with no edges between them.
    
    # Recursive search
    # current: list of vertices in independent set
    # candidates: list of vertices that can be added (not connected to any in current)
    
    # Optimization: If len(current) + len(candidates) < 9, return False.
    
    # We will implement a simple DFS.
    
    def dfs(current_size, candidates):
        if current_size + len(candidates) < 9:
            return False
        if current_size == 9:
            return True
        
        # Pick a vertex from candidates
        # Heuristic: pick vertex with lowest degree in the subgraph induced by candidates?
        # Or just pick the first one.
        
        # To speed up, we can branch on the first candidate.
        # Option 1: Include v.
        # Option 2: Exclude v.
        
        # This is standard independent set search.
        # Given n=35, depth 9, it should be fast.
        
        if not candidates:
            return False
            
        v = candidates[0]
        rest = candidates[1:]
        
        # Option 1: Include v
        # New candidates: rest \ N(v)
        new_candidates = [u for u in rest if u not in adj[v]]
        if dfs(current_size + 1, new_candidates):
            return True
            
        # Option 2: Exclude v
        if dfs(current_size, rest):
            return True
            
        return False
        
    return not dfs(0, vertices)
