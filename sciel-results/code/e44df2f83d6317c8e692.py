import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_edges = None
    best_score = 18  # Worst case is 18 (empty graph is triangle-free)
    
    # Helper: Build adjacency matrix from edge list
    def build_adj(edges):
        adj = [[0]*18 for _ in range(18)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj

    # Helper: Check if adding edge (u,v) creates a K4
    # A K4 is {u, v, x, y} where x,y are common neighbors of u and v
    def can_add_edge(adj, u, v):
        if adj[u][v]:
            return False
        common = [w for w in range(18) if adj[u][w] and adj[v][w] and w != u and w != v]
        # If there are at least 2 common neighbors that are adjacent to each other, 
        # and they are adjacent to u and v, we need to check if x-y is an edge.
        # Actually, if x and y are common neighbors, and edge(x,y) exists, then {u,v,x,y} is a K4.
        for i in range(len(common)):
            for j in range(i+1, len(common)):
                if adj[common[i]][common[j]]:
                    return False
        return True

    # Helper: Exact B&B for largest triangle-free induced subgraph
    def max_triangle_free_size(adj):
        # We want to find the max size of a subset S such that G[S] has no triangles.
        # This is the complement of the problem of finding the max clique in the "triangle graph"?
        # No, let's just use B&B on vertices.
        
        # State: current set of vertices, current size, next vertex to consider
        # We can prune if current_size + remaining_vertices <= best_found
        
        best = 0
        n = 18
        vertices = list(range(n))
        
        # Precompute: for each vertex, its neighbors
        # Triangle-free means no 3 vertices are mutually connected.
        
        def has_triangle_in_set(S):
            # S is a list of vertices
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    if adj[S[i]][S[j]]:
                        # Check if there is a k in S such that S[i]-k and S[j]-k
                        # Actually, just check all triples
                        pass
            # Simpler: check all triples
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    if adj[S[i]][S[j]]:
                        for k in range(j+1, len(S)):
                            if adj[S[i]][S[k]] and adj[S[j]][S[k]]:
                                return True
            return False
            
        # This is too slow. Let's use a different B&B.
        # We build the set incrementally.
        # If we add a vertex v to S, we must ensure no triangle is formed.
        # A triangle is formed if there exist u, w in S such that u-v, v-w, w-u are edges.
        
        def dfs(S, next_v):
            nonlocal best
            best = max(best, len(S))
            
            # Pruning
            if len(S) + (n - next_v) <= best:
                return
                
            for v in range(next_v, n):
                # Check if adding v to S creates a triangle
                ok = True
                for u in S:
                    if adj[v][u]:
                        # Check if there is w in S such that w is connected to both v and u
                        for w in S:
                            if w == u: continue
                            if adj[v][w] and adj[u][w]:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    S.append(v)
                    dfs(S, v + 1)
                    S.pop()
                    
        dfs([], 0)
        return best

    # Main loop
    for iteration in range(10000):
        if time.time() > deadline:
            break
            
        # Start with a random K4-free graph
        # Use random edges, then remove to break K4s
        adj = [[0]*18 for _ in range(18)]
        for i in range(18):
            for j in range(i+1, 18):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
                    
        # Remove K4s
        changed = True
        while changed:
            changed = False
            for i in range(18):
                for j in range(i+1, 18):
                    if not adj[i][j]: continue
                    # Check if edge (i,j) is in a K4
                    common = [k for k in range(18) if adj[i][k] and adj[j][k] and k != i and k != j]
                    for a in range(len(common)):
                        for b in range(a+1, len(common)):
                            if adj[common[a]][common[b]]:
                                # Remove a random edge from this K4
                                # Remove (i,j)
                                adj[i][j] = 0
                                adj[j][i] = 0
                                changed = True
                                break
                        if changed: break
                    if changed: break
                if changed: break
                
        # Evaluate
        edges = [[i, j] for i in range(18) for j in range(i+1, 18) if adj[i][j]]
        score = max_triangle_free_size(adj)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            
        # Local search: try to improve
        # Find largest triangle-free subgraph S
        # Try to add edges within S to break it
        # We need to find S first. The B&B above doesn't return S.
        # Let's modify B&B to return the set.
        
    if best_edges is None:
        # Fallback to baseline
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
        
    return best_edges
