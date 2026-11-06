import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Construct Paley Graph P17
    n_paley = 17
    # Quadratic residues mod 17
    residues = set([x * x % n_paley for x in range(1, n_paley)])
    residues.discard(0)
    
    paley_edges = []
    adj_paley = [set() for _ in range(n_paley)]
    for u in range(n_paley):
        for v in range(u + 1, n_paley):
            if (u - v) % n_paley in residues or (v - u) % n_paley in residues:
                paley_edges.append([u, v])
                adj_paley[u].add(v)
                adj_paley[v].add(u)
    
    # Helper to check if a set of vertices is triangle-free in a graph
    def is_triangle_free_in_paley(vertices):
        # vertices is a set of ints
        verts = list(vertices)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if v in adj_paley[u]:
                    # Check if there is a common neighbor w in vertices
                    # Actually, just check if any w in vertices is connected to both u and v
                    for w in verts:
                        if w != u and w != v:
                            if w in adj_paley[u] and w in adj_paley[v]:
                                return False
        return True

    # Helper to compute largest triangle-free induced subgraph size
    def max_tf_size(edges, n):
        # Build adjacency matrix/list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Branch and bound for largest triangle-free induced subgraph
        # A set S is triangle-free if no three vertices in S form a triangle.
        # We can use a recursive search.
        # State: current set S, available vertices A
        # To be efficient, we can use the property that if S is triangle-free,
        # adding a vertex v is valid if v has no two neighbors in S that are connected to each other.
        
        best_size = 0
        
        def dfs(S, A):
            nonlocal best_size
            if len(S) + len(A) <= best_size:
                return
            
            if not A:
                if len(S) > best_size:
                    best_size = len(S)
                return
            
            # Heuristic: pick a vertex from A
            # Try to include a vertex that allows many others
            # Simple: iterate through A
            
            # Optimization: If S is not empty, v can be added to S only if
            # N(v) intersection S is an independent set in G.
            
            # Let's just iterate all v in A
            # To speed up, order A by degree in induced subgraph of A?
            # For n=18, simple iteration is fine.
            
            for v in list(A):
                # Check if v can be added to S
                can_add = True
                if S:
                    # Check if any two neighbors of v in S are connected
                    neigh_in_S = [u for u in S if u in adj[v]]
                    for i in range(len(neigh_in_S)):
                        for j in range(i + 1, len(neigh_in_S)):
                            if neigh_in_S[j] in adj[neigh_in_S[i]]:
                                can_add = False
                                break
                        if not can_add:
                            break
                
                if can_add:
                    new_S = S | {v}
                    # New A: vertices in A that can coexist with new_S?
                    # Actually, we just remove v from A and recurse.
                    # The constraint is only on the final set.
                    # So we just need to ensure the final set is TF.
                    # This DFS explores all subsets. 2^18 = 262k. Fast enough.
                    dfs(new_S, A - {v})
                
                # Also explore NOT adding v
                # But to avoid duplicates and prune, we usually branch include/exclude.
                # Standard B&B:
                # 1. Include v
                # 2. Exclude v (and remove v from A)
                
                # Exclude v
                dfs(S, A - {v})
        
        dfs(set(), set(range(n)))
        return best_size

    best_edges = None
    best_score = 100
    
    # Candidates for N(17)
    # N(17) must be triangle-free in P17 (to avoid K4 with 17)
    # We try sizes 1, 2, 3
    
    candidates = []
    
    # Size 1
    for v in range(n_paley):
        candidates.append({v})
        
    # Size 2
    for u in range(n_paley):
        for v in range(u + 1, n_paley):
            if is_triangle_free_in_paley({u, v}):
                candidates.append({u, v})
                
    # Size 3
    # Might be too many, but let's try a few
    # Actually, size 2 is likely sufficient based on p39 score.
    # Let's add size 3 if time permits.
    
    for N17 in candidates:
        if time.time() > deadline:
            break
            
        # Construct edges
        edges = paley_edges.copy()
        for v in N17:
            edges.append([17, v])
            
        # Check K4-free
        # P17 is K4-free.
        # New K4 must include 17.
        # So 17 and 3 vertices from N17 must form a K4.
        # This requires the 3 vertices to form a triangle in P17 and be connected to 17.
        # Since N17 is triangle-free in P17, no such K4 exists.
        
        score = max_tf_size(edges, 18)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            # If we get 8 or lower, we can stop early?
            # Theoretical min is likely 8 or 9.
            if best_score <= 8:
                break

    if best_edges is None:
        # Fallback: P17 + isolated vertex
        best_edges = paley_edges.copy()
        
    return best_edges
