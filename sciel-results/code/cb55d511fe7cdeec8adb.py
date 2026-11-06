import itertools
import time

def construct():
    n = 18
    # Paley graph of order 17
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Build adjacency for Paley(17)
    adj = [[False] * 17 for _ in range(17)]
    edges_paley = []
    for u in range(17):
        for v in range(u + 1, 17):
            diff = (v - u) % 17
            if diff in residues:
                adj[u][v] = True
                adj[v][u] = True
                edges_paley.append([u, v])
                
    # Find all triangle-free sets of size 8
    # A set S is triangle-free if no 3 vertices in S form a triangle
    # Precompute triangles in Paley(17) to speed up check
    triangles = []
    for u in range(17):
        for v in range(u + 1, 17):
            if adj[u][v]:
                for w in range(v + 1, 17):
                    if adj[u][w] and adj[v][w]:
                        triangles.append((u, v, w))
                        
    def is_triangle_free(vertices):
        # vertices is a set or list
        # Check if any triangle is fully contained in vertices
        v_set = set(vertices)
        for t in triangles:
            if t[0] in v_set and t[1] in v_set and t[2] in v_set:
                return False
        return True

    # Find all triangle-free subsets of size 8
    tf_sets_8 = []
    for combo in itertools.combinations(range(17), 8):
        if is_triangle_free(combo):
            tf_sets_8.append(combo)
            
    # If no such sets, we can just use isolated vertex (max size would be determined by smaller sets + 1)
    # But we know there are some.
    
    # We need to find a set N such that:
    # 1. N is triangle-free
    # 2. For every T in tf_sets_8, T intersect N contains an edge.
    
    # Let's try to find such N using backtracking or greedy.
    # Since we just need ONE such N, we can try small N first?
    # Or just random search.
    
    # Constraint: N must be triangle-free.
    # Constraint: For each T, there is an edge in T whose both endpoints are in N.
    
    # Let's represent the requirement as:
    # For each T, let E_T be the set of edges in T.
    # We need to select edges E_N (subset of edges of Paley) such that:
    # 1. The graph formed by E_N is triangle-free (actually, the vertex set N must be triangle-free).
    # 2. E_N intersects E_T for all T.
    
    # We can search for a set of edges E_N.
    # Max number of edges in a triangle-free graph on 17 vertices?
    # We just need N to be triangle-free.
    
    # Heuristic: Pick edges one by one.
    # Start with empty N.
    # Randomly pick edges from the union of all E_T.
    # Check if N is triangle-free.
    # Check if all T are hit.
    
    # To ensure speed, we limit the search time.
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    best_N = None
    found = False
    
    # Get all candidate edges
    all_candidate_edges = set()
    for T in tf_sets_8:
        for i in range(len(T)):
            for j in range(i+1, len(T)):
                if adj[T[i]][T[j]]:
                    all_candidate_edges.add((T[i], T[j]))
                    
    candidate_edges_list = list(all_candidate_edges)
    
    # Random search for N
    import random
    random.seed(42)
    
    # Try to build N by adding edges
    # We want N to be small? No, just valid.
    # Let's try to find a set of edges that hits all T.
    
    # Sort T by size of E_T (hardest first?)
    # All T have size 8, so E_T size varies.
    
    # Simple greedy:
    # Pick an edge that covers the most uncovered T's.
    # But we also must maintain triangle-freeness of N.
    
    # Let's just do a randomized local search or restarts.
    
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Start with empty set of edges for N
        N_edges = set()
        N_vertices = set()
        
        # We need to hit all T.
        # Let's pick edges randomly from candidate_edges_list
        # and add them if they don't create a triangle in N.
        
        # Shuffle candidates
        shuffled_edges = candidate_edges_list[:]
        random.shuffle(shuffled_edges)
        
        valid = True
        # We don't know how many edges we need.
        # Let's just add edges until all T are hit or we fail triangle-free.
        
        # Actually, adding edges might make N not triangle-free.
        # So we check before adding.
        
        for e in shuffled_edges:
            u, v = e
            # Check if adding u,v to N creates a triangle
            # A triangle in N means 3 vertices in N forming a triangle in Paley.
            # If we add u,v, we need to check if there is w in N_vertices such that
            # (u,v,w) is a triangle in Paley.
            
            creates_triangle = False
            for w in N_vertices:
                if adj[u][w] and adj[v][w]:
                    creates_triangle = True
                    break
            
            if creates_triangle:
                continue # Skip this edge, try next one? Or restart?
            
            # Add edge
            N_edges.add(e)
            N_vertices.add(u)
            N_vertices.add(v)
            
            # Check if all T are hit
            all_hit = True
            for T in tf_sets_8:
                T_set = set(T)
                # Check if any edge in N_edges has both endpoints in T
                # Optimization: just check if N_vertices intersect T contains an edge
                # But N_vertices is the union of endpoints.
                # We need to check if there is an edge in Paley with both endpoints in N_vertices AND in T.
                
                # Since N_edges are edges in Paley, we just check if any e in N_edges is subset of T
                hit = False
                for ne in N_edges:
                    if ne[0] in T_set and ne[1] in T_set:
                        hit = True
                        break
                if not hit:
                    all_hit = False
                    break
            
            if all_hit:
                best_N = list(N_vertices)
                found = True
                break
        
        if found:
            break
            
    if found and best_N is not None:
        # Construct final graph
        edges = edges_paley[:]
        for v in best_N:
            edges.append([17, v])
        return edges
    else:
        # Fallback: Isolated vertex
        edges = edges_paley[:]
        return edges
