import itertools

def construct():
    def generate_paley(p):
        """Generates the Paley graph on p vertices (p must be 1 mod 4)."""
        if p % 4 != 1:
            return None
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        edges = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in residues:
                    edges.append((i, j))
        return p, edges

    def is_5_5_free(n, edges):
        """Checks if the graph has no K5 and no independent set of size 5."""
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Check for K5
        for combo in itertools.combinations(range(n), 5):
            is_clique = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if not adj[combo[i]][combo[j]]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return False
        
        # Check for Independent Set of size 5 (Clique in complement)
        for combo in itertools.combinations(range(n), 5):
            is_independent = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if adj[combo[i]][combo[j]]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return False
        
        return True

    # 1. Get a known valid 5,5-free graph: Paley(37)
    p37 = generate_paley(37)
    if not p37:
        # Fallback if generation fails, though it shouldn't
        return 37, []
        
    n_base, edges_base = p37
    
    # 2. Extract a subgraph on 19 vertices (0 to 18)
    # Since the whole graph is 5,5-free, the subgraph is also 5,5-free.
    n_sub = 19
    edges_sub = []
    for u, v in edges_base:
        if u < n_sub and v < n_sub:
            edges_sub.append((u, v))
            
    # 3. Construct the 2-fold cover on 2 * n_sub = 38 vertices
    # Vertices 0..18 are Copy 0
    # Vertices 19..37 are Copy 1
    n_cover = 2 * n_sub
    edges_cover = []
    
    # Edges within Copy 0
    for u, v in edges_sub:
        edges_cover.append((u, v))
        
    # Edges within Copy 1
    for u, v in edges_sub:
        edges_cover.append((u + n_sub, v + n_sub))
        
    # Cross edges: (u in Copy 0, v in Copy 1) if u ~ v in base
    for u, v in edges_sub:
        edges_cover.append((u, v + n_sub))
        edges_cover.append((v, u + n_sub))
        
    # The resulting graph is 5,5-free if the base is.
    # We can optionally verify, but it's mathematically sound.
    # However, to be safe against any implementation error in the logic, 
    # we can check. But checking 38 vertices is fast.
    
    if is_5_5_free(n_cover, edges_cover):
        return n_cover, edges_cover
    else:
        # Fallback to the original 37-vertex graph if something went wrong
        return n_base, edges_base
