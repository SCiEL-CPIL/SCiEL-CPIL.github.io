def construct():
    q = 19
    # Generate Paley graph on 19 vertices
    residues = {(x * x) % q for x in range(1, q)}
    base_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    n = 2 * q
    edges = []
    
    # Intra-copy edges
    for u, v in base_edges:
        edges.append((u, v))
        edges.append((u + q, v + q))
        
    # Cross-copy edges
    for u, v in base_edges:
        edges.append((u, v + q))
        
    return n, edges
