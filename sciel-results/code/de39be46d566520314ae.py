import random

def construct():
    n = 40
    k = 9
    A = list(range(k))
    B = list(range(k, n))
    
    edges = []
    
    # Add all cross edges
    for u in A:
        for v in B:
            edges.append([u, v])
    
    # Calculate remaining edges needed for B
    num_cross = k * (n - k)
    edges_needed = 401 - num_cross
    
    # Generate edges in B
    b_indices = list(range(len(B)))
    
    # Greedy construction of G_B
    max_deg = 8
    deg = [0] * len(B)
    b_edges = []
    
    # Deterministic ordering: sort by distance or just standard
    possible_edges = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            possible_edges.append((i, j))
            
    # To make it deterministic but "random-like" to avoid structural traps,
    # we can use a simple permutation.
    # Or just standard order. Standard order might create a clique first?
    # If we add (0,1), (0,2)... deg 0 grows fast.
    # Better to shuffle with a fixed seed.
    random.seed(42)
    random.shuffle(possible_edges)
    
    for i, j in possible_edges:
        if len(b_edges) == edges_needed:
            break
        if deg[i] < max_deg and deg[j] < max_deg:
            b_edges.append((i, j))
            deg[i] += 1
            deg[j] += 1
            
    # Fallback: if not enough edges, add any remaining
    if len(b_edges) < edges_needed:
        # This should not happen, but just in case
        # Add edges ignoring max_deg
        for i, j in possible_edges:
            if len(b_edges) == edges_needed:
                break
            if (i, j) not in b_edges:
                b_edges.append((i, j))
                deg[i] += 1
                deg[j] += 1

    for i, j in b_edges:
        edges.append([B[i], B[j]])
        
    return edges
