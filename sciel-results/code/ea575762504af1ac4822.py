import numpy as np

def construct():
    n = 256
    # Build conflict graph: edge between a,b if (a*b) % (a+b) == 0
    # Use numpy for efficiency
    nums = list(range(1, 257))
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Compute eigenvector centrality
    # Use power iteration for the principal eigenvector
    vec = np.ones(n, dtype=np.float64)
    vec /= np.linalg.norm(vec)
    for _ in range(200):
        new_vec = adj.astype(np.float64) @ vec
        norm = np.linalg.norm(new_vec)
        if norm < 1e-12:
            break
        new_vec /= norm
        if np.allclose(new_vec, vec, atol=1e-10):
            break
        vec = new_vec
    
    # Start with all odd numbers as initial independent set
    selected = set()
    for i in range(n):
        if (i + 1) % 2 == 1:
            selected.add(i)
    
    # Try to add even numbers in order of decreasing eigenvector centrality
    even_indices = [i for i in range(n) if (i + 1) % 2 == 0]
    even_indices.sort(key=lambda i: -vec[i])
    
    for i in even_indices:
        a = i + 1
        # Check if a conflicts with any selected node
        can_add = True
        for j in selected:
            if adj[i, j] == 1:
                can_add = False
                break
        if can_add:
            selected.add(i)
    
    # Convert to list of values
    result = sorted([i + 1 for i in selected])
    return result
