import numpy as np

def priority(adjacency):
    """
    Computes a priority score for each vertex using a static structural and spectral heuristic.
    
    Approach:
    1. Compute the degree of each vertex.
    2. Compute the local clustering coefficient for each vertex.
    3. Compute the principal eigenvector of the adjacency matrix using power iteration.
    4. Combine these features to create a score that favors vertices with:
       - Low degree (fewer conflicts).
       - Low local clustering (less dense neighborhood, so picking it removes fewer "useful" neighbors).
       - Low spectral centrality (peripheral nodes are safer).
    
    This is a static heuristic, distinct from dynamic simulation approaches.
    """
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
        
    N = adjacency.shape[0]
    if N == 0:
        return np.array([])
        
    # 1. Degree
    degrees = adjacency.sum(axis=1)
    
    # 2. Local Clustering Coefficient
    # C_i = 2 * E_i / (d_i * (d_i - 1)) where E_i is edges among neighbors
    # For efficiency with N=200, we can compute this directly.
    clustering = np.zeros(N)
    for i in range(N):
        d_i = degrees[i]
        if d_i < 2:
            clustering[i] = 0.0
            continue
        # Get neighbors of i
        nbs = np.where(adjacency[i] == 1)[0]
        # Subgraph of neighbors
        # Number of edges among neighbors
        # A[i, nbs] is the row of i restricted to nbs, but we need the submatrix A[nbs, nbs]
        sub_adj = adjacency[np.ix_(nbs, nbs)]
        e_i = sub_adj.sum() / 2.0
        clustering[i] = (2.0 * e_i) / (d_i * (d_i - 1))
        
    # 3. Principal Eigenvector via Power Iteration
    # A is symmetric, so eigenvalues are real.
    # Initialize vector
    u = np.random.rand(N)
    u /= np.linalg.norm(u)
    
    for _ in range(50): # 50 iterations is usually enough for convergence for N=200
        u_new = adjacency @ u
        norm = np.linalg.norm(u_new)
        if norm < 1e-10:
            break
        u_new /= norm
        # Check convergence
        if np.linalg.norm(u_new - u) < 1e-8:
            u = u_new
            break
        u = u_new
        
    # Spectral centrality is the absolute value of the eigenvector components
    spectral_centrality = np.abs(u)
    
    # 4. Combine Scores
    # We want high score for: low degree, low clustering, low spectral centrality.
    # Use inverse scaling with a small epsilon to avoid division by zero.
    eps = 1e-6
    
    score_degree = 1.0 / (degrees + 1.0)
    score_clustering = 1.0 / (clustering + 1.0)
    score_spectral = 1.0 / (spectral_centrality + 1.0)
    
    # Geometric mean or product? Product emphasizes all three factors.
    # Let's try a weighted sum or product.
    # Product:
    final_scores = score_degree * score_clustering * score_spectral
    
    return final_scores
