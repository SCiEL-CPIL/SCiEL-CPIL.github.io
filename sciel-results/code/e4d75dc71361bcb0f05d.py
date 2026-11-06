"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import time


def priority(adjacency):
    """
    Compute vertex priority scores for a greedy independent set decoder.
    
    Strategy: Use spectral approach with Fiedler vector (eigenvector of second
    smallest eigenvalue of Laplacian) to identify peripheral vertices that are
    less central to graph connectivity. Combine with degree-based heuristics.
    
    The Fiedler vector identifies vertices at the periphery of the main connected
    component, which are safer to include in an independent set without fragmenting
    the remaining subgraph.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = adjacency.shape[0]
    
    # Convert to sparse for efficiency
    adj_sparse = csr_matrix(adjacency)
    
    # Compute degree vector
    degrees = np.array(adj_sparse.sum(axis=1)).flatten()
    
    # Build Laplacian matrix: L = D - A
    D = csr_matrix((degrees, (np.arange(n), np.arange(n))))
    L = D - adj_sparse
    
    # Compute Fiedler vector (eigenvector of second smallest eigenvalue)
    try:
        # For connected graphs, the second smallest eigenvalue is the Fiedler value
        # We want the eigenvector corresponding to lambda_1 (second smallest)
        # Use shift-invert mode to find small eigenvalues
        if n > 10:
            # Find the two smallest eigenvalues
            vals, vecs = eigs(L, k=2, sigma=0, which='LM')
        else:
            # For small graphs, use dense eigendecomposition
            vals, vecs = np.linalg.eigh(L.toarray())
            vals = vals[1]  # Second smallest
            vecs = vecs[:, 1]
        
        # The Fiedler vector is the eigenvector for the second smallest eigenvalue
        # Sort eigenvalues to ensure we get the correct one
        if isinstance(vecs, np.ndarray) and vecs.ndim == 2:
            # For eigs, we need to sort by eigenvalue
            idx = np.argsort(vals.real)
            fiedler_vec = vecs[:, idx[1]].real
        else:
            fiedler_vec = vecs.real
        
        # Normalize the Fiedler vector
        fiedler_norm = np.linalg.norm(fiedler_vec)
        if fiedler_norm > 1e-10:
            fiedler_vec = fiedler_vec / fiedler_norm
    except Exception:
        # Fallback: use random vector if spectral computation fails
        rng = np.random.default_rng(42)
        fiedler_vec = rng.standard_normal(n)
        fiedler_norm = np.linalg.norm(fiedler_vec)
        if fiedler_norm > 1e-10:
            fiedler_vec = fiedler_vec / fiedler_norm
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return degree-based scores as fallback
        return -degrees.astype(float)
    
    # Compute centrality measures
    # 1. Degree centrality (lower degree = better for independent set)
    degree_score = -degrees.astype(float)
    
    # 2. Fiedler vector magnitude (higher magnitude = more peripheral = better)
    fiedler_magnitude = np.abs(fiedler_vec)
    
    # 3. Local clustering coefficient approximation
    # For each vertex, count triangles it participates in
    # This is expensive, so use a simpler proxy: common neighbors with neighbors
    clustering_scores = np.zeros(n)
    if n <= 200:  # Only compute for small graphs
        for i in range(n):
            neighbors = np.where(adjacency[i] > 0)[0]
            if len(neighbors) >= 2:
                # Count edges among neighbors
                sub_adj = adjacency[np.ix_(neighbors, neighbors)]
                clustering_scores[i] = sub_adj.sum() / (len(neighbors) * (len(neighbors) - 1))
            else:
                clustering_scores[i] = 0
    
    # Normalize scores to [0, 1] range
    def normalize(scores):
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val > 1e-10:
            return (scores - min_val) / (max_val - min_val)
        return np.zeros_like(scores)
    
    # Combine scores with weighted sum
    # Weighting: Fiedler magnitude (0.4), low degree (0.35), low clustering (0.25)
    w_fiedler = 0.4
    w_degree = 0.35
    w_clustering = 0.25
    
    # For clustering, we want LOW clustering (less interconnected neighbors)
    # So we invert it
    combined_score = (
        w_fiedler * normalize(fiedler_magnitude) +
        w_degree * normalize(degree_score) +
        w_clustering * normalize(-clustering_scores)
    )
    
    # Add small random perturbation to break ties and add diversity
    rng = np.random.default_rng(123)
    perturbation = rng.standard_normal(n) * 0.01
    combined_score += perturbation
    
    # Return the combined score (higher = better priority)
    return combined_score.astype(float)
# EVOLVE-BLOCK-END
