"""Rewritten program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute vertex scores for greedy Maximum Independent Set selection.
    
    Strategy: Combine multiple structural features to identify vertices that
    are good candidates for the independent set:
    1. Low degree (less blocking of other candidates)
    2. High sum of inverse neighbor degrees (selecting removes low-cost neighbors)
    3. Spectral information (Fiedler vector) for global structure
    4. Local density measures
    
    The greedy decoder picks vertices in descending score order, skipping
    neighbors of already-selected vertices.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    A = adjacency.astype(float)
    degrees = A.sum(axis=1)
    
    # Feature 1: Inverse neighbor degree sum
    # For each vertex i, sum 1/(d_j + 1) over neighbors j
    # This quantifies the "independence value" gained by selecting i
    inv_deg = 1.0 / (degrees + 1.0)
    neighbor_inv_deg_sum = A @ inv_deg
    
    # Feature 2: Negative degree (prefer low degree)
    neg_degree = -degrees
    
    # Feature 3: Local density - number of edges among neighbors / possible edges
    # This is computable via A^2 diagonal
    A2 = A @ A
    neighbor_edges = (A2.diagonal() - degrees) / 2.0  # edges among neighbors
    
    # Normalize features
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-10:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    # Check time
    if time.time() > deadline - 10:
        # Return simple score based on low degree and high neighbor inverse degree
        scores = 0.5 * normalize(neighbor_inv_deg_sum) + 0.5 * normalize(-degrees)
        return scores
    
    # Compute Fiedler vector for spectral information
    try:
        D = np.diag(degrees)
        L = D - A
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        idx = np.argsort(eigenvalues)
        fiedler = eigenvectors[:, idx[1]]
        if np.sum(fiedler) < 0:
            fiedler = -fiedler
        fiedler_abs = np.abs(fiedler)
        fiedler_norm = normalize(fiedler_abs)
    except np.linalg.LinAlgError:
        fiedler_norm = np.zeros(n)
    
    # Check time again
    if time.time() > deadline - 5:
        scores = (0.3 * normalize(neighbor_inv_deg_sum) + 
                  0.3 * normalize(-degrees) + 
                  0.2 * fiedler_norm +
                  0.2 * normalize(-neighbor_edges))
        return scores
    
    # Feature 4: Negative neighbor edge count (prefer vertices whose neighbors
    # are not densely connected - less "wasted" blocking)
    neg_neighbor_edges = -neighbor_edges
    
    # Feature 5: Eigenvector centrality (iterative power method)
    # High eigenvector centrality means connected to important nodes
    try:
        v = np.ones(n) / n
        for _ in range(50):
            v_new = A @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-10:
                break
            v = v_new / norm
            if time.time() > deadline - 3:
                break
        eigenvector_cent = normalize(v)
    except Exception:
        eigenvector_cent = np.zeros(n)
    
    # Composite score with tuned weights
    # Emphasize: low degree, high neighbor inverse degree sum, spectral periphery
    scores = (
        0.30 * normalize(neighbor_inv_deg_sum) +
        0.25 * normalize(-degrees) +
        0.20 * fiedler_norm +
        0.15 * normalize(-neighbor_edges) +
        0.10 * eigenvector_cent
    )
    
    # Add tiny deterministic tiebreaker
    scores += np.arange(n) * 1e-12
    
    return scores
# EVOLVE-BLOCK-END
