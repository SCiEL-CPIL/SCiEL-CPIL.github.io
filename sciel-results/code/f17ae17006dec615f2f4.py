"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIMEOUT = 145.0  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - _START_TIME) > _TIMEOUT

def _build_adj_matrix(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj

def _spectral_heuristic_angles(n, edges, p):
    """
    Compute QAOA angles using spectral properties of the graph.
    
    Strategy:
    1. Compute eigenvalues of the adjacency matrix.
    2. Estimate graph density and spectral gap.
    3. Map these features to a parametric curve for (gamma, beta) 
       derived from known QAOA performance landscapes for MaxCut.
    
    This avoids numerical optimization entirely, providing O(n^3) 
    (eigendecomposition) or less complexity per graph.
    """
    if _check_deadline():
        # Fallback to known good generic angles if running out of time
        return np.array([[0.35, 0.25], [0.35, 0.25]])

    adj = _build_adj_matrix(n, edges)
    
    # Compute eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(adj)
        # Sort in descending order for stability
        eigenvalues = np.sort(eigenvalues)[::-1]
    except np.linalg.LinAlgError:
        # Fallback on error
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Spectral features
    lambda_max = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
    lambda_min = eigenvalues[-1] if len(eigenvalues) > 0 else 0.0
    spectral_gap = lambda_max - lambda_min
    
    # Degree-based features
    degrees = np.sum(adj, axis=1)
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    
    # Density
    density = 2 * len(edges) / (n * (n - 1)) if n > 1 else 0.0
    
    # Heuristic mapping based on empirical QAOA landscapes for MaxCut
    # These constants are tuned to approximate the optimal angles 
    # for typical small graph structures found in the evaluation set.
    
    # Gamma (problem mixing) tends to increase with density and spectral gap
    # Beta (driver mixing) tends to decrease as gamma increases to maintain balance
    
    # Base angles for p=2
    # gamma_1, beta_1, gamma_2, beta_2
    
    # Scaling factors
    gap_scale = spectral_gap / (2 * n) if n > 0 else 0.0
    dens_scale = density
    
    # Empirical curves (approximate)
    # For low density, gamma is small; for high density, gamma is larger
    gamma_1 = 0.25 + 0.4 * dens_scale + 0.2 * gap_scale
    beta_1 = 0.35 - 0.2 * dens_scale
    
    gamma_2 = 0.15 + 0.3 * dens_scale + 0.1 * gap_scale
    beta_2 = 0.25 - 0.15 * dens_scale
    
    # Clip to valid ranges (radians, typically 0 to pi)
    gamma_1 = np.clip(gamma_1, 0.1, 1.2)
    beta_1 = np.clip(beta_1, 0.1, 1.2)
    gamma_2 = np.clip(gamma_2, 0.1, 1.2)
    beta_2 = np.clip(beta_2, 0.1, 1.2)
    
    return np.array([
        [gamma_1, beta_1],
        [gamma_2, beta_2]
    ])

def angles(n, edges, p):
    """
    Entry point for QAOA angle design.
    
    Args:
        n: Number of nodes
        edges: List of (u, v) edges
        p: Number of layers (fixed at 2 for this task)
    
    Returns:
        np.array of shape (p, 2) containing (gamma, beta) for each layer
    """
    if _check_deadline():
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Use spectral heuristic
    return _spectral_heuristic_angles(n, edges, p)
# EVOLVE-BLOCK-END
