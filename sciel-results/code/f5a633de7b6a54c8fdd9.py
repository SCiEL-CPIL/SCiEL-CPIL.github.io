"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ACO on CVRP.
    
    Uses a K-Nearest-Neighbors based gate (via cKDTree) to encourage short,
    locally-connected edges while preserving enough flexibility for the colony
    to find good routes. Combines power-transformed inverse distance with a
    Clarke-Wright savings component.
    
    The heuristic is:
        eta[i, j] = (1 / d[i, j]^alpha) * (1 + beta * savings_norm[i, j]) * gate[i, j]
    
    where gate[i, j] is high for K-nearest neighbors and all depot edges.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    d = distance_matrix.astype(np.float64)
    
    # Avoid division by zero
    d_safe = np.where(d > 1e-9, d, 1e9)
    
    # Power-transformed inverse distance
    alpha = 2.0
    eta_base = 1.0 / (d_safe ** alpha)
    
    # Clarke-Wright savings
    depot_dist = d[:, 0].copy()
    savings = depot_dist[:, np.newaxis] + depot_dist[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    
    max_savings = savings_pos.max()
    if max_savings > 1e-9:
        savings_norm = savings_pos / max_savings
    else:
        savings_norm = np.zeros_like(savings_pos)
    
    # Combine base with savings
    beta = 0.5
    eta = eta_base * (1.0 + beta * savings_norm)
    
    # K-Nearest-Neighbors gate
    K = 12  # number of neighbors to allow
    try:
        tree = cKDTree(coordinates)
        # Get K nearest neighbors for each point
        distances, indices = tree.query(coordinates, k=K + 1)  # +1 for self
        
        # Build gate matrix
        gate = np.full((n, n), 1e-6)
        
        for i in range(n):
            for j in indices[i]:
                if j != i:
                    gate[i, j] = 1.0
        
        # Allow all depot edges
        gate[0, :] = 1.0
        gate[:, 0] = 1.0
        
        # Apply gate
        eta = eta * gate
        
    except Exception:
        # Fallback: no gating
        pass
    
    # Ensure positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    np.fill_diagonal(eta, 1e-9)
    
    # Time safety
    elapsed = time.time() - start_time
    if elapsed > 10:
        d_safe2 = np.where(d > 1e-9, d, 1e9)
        eta_simple = 1.0 / (d_safe2 ** 2.0)
        eta_simple = np.where(np.isfinite(eta_simple) & (eta_simple > 0), eta_simple, 1e-9)
        np.fill_diagonal(eta_simple, 1e-9)
        return eta_simple
    
    return eta
# EVOLVE-BLOCK-END
