import numpy as np
import time

_T0 = time.time()
_BUDGET = 330.0
_DEADLINE = _T0 + _BUDGET
_TIME_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    
    Strategy:
    - Uses a sharp inverse distance exponent (1.5) to strongly prefer short edges.
    - Structurally separates depot edges from customer-customer edges.
    - For customer-customer edges, uses a 'savings per distance cost' metric:
      h[i, j] = (savings[i, j] + eps) * (distance[i, j] ** -1.5)
    - For depot edges, uses pure sharp inverse distance:
      h[0, j] = (distance[0, j] ** -1.5)
    
    This structural change moves from an additive boost (current) to a multiplicative 
    efficiency metric, which may better balance the trade-off between route consolidation 
    (savings) and compactness (distance).
    """
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    # Basic validation and time check
    if dm.shape != (n, n) or not np.isfinite(dm).all():
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        h = 1.0 / (d + 1e-9)
        np.fill_diagonal(h, 0.0)
        return h
    
    if _remaining() < 10.0:
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        h = 1.0 / (d + 1e-9)
        np.fill_diagonal(h, 0.0)
        return h

    eps = 1e-9
    
    # Compute savings: s[i, j] = d[i, 0] + d[0, j] - d[i, j]
    d_i0 = dm[:, 0]
    d_0j = dm[0, :]
    savings = d_i0[:, np.newaxis] + d_0j[np.newaxis, :] - dm
    savings = np.maximum(savings, 0.0)
    
    # Compute sharp inverse distance: d^-1.5
    # Add small epsilon to avoid division by zero for diagonal (though we handle diagonal separately)
    d_safe = dm + eps
    inv_dist_sharp = 1.0 / (d_safe ** 1.5)
    
    # Initialize heuristic matrix
    h = np.zeros((n, n))
    
    # 1. Depot Edges (i=0 or j=0)
    # For these, savings are not the primary driver (s[0, j] = 0 + d[0,j] - d[0,j] = 0).
    # Use pure inverse distance to guide the start/end of routes.
    # h[0, j] = inv_dist_sharp[0, j]
    # h[i, 0] = inv_dist_sharp[i, 0]
    h[0, :] = inv_dist_sharp[0, :]
    h[:, 0] = inv_dist_sharp[:, 0]
    
    # 2. Customer-Customer Edges (i>0, j>0)
    # Use the structural metric: (savings + eps) * inv_dist_sharp
    # This rewards edges that have high savings relative to their distance cost.
    cc_mask = np.ones((n, n), dtype=bool)
    cc_mask[0, :] = False
    cc_mask[:, 0] = False
    np.fill_diagonal(cc_mask, False)
    
    # Calculate for all, then mask
    h_cc_candidate = (savings + eps) * inv_dist_sharp
    
    # Apply only to customer-customer edges
    h[cc_mask] = h_cc_candidate[cc_mask]
    
    # Ensure diagonal is 0 (no self-loops)
    np.fill_diagonal(h, 0.0)
    
    # Handle non-finite or non-positive values
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    # Optional: Normalize to prevent extreme values from dominating pheromone updates
    # The ACO algorithm multiplies pheromone * heuristic. If heuristic values are very large,
    # pheromone updates (which add 1/cost) might be negligible. If very small, they might vanish.
    # Normalizing to max 1.0 is a safe standard practice.
    max_h = np.max(h)
    if max_h > 0:
        h = h / max_h
        
    # Re-apply positivity constraint after normalization (just in case of floating point issues)
    h[h <= 0] = 1e-9
    np.fill_diagonal(h, 0.0)
    
    return h
