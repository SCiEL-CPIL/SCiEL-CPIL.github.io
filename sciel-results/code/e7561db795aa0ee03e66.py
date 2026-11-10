"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Return an edge-attractiveness matrix (larger = more attractive) for the
    capacitated vehicle routing problem solved by ant colony optimization.

    The heuristic combines:
    1. Inverse distance (closer is more attractive).
    2. Demand-awareness: nodes with higher demand are slightly less attractive
       as destinations when approaching capacity, but we don't know remaining
       capacity at heuristic computation time. Instead, we use a static
       demand-based adjustment: edges leading to high-demand nodes get a mild
       penalty (they're harder to pack), while edges from high-demand nodes
       get a mild bonus (visit them early when the truck is full).
    3. Angle-based preference: prefer edges that continue in roughly the same
       direction (reduces zigzagging), computed from coordinates.
    """
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones_like(distance_matrix)

    # Base: inverse distance
    with np.errstate(divide='ignore', invalid='ignore'):
        base = 1.0 / (distance_matrix + 1e-9)

    # Demand-based adjustment
    # demands[0] is depot (0). For customer i, higher demand means harder to serve.
    # Edge i->j: penalize if j has high demand (harder to fit later), 
    # bonus if i has high demand (should be visited early, so leaving i is fine).
    # Actually, let's think differently:
    # When at node i, going to node j: if j has high demand, it's risky (might not fit later).
    # So penalize high-demand destinations slightly.
    # But also, high-demand nodes should be visited early (when truck is full), 
    # so edges FROM depot TO high-demand nodes should be attractive.
    
    # Simple approach: scale by (1 / (1 + demand[j])) to mildly penalize high-demand destinations
    # and (1 + demand[i]) to mildly reward leaving high-demand nodes (visit them, then move on)
    demand_col = demands[np.newaxis, :]  # shape (1, n) - destination demand
    demand_row = demands[:, np.newaxis]  # shape (n, 1) - source demand
    
    # Mild demand adjustment
    demand_factor = (1.0 + 0.1 * demand_row) / (1.0 + 0.1 * demand_col)
    
    # Angle-based preference: prefer continuing in similar direction
    # For each pair (i, j), compute the angle between the vector from depot to i
    # and the vector from i to j. Prefer small angles (continuing direction).
    # Actually, a simpler heuristic: use the angle at node i between the 
    # previous "virtual" direction and the next. Since we don't have history,
    # use: for each edge i->j, compute how well j aligns with the general
    # direction from depot.
    
    # Alternative: use a "smoothness" factor based on the angle between
    # (depot -> i) and (i -> j). Small angle = smoother path.
    coords = np.asarray(coordinates, dtype=np.float64)
    depot = coords[0]
    
    # Vector from depot to each node
    vec_from_depot = coords - depot  # shape (n, 2)
    
    # For edge i->j: vector from i to j
    # vec_i_to_j = coords[j] - coords[i]
    # Angle between vec_from_depot[i] and vec_i_to_j
    # cos_angle = dot(vec_from_depot[i], vec_i_to_j) / (|vec_from_depot[i]| * |vec_i_to_j|)
    
    # Compute for all pairs
    # vec_i_to_j[i, j] = coords[j] - coords[i]
    vec_i_to_j = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]  # (n, n, 2)
    
    # Norms
    norm_depot = np.linalg.norm(vec_from_depot, axis=1)  # (n,)
    norm_i_to_j = np.linalg.norm(vec_i_to_j, axis=2)  # (n, n)
    
    # Dot product: vec_from_depot[i] . vec_i_to_j[i, j]
    dot = np.sum(vec_from_depot[:, np.newaxis, :] * vec_i_to_j, axis=2)  # (n, n)
    
    # Cosine of angle
    denom = np.outer(norm_depot, norm_i_to_j)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_angle = np.where(denom > 1e-12, dot / denom, 1.0)
    
    # Map cos_angle to a factor: prefer cos_angle close to 1 (same direction)
    # factor = 1 + alpha * cos_angle, clipped to positive
    alpha_angle = 0.3
    angle_factor = 1.0 + alpha_angle * cos_angle
    angle_factor = np.maximum(angle_factor, 0.1)
    
    # Combine
    heuristic = base * demand_factor * angle_factor
    
    # Ensure depot row is not used for transitions (ants start at depot, 
    # but the heuristic for depot->j should be well-defined)
    # The matrix should be the same shape as distance_matrix
    # Ensure no zeros or negatives
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Symmetrize? No, the problem may be asymmetric in practice, but distance_matrix
    # is likely symmetric. Keep as-is.
    
    return heuristic
# EVOLVE-BLOCK-END
