import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for ACO.
    
    Structural Change:
    Instead of using just the distance from the current node (d_ij),
    we use an 'effective distance' that accounts for the cost of returning
    to the origin (d_j0). This is because the tour must return to 0,
    and the constraint is on the total length including the return.
    
    Formula: eta_ij = prize_j / (d_ij + d_j0)^2
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Extract distance from each node to the origin (node 0)
    d_to_origin = distance_matrix[:, 0]
    
    # We need to compute eta_ij for all i, j.
    # eta_ij depends on d_ij (row i, col j) and d_j0 (col j, row 0).
    
    # d_ij is distance_matrix
    # d_j0 is d_to_origin[j] for each column j.
    # So for a fixed j, the term (d_ij + d_j0) is distance_matrix[:, j] + d_to_origin[j].
    
    # Let's vectorize this.
    # d_eff[i, j] = distance_matrix[i, j] + d_to_origin[j]
    # d_to_origin is shape (n,). We need to add it to each column of distance_matrix.
    # d_to_origin[None, :] has shape (1, n).
    # distance_matrix + d_to_origin[None, :] works via broadcasting.
    
    d_eff = distance_matrix + d_to_origin[None, :]
    
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    d_eff_safe = d_eff + eps
    
    # Calculate heuristic: prize_j / (d_eff_ij)^2
    # prizes[None, :] has shape (1, n).
    # We want prize_j to be repeated for each row i.
    # So we divide the matrix d_eff_safe^2 by the row vector of prizes?
    # No, eta_ij = prize_j / (d_eff_ij)^2.
    # So we take 1 / (d_eff_safe^2) and multiply by prizes[None, :].
    
    inv_d_eff_sq = 1.0 / (d_eff_safe ** 2)
    eta = inv_d_eff_sq * prizes[None, :]
    
    # Ensure non-negative and handle any potential numerical issues
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # Since prizes >= 0 and d_eff > 0, eta >= 0.
    # If prize is 0, eta is 0. We replace 0 with 1e-9.
    
    # Also, if any value is non-finite, replace with 1e-9
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
