import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic edge-attractiveness matrix for ACO orienteering.
    
    Design:
    - Efficiency ratio: prize_j / (d_ij + d_j0) captures value per unit total detour cost.
    - We use a soft-max / log-sum-exp inspired amplification to create a smooth,
      differentiable surrogate that strongly favors high-efficiency edges while
      keeping all entries positive and finite.
    - Specifically: eta_ij = exp(alpha * (log(prize_j + eps) - log(d_ij + d_j0 + eps)))
      which is equivalent to ((prize_j + eps) / (d_ij + d_j0 + eps))^alpha
    - We also incorporate a length-feasibility filter: edges that would make it
      impossible to return to 0 within max_len are given very low attractiveness.
    - gamma (alpha) is tuned to amplify differences (gamma=2 worked well in observations).
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    eps = 1e-9
    
    # Return cost from each node to node 0
    d_to_0 = distance_matrix[:, 0].copy()
    
    # Total cost of visiting node j from node i: d[i,j] + d[j,0]
    # Shape: (n, n)
    total_cost = distance_matrix + d_to_0[None, :]
    
    # Ensure positivity
    total_cost = np.maximum(total_cost, eps)
    prize_row = np.maximum(prizes[None, :], eps)
    
    # Base efficiency: prize / total_cost
    efficiency = prize_row / total_cost
    
    # Amplify with gamma=2 (measured improvement)
    gamma = 2.0
    eta = np.power(efficiency, gamma)
    
    # Additional refinement: incorporate feasibility more directly.
    # An edge (i, j) is only useful if there exists some path length budget.
    # Since max_len is small (3), many edges are infeasible. We can't know
    # the current path length in a static heuristic, but we can penalize
    # edges where d[i,j] + d[j,0] > max_len (always infeasible regardless of path).
    # Actually, the constraint is: current_len + d[i,j] + d[j,0] <= max_len
    # The minimum current_len is 0 (at start), so if d[i,j] + d[j,0] > max_len,
    # it's never feasible. Let's zero those out (set to very small).
    
    infeasible = (distance_matrix + d_to_0[None, :]) > max_len
    eta[infeasible] = 1e-9
    
    # Also, node 0 has prize 0, so edges to/from node 0 with 0 prize get low heuristic.
    # That's already handled since prizes[0] = 0.
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Normalize to keep values in a reasonable range (avoid overflow in pheromone*heuristic)
    max_eta = np.max(eta)
    if max_eta > 0:
        eta = eta / max_eta
    
    # Ensure minimum value
    eta = np.maximum(eta, 1e-9)
    
    return eta
