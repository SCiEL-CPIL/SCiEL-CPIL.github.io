"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    The heuristic combines:
    - Inverse distance (higher for closer nodes)
    - A demand-based penalty (discouraging edges to high-demand customers when
      capacity is tight, encouraging balanced route loads)
    - A directional bias toward nodes that are "forward" in the route (reducing
      backtracking)
    
    Returns a matrix of the same shape as distance_matrix where larger values
    indicate more attractive edges.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance with a moderate exponent
    # Exponent 2-4 tends to work well; we use 2.5 as a balance
    base = 1.0 / (distance_matrix + 1e-9) ** 2.5
    
    # Demand-based adjustment:
    # For customer j, higher demand means it's "heavier" to visit.
    # We slightly penalize edges to high-demand customers to encourage
    # spreading heavy customers across routes (since capacity is 50 and
    # demands are 1-9, most routes will carry several customers).
    # The depot (node 0) has demand 0, so no penalty there.
    demand_penalty = np.ones((n, n))
    # For edges going TO a customer j (j > 0), apply a mild penalty proportional
    # to the customer's demand. This helps ants avoid clustering too many
    # high-demand customers in one route.
    for j in range(1, n):
        demand_penalty[:, j] *= (1.0 / (1.0 + demands[j] / capacity))
    
    # Directional bias: prefer edges that move "forward" relative to the depot.
    # Compute angle from depot to each node. For edge i->j, prefer if j is
    # roughly in the same angular sector as i (reduces crossing routes).
    depot = coordinates[0]
    angles = np.arctan2(coordinates[1:, 1] - depot[1], coordinates[1:, 0] - depot[0])
    
    # Create a directional bonus matrix
    directional_bonus = np.ones((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            # Angular difference
            da = angles[i-1] - angles[j-1]
            # Normalize to [-pi, pi]
            da = (da + np.pi) % (2 * np.pi) - np.pi
            # Bonus for small angular differences (same direction from depot)
            directional_bonus[i, j] = 1.0 + 0.3 * np.cos(da)
    
    # Combine all components
    result = base * demand_penalty * directional_bonus
    
    # Ensure all values are positive
    result = np.maximum(result, 1e-9)
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
