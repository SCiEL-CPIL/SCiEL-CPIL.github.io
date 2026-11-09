"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix for CVRP using a 2-step look-ahead heuristic.
    
    The heuristic value for moving from i to j is based on the cost of the path i->j->k,
    where k is the nearest neighbor of j (excluding i). This encourages ants to choose
    moves that lead to locally optimal, smooth paths rather than just the single shortest
    edge.
    """
    n = distance_matrix.shape[0]
    
    # 1. Find the nearest neighbor for each node j, excluding the node itself.
    # We set the diagonal to infinity so the node itself is not picked.
    # Note: In CVRP, moving from depot to customer or customer to depot is allowed.
    # The 'nearest neighbor' logic here is a proxy for 'where do I likely go next'.
    
    # Create a copy of distance matrix to find nearest neighbors
    dist_for_nn = distance_matrix.copy()
    np.fill_diagonal(dist_for_nn, np.inf)
    
    # For each row j (current node), find the index of the minimum distance (next node k)
    # This gives us the 'preferred' next step from any node j.
    nearest_neighbor_idx = np.argmin(dist_for_nn, axis=1)
    
    # 2. Calculate the 'look-ahead' cost for each edge i->j.
    # Cost(i, j) = d(i, j) + alpha * d(j, k(j))
    # where k(j) is the nearest neighbor of j.
    # We use alpha=1.0 as a simple balance.
    
    # d(i, j) is just distance_matrix
    d_ij = distance_matrix
    
    # d(j, k(j))
    # We need to gather d[j, nearest_neighbor_idx[j]] for all j
    # Let's create a matrix where row i and col j contains d(j, k(j))
    # Actually, d(j, k(j)) depends only on j.
    # Let next_cost[j] = distance_matrix[j, nearest_neighbor_idx[j]]
    next_cost = distance_matrix[np.arange(n), nearest_neighbor_idx]
    
    # The heuristic eta[i, j] should be proportional to 1 / (d[i, j] + next_cost[j])
    # Note: next_cost[j] is the same for all i in column j.
    # So we can broadcast next_cost[j] across rows.
    
    # Avoid division by zero or negative values (though distances are positive)
    # d_ij + next_cost[:, None] adds the next cost to each column j
    denominator = d_ij + next_cost[:, None]
    
    # Ensure positivity
    denominator = np.maximum(denominator, 1e-9)
    
    eta = 1.0 / denominator
    
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # Our calculation ensures positive values if inputs are valid distances.
    # But we must handle any potential numerical issues.
    eta = np.where(eta > 0, eta, 1e-9)
    
    # Ensure finite values
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
