import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a greedy ratio heuristic matrix for ant colony optimization.
    
    The heuristic is:
        eta[i, j] = prizes[j] / (d[i, j] + d[j, 0] + eps)
    
    This represents the "prize density" or value per unit of length cost
    associated with moving from i to j and then returning to the depot.
    This structural change moves from a complex exponential penalty model
    to a direct reward-to-cost ratio, which is a fundamental heuristic
    in orienteering problems.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    
    n = len(prizes)
    
    # Ensure prizes[0] is 0 (depot has no prize)
    prizes = prizes.copy()
    prizes[0] = 0.0
    
    # d[j, 0] is the cost to return to depot from j
    d_to_depot = distance_matrix[:, 0]  # shape (n,)
    
    # Cost of moving from i to j and then back to depot
    # cost[i, j] = d[i, j] + d[j, 0]
    # We add a small epsilon to avoid division by zero
    eps = 1e-9
    cost = distance_matrix + d_to_depot[None, :] + eps  # shape (n, n)
    
    # Prize vector (broadcast over rows)
    # We use the prize at the destination node j
    prize_vec = prizes[None, :]  # shape (1, n)
    
    # Compute the greedy ratio
    eta = prize_vec / cost
    
    # For the diagonal (i == j), the ant can't move to itself.
    # Set to a very small value.
    np.fill_diagonal(eta, eps)
    
    # Ensure all values are positive and finite
    # The task states: "Heuristic entries that are not positive become 1e-9"
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)
    
    return eta
