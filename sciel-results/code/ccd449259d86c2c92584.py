import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    The heuristic $\eta_{ij}$ represents the attractiveness of moving from node i to node j.
    We define it as:
    $\eta_{ij} = \frac{prize_j}{d_{ij} \cdot d_{j0}}$
    
    Where:
    - $prize_j$ is the prize at node j.
    - $d_{ij}$ is the distance between i and j.
    - $d_{j0}$ is the distance between j and the depot (node 0).
    
    This formulation favors nodes that have high prizes, are close to the current node i,
    and are close to the depot (making them easier to fit within the length constraint).
    """
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    # Ensure shapes are correct
    # distance_matrix should be (N, N)
    # prizes should be (N,)
    
    n = distance_matrix.shape[0]
    
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    
    d = distance_matrix + eps
    d_to_depot = d[:, 0]
    
    # Intrinsic attractiveness of each node j
    # High prize, low distance to depot
    node_attraction = prizes / d_to_depot
    
    # Edge attractiveness from i to j
    # Node j's intrinsic attraction divided by the step distance
    eta = node_attraction[None, :] / d
    
    # Ensure all values are positive and finite
    # Replace any non-finite or non-positive values with 1e-9
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
