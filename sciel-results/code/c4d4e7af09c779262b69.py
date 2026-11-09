"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ACO orienteering.
    
    The heuristic is designed to encourage ants to visit high-prize nodes
    that are close to the current position and to the origin, while
    strongly penalizing long-distance moves.
    
    Key design choices:
    - Prize term: emphasizes high-value targets.
    - Distance penalty with exponent 3: sharp contrast between nearby and far nodes.
    - Origin proximity bonus: stabilizes early moves by favoring nodes close to node 0.
    - All entries are made strictly positive to avoid zero probabilities.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Prize attractiveness: higher prize = more attractive
    # Add a small base to avoid zero for prize-0 nodes
    prize_term = prizes[None, :] + 0.1
    
    # Distance penalty: sharper contrast with exponent 3
    # Use (1 + d)^(-3) so that d=0 gives 1 and larger d gives smaller values
    dist_from_i = distance_matrix  # shape (n, n), d[i, j]
    dist_penalty = (1.0 + dist_from_i) ** (-3.0)
    
    # Origin proximity bonus: nodes close to origin are slightly more attractive
    # This helps ants make reasonable early moves
    dist_to_origin = distance_matrix[:, 0]  # d[i, 0] for each i
    origin_bonus = 1.0 + 0.5 * (1.0 + dist_to_origin) ** (-2.0)
    # Broadcast to (n, n)
    origin_bonus_matrix = origin_bonus[:, None]
    
    # Combine: prize * distance_penalty * origin_bonus
    heuristic = prize_term * dist_penalty * origin_bonus_matrix
    
    # Ensure all entries are positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Normalize to a reasonable scale
    max_val = heuristic.max()
    if max_val > 0:
        heuristic = heuristic / max_val * 10.0
    
    return heuristic
# EVOLVE-BLOCK-END
