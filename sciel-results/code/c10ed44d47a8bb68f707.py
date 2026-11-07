"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the Ant Colony Optimization.
    
    Uses a marginal gain approach: weights prize[j] by the slack remaining
    after visiting j from i and returning to origin. This prioritizes nodes
    that maximize prize density per unit of remaining path budget.
    
    eta[i, j] = (prize[j] * slack[i, j]) / (d[i, j] + d[j, 0] + epsilon)
    where slack[i, j] = max_len - d[i, j] - d[j, 0]
    """
    # Get dimensions
    n = len(prizes)
    
    # Epsilon for numerical stability
    epsilon = 1e-9
    
    # Vector of prizes
    # prizes shape: (n,)
    # We need prize[j] for all i, j. So we broadcast prizes to (1, n)
    prize_row = prizes[None, :]
    
    # Distance to origin d[j, 0]
    # distance_matrix[:, 0] has shape (n,)
    # We broadcast it to (1, n) to add to distance_matrix (n, n)
    dist_to_origin = distance_matrix[:, 0][None, :]
    
    # Calculate the combined cost in the denominator
    # cost[i, j] = d[i, j] + d[j, 0]
    cost_matrix = distance_matrix + dist_to_origin
    
    # Calculate slack: remaining budget after visiting j from i and returning
    # slack[i, j] = max_len - cost[i, j]
    slack = max_len - cost_matrix
    
    # Clip slack to be non-negative (if cost exceeds max_len, slack is 0)
    slack = np.maximum(slack, 0.0)
    
    # Calculate heuristic: prize weighted by slack, divided by cost
    # This favors nodes with high prize AND good slack (efficient use of budget)
    # eta[i, j] = (prize[j] * slack[i, j]) / (cost[i, j] + epsilon)
    numerator = prize_row * slack
    eta = numerator / (cost_matrix + epsilon)
    
    # Ensure non-positive values are replaced by 1e-9
    # The spec says "Heuristic entries that are not positive become 1e-9".
    eta = np.where(eta > 0, eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
