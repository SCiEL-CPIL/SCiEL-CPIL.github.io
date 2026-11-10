"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for Ant Colony Optimization.
    
    The heuristic favors edges (i, j) that lead to nodes with high prizes
    and low combined cost of moving to j and returning to start.
    
    Formula: eta[i, j] = prize[j] / (distance[i, j] + distance[j, 0] + epsilon)
    """
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    # Ensure shapes are compatible
    n = prizes.shape[0]
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-9
    
    # distance[j, 0] is the cost to return to start from j
    # We broadcast this to shape (n, 1) to add to distance_matrix (n, n)
    # distance_matrix[i, j] is cost i -> j
    # return_cost[j] is cost j -> 0
    return_cost = distance_matrix[:, 0]  # Shape (n,)
    
    # Total cost of visiting j from i and then returning to start
    # This is a proxy for the "budget consumption" of this move in the context of the tour constraint
    # Note: The actual constraint is len_so_far + d[i,j] + d[j,0] <= max_len.
    # Since len_so_far varies, we can't put it in the static heuristic.
    # But d[i,j] + d[j,0] is the static part of the constraint check for this specific move's feasibility 
    # relative to the remaining budget.
    
    total_cost = distance_matrix + return_cost[None, :] + epsilon
    
    # Heuristic: Prize / Total Cost
    heuristic = prizes[None, :] / total_cost
    
    # Handle any non-finite values (shouldn't happen with epsilon, but safe)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
