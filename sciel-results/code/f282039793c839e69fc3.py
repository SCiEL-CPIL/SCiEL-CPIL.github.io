"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=np.float64)
    
    # Use a moderately sharp heuristic: 1/d^1.5
    # This balances greediness (favoring short edges) with exploration.
    # 1/d^2 was too sharp, potentially causing premature convergence.
    # 1/d^1.5 provides a smoother gradient in attractiveness.
    
    # Add small epsilon to avoid division by zero on diagonal
    # The evaluator will replace non-positive values with 1e-9,
    # but we must return finite values.
    eta = 1.0 / (np.power(d, 1.5) + 1e-9)
    
    # Ensure all values are finite and positive
    # For d=0, eta = 1e9 (finite)
    # For d>0, eta is positive and finite
    
    return eta
# EVOLVE-BLOCK-END
