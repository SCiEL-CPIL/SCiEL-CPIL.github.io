"""Lagrangian Relaxation-based edge weight heuristic for CVRP ant colony."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using a Lagrangian Relaxation approach.
    
    This implementation derives edge weights by averaging the inverse Lagrangian costs
    over a spectrum of dual variables (Lagrange multipliers). This captures edges that
    are attractive under various trade-offs between distance and demand, providing a
    robust heuristic derived from a dual optimization perspective.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=float)
    demands[0] = 0.0  # Depot has no demand
    
    # Determine a physically meaningful range for the Lagrange multiplier lambda.
    max_dist = np.max(distance_matrix)
    avg_demand = np.mean(demands[1:])
    lambda_balance = max_dist / max(avg_demand, 1e-6)
    
    # Sample lambda values in a log-scale range to cover different sensitivities
    num_lambdas = 15
    lambda_min = 0.01 * lambda_balance
    lambda_max = 10.0 * lambda_balance
    
    # Include lambda=0 (pure distance) and the geometric progression
    lambdas = np.geomspace(lambda_min, lambda_max, num_lambdas)
    lambdas = np.concatenate(([0.0], lambdas))
    
    # Initialize sum of inverse costs
    H_sum = np.zeros_like(distance_matrix, dtype=float)
    
    p = 1.5  # Sharpness parameter
    
    for lam in lambdas:
        # Compute Lagrangian cost matrix: W = D + lambda * d
        W = distance_matrix + lam * demands
        W = np.maximum(W, 1e-9)
        
        # Compute inverse cost raised to power p
        H_part = 1.0 / (W ** p)
        
        # Accumulate
        H_sum += H_part
        
    # Average the contributions
    H = H_sum / len(lambdas)
    
    # Ensure finite values
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # Normalize to ensure comparable magnitude
    max_val = np.max(H)
    if max_val > 0:
        H = H / max_val
        
    return H
# EVOLVE-BLOCK-END
