import numpy as np
import time

def heuristics(prizes, weights):
    """
    Calculates attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Args:
        prizes: np.array of shape (5, 100) with prize values.
        weights: np.array of shape (5, 100) with weight consumptions.
        
    Returns:
        np.array of shape (100,) with heuristic scores.
    """
    start_time = time.time()
    
    # Convert inputs to numpy arrays for consistent vectorized operations
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure shapes are as expected (5, 100)
    # If shapes are different, we might need to handle it, but the task guarantees (5, 100)
    
    # Calculate the total prize value for each item (sum across the 5 dimensions)
    # This assumes the objective is to maximize the sum of all prize components.
    total_prizes = prizes.sum(axis=0)
    
    # Calculate the total weight cost for each item (sum across the 5 constraints)
    # Using the sum of weights serves as a scalar proxy for the multi-dimensional cost.
    total_weights = weights.sum(axis=0)
    
    # Calculate the heuristic score: Value / Cost
    # Add a small epsilon to the denominator to prevent division by zero
    heuristic = total_prizes / (total_weights + 1e-9)
    
    # Handle non-finite values (inf, nan) by replacing with a small positive number
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Ensure all heuristic entries are positive, as required by the task
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Deadline guard: Although this function is very fast, we check elapsed time
    # to comply with the directive to have an explicit deadline guard.
    # The limit is 330s for the whole eval, so we use a generous margin here.
    if time.time() - start_time > 290:
        # Fallback: Return a uniform heuristic if we are somehow stuck (unlikely)
        return np.ones(prizes.shape[1], dtype=float)
        
    return heuristic
