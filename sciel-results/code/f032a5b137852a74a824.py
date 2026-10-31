import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining space
    remaining = bins - item
    
    # Only consider bins that can fit the item
    feasible_mask = remaining >= 0
    
    # If no feasible bins, return very low scores (or handle error, but contract implies valid input)
    if not np.any(feasible_mask):
        return np.full_like(bins, -np.inf)
        
    # Sigmoid weight for Best Fit vs First Fit
    # For large items (item > 10), weight -> 1 (Best Fit)
    # For small items (item < 10), weight -> 0 (First Fit)
    # Transition centered at 10, slope 0.5
    alpha = 1.0 / (1.0 + np.exp(-0.5 * (item - 10)))
    
    # Best Fit Score: prefer smallest remaining space
    # We want higher score for smaller remaining space.
    # So score = -remaining.
    score_bf = -remaining
    
    # First Fit Score: prefer lowest index.
    # We can represent this by giving a score that is 0 for all, 
    # but to make argmax pick the first feasible bin, we can just rely on 
    # the fact that if all scores are equal, argmax picks the first index.
    # However, mixing with BF score:
    # If alpha is 0, score is 0 for all feasible bins. Argmax picks first feasible.
    # If alpha is 1, score is -remaining. Argmax picks min remaining.
    
    # Initialize scores with -inf for infeasible bins
    scores = np.full_like(bins, -np.inf)
    
    # Calculate scores for feasible bins
    # We use the formula: Score = alpha * (-remaining) + (1-alpha) * 0
    # This simplifies to alpha * (-remaining)
    scores[feasible_mask] = alpha * (-remaining[feasible_mask])
    
    return scores
