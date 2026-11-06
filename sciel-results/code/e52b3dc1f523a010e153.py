import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    
    # Heuristic: 
    # For small items, use Best-Fit (minimize remaining space) to pack tightly.
    # For large items, use Worst-Fit (maximize remaining space) to keep bins open for other large items.
    
    # Threshold: 25% of capacity (100)
    threshold = 25.0
    
    if item > threshold:
        # Maximize remaining space
        scores = np.where(valid, remaining, -np.inf)
    else:
        # Minimize remaining space (Maximize negative remaining)
        scores = np.where(valid, -remaining, -np.inf)
        
    return scores
