import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Ensure positive values to avoid division by zero
    eps = 1e-9
    p = prizes + eps
    w = weights + eps
    
    # Calculate densities (prizes/weights) per constraint
    # Shape: (5, 100)
    densities = p / w
    
    # Sum across constraints for each item (Arithmetic Mean * 5)
    # Shape: (100,)
    heur = np.sum(densities, axis=0)
    
    # Ensure positive and finite
    heur = np.maximum(heur, 1e-9)
    heur = np.nan_to_num(heur, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    return heur
