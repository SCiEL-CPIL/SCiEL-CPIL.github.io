import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # Calculate max weight per item across constraints
    # weights.max(axis=0) gives shape (100,)
    max_weights = weights.max(axis=0)
    
    # Calculate sum of prizes per item
    # prizes.sum(axis=0) gives shape (100,)
    total_prizes = prizes.sum(axis=0)
    
    # Heuristic: Prize / (Max Weight + epsilon)
    # This favors items with high total prize and low maximum weight in any single constraint.
    # This promotes "compatible" items that fit well with others.
    epsilon = 1e-9
    heur = total_prizes / (max_weights + epsilon)
    
    # Ensure positive values as per task description (though max_weights >= 0, 
    # if max_weights is 0, heur is inf. But weights are uniform [0,1], so 0 is unlikely 
    # but possible. The task says "Heuristic entries that are not positive become 1e-9".
    # So we just return the raw values, and the ACO code handles the non-positive check.
    # However, if max_weights is 0, division by zero -> inf. 
    # Uniform [0,1] means 0 is possible.
    # Let's add epsilon to denominator to avoid inf.
    
    return heur
