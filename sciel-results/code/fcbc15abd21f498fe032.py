import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    demands = np.array(demands)
    
    # Pair sums
    pair_sum = demands[:, None] + demands[None, :]
    
    # Remaining gap
    gap = capacity - pair_sum
    
    # Initialize with 0 (invalid)
    H = np.zeros((n, n))
    
    # Valid pairs: gap >= 0
    valid = gap >= 0
    
    # Minimum item size
    min_item = np.min(demands)
    
    # Usable gaps: gap >= min_item
    usable = valid & (gap >= min_item)
    
    # Wasted gaps: 0 <= gap < min_item
    wasted = valid & (gap < min_item)
    
    # For usable gaps, prefer smaller gaps (tighter fit)
    # Heuristic value is inversely proportional to gap
    H[usable] = 1.0 / (gap[usable] + 1.0)
    
    # For wasted gaps, assign a low constant value
    # This value should be lower than the minimum value for usable gaps
    # Min usable gap is min_item, so min usable H is 1/(min_item + 1)
    # Let's set wasted H to half of that, or a small epsilon
    # To ensure it's "attractive" enough to be considered if no better option, but discouraged
    base_waste_val = 1.0 / (min_item + 1.0) * 0.5
    H[wasted] = base_waste_val
    
    # Ensure non-finite or invalid are handled (already 0, but task says non-positive becomes 1e-9)
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # So we can leave invalid as 0, and the evaluator will handle it? 
    # "An exception, non-finite values or a wrong shape make the whole result invalid."
    # "Heuristic entries that are not positive become 1e-9" implies the evaluator does this replacement? 
    # Or should we? "Write heuristics... returns an item-by-item matrix... Heuristic entries that are not positive become 1e-9"
    # Usually this implies the evaluator sanitizes it, or we should. 
    # The current program sets invalid to 1e-9 explicitly. I will do the same to be safe.
    
    H[~valid] = 1e-9
    
    return H
