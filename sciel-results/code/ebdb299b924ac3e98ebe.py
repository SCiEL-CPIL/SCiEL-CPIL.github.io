import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 1.0 # 1 second safety margin for a trivial operation
    
    # bins contains remaining capacity
    # We want to pick the bin with the most remaining capacity after placing the item (Worst-Fit)
    # This preserves space for larger future items.
    
    # Calculate new remaining capacity if item is placed in each bin
    new_rem = bins - item
    
    # Invalid bins have new_rem < 0.
    # Valid bins have new_rem >= 0.
    # We want to maximize new_rem.
    # So we return new_rem as the score.
    # Invalid bins will have negative scores, valid bins positive/zero.
    # The evaluator picks the max score.
    
    scores = new_rem
    
    # Check deadline (just in case, though this is O(N) vectorized)
    if time.time() > deadline:
        # Fallback to a simple valid bin if we timed out (unlikely)
        # Just return the scores calculated so far
        pass
        
    return scores
