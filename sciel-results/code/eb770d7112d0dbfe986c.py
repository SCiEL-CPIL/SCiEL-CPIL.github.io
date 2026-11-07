import numpy as np

def heuristics(prizes, weights):
    # weights shape (5, 100)
    # prizes shape (5, 100)? 
    # Task says: prizes and weights, of shape (5, 100)
    # Wait, "prizes" is usually a single value per item.
    # "Item scores... for 3 fixed instances... prizes and weights, of shape (5, 100)"
    # This implies prizes is also (5, 100)?
    # "prizes and weights ... are uniform in [0, 1]"
    # If prizes is (5, 100), does it mean 5 prizes per item?
    # Or is it a typo and prizes is (100,)?
    # "returns one attractiveness score per item"
    # "prizes and weights, of shape (5, 100)"
    # If prizes is (5, 100), maybe it's 5 different prize vectors?
    # But the task is "multiple-constraint knapsack". Usually 1 prize, 5 weights.
    # Let's check the seed code: `prizes / (weights.sum(axis=0) + 1e-9)`
    # If prizes is (5, 100) and weights is (5, 100), the result is (5, 100).
    # But the function must return "one attractiveness score per item", i.e., shape (100,).
    # The seed code `prizes / ...` would return (5, 100) if prizes is (5, 100).
    # This would be a "wrong shape" and result in invalid score (0).
    # BUT the current program scores 1.0.
    # This implies `prizes` is likely shape (100,) or the seed code is doing something else?
    # Or maybe `prizes` is (1, 100) and broadcasting works?
    # If `prizes` is (100,) and `weights.sum(axis=0)` is (100,), result is (100,).
    # If `prizes` is (5, 100), the seed code is broken.
    # Since the seed scores 1.0, `prizes` MUST be compatible with returning (100,).
    # Most likely `prizes` is (100,) or (1, 100).
    # The prompt says "prizes and weights, of shape (5, 100)". This is ambiguous.
    # "prizes and weights ... are uniform in [0, 1]".
    # If prizes were (5, 100), the seed would fail.
    # I will assume `prizes` is effectively a vector of length 100, or I need to reduce it.
    # However, if the prompt explicitly says shape (5, 100) for BOTH, and the seed works,
    # maybe the seed code `prizes / ...` implies `prizes` is (100,) and the prompt description is slightly loose or refers to the data generation?
    # Or maybe `prizes` is (5, 100) and the seed code is actually `prizes.sum(axis=0) / ...`?
    # No, the code is `prizes / ...`.
    # If I change the code, I must ensure the output is (100,).
    # If `prizes` is (5, 100), I should probably sum them or take the mean to get a single prize per item.
    # But "prize" is singular in knapsack.
    # Let's assume `prizes` is (100,) based on the seed working.
    # If `prizes` is (5, 100), I will sum it to be safe, or check the shape at runtime.
    
    # To be safe:
    if prizes.ndim == 2:
        # If prizes is (5, 100), sum along axis 0 to get (100,)
        # Or maybe it's (1, 100)?
        # If it's (5, 100), summing makes sense if they are components of value.
        # But standard knapsack has 1 value.
        # Let's assume the prompt meant `prizes` is (100,) and `weights` is (5, 100).
        # But if it IS (5, 100), I must handle it.
        # Let's sum prizes to get a scalar per item.
        p = np.sum(prizes, axis=0)
    else:
        p = prizes
        
    w_sum = np.sum(weights, axis=0)
    w_max = np.max(weights, axis=0)
    
    eps = 1e-9
    denom_seed = w_sum + eps
    denom_min = w_max + eps
    
    # Geometric Blend
    # eta = p / sqrt(w_sum * w_max)
    denom_blend = np.sqrt(w_sum * w_max) + eps
    
    # Let's try the Blend
    return p / denom_blend
