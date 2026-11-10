"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    # Convert to numpy arrays for speed
    p = np.asarray(prizes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Ensure no division by zero or negative values
    p = np.maximum(p, 1e-9)
    w = np.maximum(w, 1e-9)
    
    # Capacity is 25 for all constraints
    C = 25.0
    
    # Per-constraint efficiency: prize / (weight * capacity)
    # Shape: (5, 100)
    # e[i, j] = p[i, j] / (w[i, j] * C)
    # Note: prizes and weights are (5, 100) based on task description "prizes and weights, of shape (5, 100)"
    # Wait, let's re-read carefully.
    # "prizes and weights, of shape (5, 100)"
    # Usually in MCKP, prizes is (N,) and weights is (M, N).
    # Here, prizes is (5, 100)? That implies 5 prizes per item?
    # "Item scores... for 3 fixed instances with 100 items and 5 constraints."
    # "prizes and weights, of shape (5, 100)"
    # If prizes is (5, 100), does item 0 have 5 prizes?
    # "returns one attractiveness score per item"
    # "mean over instances of the highest total prize found by any ant"
    # "Each ant adds items... until none fits."
    # "best prize of that iteration"
    # If an ant picks a set of items, the "total prize" is likely the sum of prizes.
    # If prizes is (5, 100), maybe it's a multi-objective prize?
    # Or is the shape (100, 5)?
    # The seed code: `prizes / (weights.sum(axis=0) + 1e-9)`
    # If prizes is (5, 100) and weights is (5, 100):
    # weights.sum(axis=0) sums over the 5 constraints for each of the 100 items. Result shape (100,).
    # prizes / (100,) -> broadcasting. If prizes is (5, 100), it broadcasts to (5, 100) / (1, 100) -> (5, 100).
    # But the function must return "one attractiveness score per item", i.e., shape (100,).
    # The seed code returns `prizes / ...`. If `prizes` is (5, 100), the return is (5, 100).
    # This contradicts "one score per item" unless the evaluator sums them or takes the first row?
    # Let's look at the seed again.
    # `return prizes / (weights.sum(axis=0) + 1e-9)`
    # If the evaluator expects (100,), and the seed returns (5, 100), it might be an error or the evaluator handles it.
    # However, the task says "returns one attractiveness score per item".
    # It is highly probable that `prizes` is actually a vector of 100 prizes, or the shape description in the prompt is slightly misleading and `prizes` is (100,) and `weights` is (5, 100).
    # BUT the prompt explicitly says: "prizes and weights, of shape (5, 100)".
    # This is a contradiction. "One score per item" vs "prizes shape (5, 100)".
    # If `prizes` is (5, 100), maybe the "total prize" is the sum of the 5 prizes for each item?
    # Or maybe the 5 rows are 5 different prize vectors for 5 different objectives?
    # "mean over instances of the highest total prize found by any ant".
    # If an ant picks items $S$, the total prize is $\sum_{i \in S} \text{prize}_i$.
    # If $\text{prize}_i$ is a vector of 5, what is "total prize"?
    # Perhaps the 5 rows of `prizes` are just 5 different instances? No, "3 fixed instances".
    # Perhaps `prizes` is (100,) and `weights` is (5, 100)?
    # Let's assume the prompt's "shape (5, 100)" applies to `weights` primarily, and `prizes` might be (100,) or (5, 100) is a typo for weights only?
    # Let's look at the seed code again.
    # `weights.sum(axis=0)` implies `weights` has at least 2 dimensions, and summing axis 0 reduces it.
    # If `weights` is (5, 100), sum(axis=0) is (100,).
    # If `prizes` is (100,), `prizes / (100,)` is (100,). This matches "one score per item".
    # If `prizes` is (5, 100), `prizes / (100,)` is (5, 100). This does NOT match "one score per item".
    # Therefore, it is most likely that `prizes` is of shape (100,) or (1, 100), and `weights` is (5, 100).
    # OR, `prizes` is (5, 100) and the intended return is the sum over the 5 prize dimensions?
    # Given the ambiguity, I should write code that handles both or makes a robust assumption.
    # The safest assumption for "one score per item" is that the final return is (100,).
    # If `prizes` is (5, 100), I will sum the prizes across the 5 dimensions to get a scalar prize per item.
    # Let's check the seed: `prizes / ...`. If `prizes` was (5, 100), the seed would return (5, 100).
    # If the seed scored 1.0, and the evaluator accepted it, then the evaluator must be able to handle the seed's output.
    # If the seed's output is (5, 100), and the evaluator expects (100,), it would likely crash or score 0.
    # Since it scored 1.0, the seed's output shape MUST be compatible.
    # This implies `prizes` is likely (100,) or (1, 100) and the prompt's "shape (5, 100)" for prizes is a mistake, OR the evaluator sums the first axis.
    # However, if I return (100,), I am safe.
    # I will assume `prizes` might be (5, 100) and sum it to get (100,) to be safe, or if it's (100,), sum does nothing (or fails if 1D).
    # Let's check dimensions dynamically.
    
    if p.ndim == 2:
        # If prizes is (5, 100), sum over axis 0 to get total prize per item
        p_flat = p.sum(axis=0)
    else:
        p_flat = p
        
    # Now p_flat is (100,)
    # w is (5, 100)
    
    # Calculate efficiencies
    # e[i, j] = p_flat[j] / (w[i, j] * C)
    # We want to combine these 5 efficiencies per item.
    
    # Avoid division by zero
    denom = w * C
    denom = np.maximum(denom, 1e-9)
    
    eff = p_flat[None, :] / denom  # Shape (5, 100)
    
    # Geometric Mean
    # log(eff) -> mean -> exp
    # Clip eff to avoid log(0)
    eff_clipped = np.maximum(eff, 1e-9)
    
    log_eff = np.log(eff_clipped)
    mean_log_eff = log_eff.mean(axis=0)
    
    heuristic = np.exp(mean_log_eff)
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
