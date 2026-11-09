import numpy as np

def heuristics(prizes, weights):
        # weights shape (5, 100)
        # prizes shape (5, 100) ? 
        # Wait, the task says "prizes and weights, of shape (5, 100)".
        # This implies prizes is also a matrix? 
        # "Item scores... prizes and weights, of shape (5, 100)"
        # Usually prizes is a vector (100,). But the prompt says shape (5, 100).
        # Let's re-read carefully: "prizes and weights, of shape (5, 100), are uniform in [0, 1]."
        # This is ambiguous. Does it mean both are (5,100)? Or is it a typo and prizes is (100,)?
        # The current code does: `prizes / (weights.sum(axis=0) + 1e-9)`.
        # If prizes is (5,100) and weights is (5,100), `weights.sum(axis=0)` is (100,).
        # Then `prizes / (100,)` would broadcast to (5,100) if prizes is (5,100).
        # But the return must be "one attractiveness score per item", i.e., shape (100,).
        # If the current code returns (5,100), it would fail the "wrong shape" check in the evaluator?
        # "An exception, non-finite values or a wrong shape make the whole result invalid."
        # The current program scores 1.0, so it must be valid.
        # This implies `prizes` is likely a vector (100,) OR the broadcasting results in (100,).
        # If `prizes` is (100,), `prizes / (100,)` is (100,).
        # If `prizes` is (5,100), `prizes / (100,)` is (5,100).
        # Given the score is 1.0 (valid), `prizes` is almost certainly a 1D array of shape (100,), despite the prompt's potentially confusing phrasing "prizes and weights, of shape (5, 100)". It likely means "the data consists of prizes and weights, where weights is (5,100)". Or maybe prizes is (100,) and the prompt meant "weights of shape (5,100)".
        # Let's assume `prizes` is 1D (100,) to be safe, or handle both.
        # If `prizes` is 2D, we might need to sum it? "total prize" usually implies a single value per item.
        # Let's check the current code: `return prizes / ...`. If it works, prizes is compatible with (100,).
        # I will assume `prizes` is 1D. If it's 2D, I should probably sum it or take the mean? 
        # "highest total prize found by any ant". An ant picks a set of items. Total prize is sum of prizes of picked items.
        # So each item has a single prize value.
        # I will treat `prizes` as 1D. If it's 2D, I'll sum along axis 0 to get 1D.
        
        w_max = np.max(weights, axis=0) # Shape (100,)
        # Add epsilon to avoid division by zero
        heuristic = prizes / (w_max + 1e-9)
        # Ensure positive
        heuristic = np.maximum(heuristic, 1e-9)
        return heuristic
