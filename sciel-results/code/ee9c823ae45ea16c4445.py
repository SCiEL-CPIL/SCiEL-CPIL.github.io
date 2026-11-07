import numpy as np
def heuristics(prizes, weights):
        # prizes: (5, 100) ? No, task says prizes and weights are shape (5, 100).
        # Wait, "prizes and weights, of shape (5, 100)".
        # Usually prizes is (1, 100) or (100,).
        # The prompt says: "prizes and weights, of shape (5, 100)".
        # This implies prizes is also (5, 100)?
        # "Item scores ... for 3 fixed instances ... prizes and weights, of shape (5, 100)"
        # This is ambiguous. Does "prizes" have shape (5,100)?
        # "Each constraint has capacity 25".
        # "best prize of that iteration".
        # "total prize found by any ant".
        # If prizes were (5,100), what is the "total prize"? Sum over items?
        # Usually, knapsack has one prize per item.
        # Let's re-read carefully: "prizes and weights, of shape (5, 100)".
        # It likely means `weights` is (5, 100) and `prizes` is... also (5, 100)?
        # Or is it a tuple `(prizes, weights)` where the *pair* has that shape?
        # No, "prizes and weights, of shape (5, 100)" usually applies to the variable mentioned or both.
        # If prizes is (5, 100), then each item has 5 prizes?
        # "total prize" would be sum of prizes?
        # Let's check the seed: `prizes / (weights.sum(axis=0) + 1e-9)`.
        # If `prizes` is (5, 100) and `weights` is (5, 100), `weights.sum(axis=0)` is (100,).
        # `prizes / (100,)` would broadcast if `prizes` is (5, 100)?
        # Result would be (5, 100).
        # But the return type is "array of one score per item", so shape (100,).
        # So `prizes` must be reducible to (100,) or the operation reduces it.
        # If `prizes` is (1, 100) or (100,), it works.
        # If `prizes` is (5, 100), `prizes.sum(axis=0)` would be (100,).
        # The seed does `prizes / ...`. If `prizes` is (5,100), this fails or returns (5,100).
        # The seed returns `prizes / ...`. If this returns (5,100), it's invalid ("wrong shape").
        # The seed scored 1.0, so it is valid.
        # Therefore, `prizes` MUST be of shape (100,) or (1, 100) or similar, such that the division results in (100,).
        # Wait, if `prizes` is (5, 100) and `weights.sum(axis=0)` is (100,), `prizes / denom` is (5, 100).
        # The task says "returns one attractiveness score per item".
        # So the seed `prizes / ...` implies `prizes` is likely (100,) or the user made a mistake in the seed description?
        # "prizes and weights, of shape (5, 100)" might mean `weights` is (5,100) and `prizes` is (100,).
        # Let's assume `prizes` is 1D (100,) and `weights` is 2D (5, 100).
        # This is the standard MCKP format.
        
        # Let's verify with the seed code:
        # return prizes / (weights.sum(axis=0) + 1e-9)
        # If prizes is (100,) and weights is (5,100), sum is (100,). Result (100,). Valid.
        # If prizes is (5,100), result is (5,100). Invalid.
        # So prizes is (100,).
        
        # Heuristic strategy:
        # Use max weight ratio.
        max_w = np.max(weights, axis=0)
        eta = prizes / (max_w + 1e-9)
        
        # Or use the sum ratio (seed) but modified?
        # The seed is 1.0.
        # Let's try a mix:
        sum_w = np.sum(weights, axis=0)
        # eta = prizes / (sum_w + max_w)
        
        # Let's try the max ratio. It is more robust to bottlenecks.
        # Also, ensure positivity.
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
