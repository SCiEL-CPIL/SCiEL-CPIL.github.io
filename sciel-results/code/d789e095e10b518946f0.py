import numpy as np

def heuristics(prizes, weights):
    # weights shape (5, 100)
    # prizes shape (5, 100) -- Wait, the prompt says "prizes and weights, of shape (5, 100)".
    # "Item scores ... returns one attractiveness score per item".
    # Usually prizes are per item, not per constraint.
    # Let's re-read carefully: "prizes and weights, of shape (5, 100)".
    # This implies prizes is also (5, 100)?
    # "Every constraint has capacity 25".
    # "highest total prize found by any ant".
    # If prizes is (5, 100), does the total prize depend on the constraint?
    # Usually in MCKP, there is one prize per item.
    # However, the shape (5, 100) for prizes is unusual if it's just a scalar per item.
    # Let's look at the seed program: `return prizes / (weights.sum(axis=0) + 1e-9)`.
    # If `prizes` is (5, 100), then `prizes / ...` results in (5, 100).
    # But the function must return "one attractiveness score per item", i.e., shape (100,).
    # The seed program returns `prizes / ...`. If `prizes` is (5,100), the output is (5,100).
    # This contradicts "returns one attractiveness score per item".
    # Unless `prizes` is actually (100,) in the seed's context?
    # "prizes and weights, of shape (5, 100)" is in the description.
    # Let's assume the description is precise.
    # If `prizes` is (5, 100), how do we get a single score per item?
    # Maybe the prize of an item is the sum of prizes across constraints? Or the max?
    # Or maybe the "prizes" input is actually just the item values, and the shape description is a typo for weights only?
    # "prizes and weights, of shape (5, 100)". This phrasing usually applies to both.
    # Let's check the seed again.
    # If I run the seed, and it returns (5,100), and the evaluator expects (100,), it would fail or broadcast?
    # The prompt says: "An exception, non-finite values or a wrong shape make the whole result invalid."
    # So the seed *must* be returning a valid shape.
    # If `prizes` is (5, 100), `prizes / weights.sum(axis=0)` is (5, 100) / (100,) -> (5, 100).
    # This is NOT "one score per item".
    # Therefore, it is highly likely that `prizes` is actually shape (100,) or the seed program is implicitly summing/squeezing?
    # No, the seed code is explicit: `return prizes / ...`.
    # If the seed scores 1.0, it is valid.
    # This implies either:
    # 1. `prizes` is shape (100,). The description "prizes and weights, of shape (5, 100)" might be a slight inaccuracy, meaning weights is (5,100) and prizes is (100,).
    # 2. The evaluator accepts (5,100) and sums it? Unlikely.
    # 3. The evaluator expects (100,) and the seed is actually returning (100,) because `prizes` is (100,).
    
    # Let's assume `prizes` is (100,) and `weights` is (5, 100).
    # This is the standard MCKP formulation: one value per item, multiple weights per item.
    
    # Let's verify with the text: "prizes and weights, of shape (5, 100)".
    # If this is strictly true, then `prizes` has 5 rows.
    # If I sum `prizes` along axis 0, I get (100,).
    # If I take `prizes.max(axis=0)`, I get (100,).
    # Which one is the "total prize"?
    # "highest total prize found by any ant".
    # If an ant picks item i, does it get `prizes[:, i].sum()` or `prizes[:, i].max()`?
    # Usually, "total prize" is the sum of values of picked items.
    # If `prizes` is (5,100), maybe the value of item i is `prizes[:, i].sum()`?
    # Or maybe the problem is 5 independent knapsacks? No, "multiple-constraint knapsack" is a single knapsack with 5 constraints.
    # It is most probable that `prizes` is (100,) and the description is slightly loose, OR `prizes` is (5,100) and the value is the sum.
    
    # Let's look at the seed again.
    # If `prizes` were (5,100), the seed would return (5,100).
    # If the evaluator rejects (5,100), the seed would score 0.
    # The seed scores 1.0.
    # So the seed's output shape is valid.
    # If the required output is (100,), then the seed's output MUST be (100,).
    # This implies `prizes` in the seed's execution environment is (100,).
    # Therefore, I will assume `prizes` is (100,) and `weights` is (5, 100).
    
    # Code:
    # weights: (5, 100)
    # prizes: (100,)
    
    # Heuristic: prizes / (max_weight + eps)
    max_w = weights.max(axis=0)
    return prizes / (max_w + 1e-9)
