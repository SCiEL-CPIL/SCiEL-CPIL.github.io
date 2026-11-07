import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10.0  # Very fast, but guard just in case
    
    # Handle shapes
    if prizes.ndim == 2:
        P = prizes.sum(axis=0)  # (100,)
    else:
        P = np.asarray(prizes).flatten()
        
    if weights.ndim == 2:
        W = weights  # (5, 100)
    else:
        W = weights.reshape(1, -1)
        
    n_constraints, n_items = W.shape
    
    # Transpose W to (100, 5) for easier broadcasting with items
    W_T = W.T  # (100, 5)
    
    # Check for non-finite values in input, replace with 0 to avoid NaN propagation
    P = np.where(np.isfinite(P), P, 0.0)
    W_T = np.where(np.isfinite(W_T), W_T, 0.0)
    
    # Pairwise Dominance Calculation
    # Item i dominates Item j if:
    # 1. P_i >= P_j
    # 2. W_i[c] <= W_j[c] for all c
    # 3. (P_i > P_j) OR (exists c such that W_i[c] < W_j[c])
    
    # Condition 1: P_i >= P_j
    # P[:, None] is (100, 1), P[None, :] is (1, 100) -> result (100, 100)
    cond_prize_ge = P[:, None] >= P[None, :]
    
    # Condition 2: W_i <= W_j for all constraints
    # W_T[:, None, :] is (100, 1, 5), W_T[None, :, :] is (1, 100, 5)
    # Result (100, 100, 5)
    cond_weights_le = W_T[:, None, :] <= W_T[None, :, :]
    cond_all_weights = np.all(cond_weights_le, axis=2)  # (100, 100)
    
    # Condition 3: Strict improvement
    # P_i > P_j
    cond_prize_gt = P[:, None] > P[None, :]
    
    # Any weight strictly less
    cond_weights_lt = W_T[:, None, :] < W_T[None, :, :]
    cond_any_weight_lt = np.any(cond_weights_lt, axis=2)  # (100, 100)
    
    # Strictly better if prize is strictly higher OR (prize equal and some weight strictly lower)
    # Note: If P_i > P_j, it's strictly better regardless of weights (as long as weights <=).
    # If P_i == P_j, it must have at least one weight strictly lower.
    cond_strict = cond_prize_gt | ((~cond_prize_gt) & cond_any_weight_lt)
    
    # Full Dominance Mask: i dominates j
    dominance_mask = cond_prize_ge & cond_all_weights & cond_strict
    
    # Count how many items each item dominates
    # dominance_mask[i, j] is True if i dominates j
    # Sum over j (axis=1) to get count for each i
    dominance_counts = np.sum(dominance_mask, axis=1).astype(np.float64)
    
    # The heuristic score is the dominance count.
    # Items that dominate more others are "more attractive" as they are structurally superior.
    # Add a small epsilon to ensure all values are positive and non-zero for ACO.
    # Using 1.0 + count ensures that even non-dominated items (count 0) have a baseline attractiveness.
    H = 1.0 + dominance_counts
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H), H, 1.0)
    H = np.maximum(H, 1e-9)
    
    # Optional: Normalize to mean 1 to keep magnitude consistent with other heuristics?
    # The ACO algorithm uses pheromone * heuristic. If heuristic magnitude changes,
    # the relative importance of pheromone vs heuristic changes.
    # The seed program returned values that were not normalized to mean 1 explicitly, 
    # but were ratios. 
    # Let's check the scale. Counts are 0-99. 
    # If we don't normalize, the heuristic is very strong. 
    # In ACO, if heuristic is very strong, ants follow it greedily. 
    # This is often good for MCKP. 
    # However, to be safe and comparable to previous "ratio" heuristics which might be O(1), 
    # we might want to scale it. 
    # But the directive says "use this count as the heuristic score". 
    # Let's stick to the raw count + 1. It directly reflects the Pareto structure.
    
    if time.time() > deadline - 1.0:
        return H
        
    return H
