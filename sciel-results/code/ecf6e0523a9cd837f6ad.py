"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Structural change: use a normalized, rank-based heuristic that emphasizes
    # prize-to-cost ratio with a different functional form.
    # Instead of prize^alpha / cost^beta, we use:
    #   heuristic[i,j] = (prize_j / (d[i,j] + d[j,0])) * log(1 + prize_j)
    # This gives a multiplicative boost to high-prize nodes while keeping the
    # cost in the denominator. The log term provides sublinear amplification.
    
    # Marginal cost: d[i,j] + d[j,0]
    d_j0 = distance_matrix[:, 0]
    marginal_cost = distance_matrix + d_j0[None, :]
    
    # Avoid division by zero
    eps = 1e-9
    safe_cost = np.maximum(marginal_cost, eps)
    
    # Prize ratio: prize_j / marginal_cost[i,j]
    prize_ratio = prizes[None, :] / safe_cost
    
    # Log amplification: log(1 + prize_j)
    log_amp = np.log1p(prizes[None, :])
    
    # Combined heuristic
    heuristic = prize_ratio * log_amp
    
    # Normalize per row to have mean 1, so the scale is consistent
    row_means = np.mean(heuristic, axis=1, keepdims=True)
    row_means = np.maximum(row_means, eps)
    heuristic = heuristic / row_means
    
    # Set diagonal to very small (no self-moves)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
