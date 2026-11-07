"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    # Deadline guard (trivial for this function, but required by directive)
    start_time = time.time()
    deadline = start_time + 300 # 300s limit, use most of it if needed, but this is fast
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Check time, though unlikely to be hit
    if time.time() > deadline - 1:
         # Return a simple fallback if we are somehow stuck (should not happen)
         return np.ones((n, n)) * 1e-9

    d_j0 = distance_matrix[:, 0]  # (n,) return cost for each node
    
    eps = 1e-9
    
    # Marginal gain approach: for each potential move i->j, estimate the
    # remaining budget after taking this step and returning to start.
    # The heuristic should reward moves that maximize prize density per unit
    # of budget consumed, while preferring moves that leave more slack.
    
    # Total path cost if we go i->j->0: d[i,j] + d[j,0]
    # Remaining budget: max_len - (d[i,j] + d[j,0])
    total_cost = distance_matrix + d_j0[None, :]  # (n, n)
    
    # Only consider moves that are feasible (remaining budget > 0)
    remaining_budget = max_len - total_cost
    
    # Prize density: prize[j] / (budget consumed by this move)
    # Budget consumed = d[i,j] + d[j,0]
    budget_consumed = np.maximum(total_cost, eps)
    
    # Prize term: prize[j] for each target node j
    prize_term = prizes[None, :]  # (1, n)
    
    # Marginal gain: prize[j] / budget_consumed[i,j]
    # This rewards high-prize nodes that are cheap to reach
    marginal_gain = prize_term / budget_consumed
    
    # Slack bonus: moves that leave more remaining budget get a boost
    # This encourages ants to not exhaust their budget too quickly
    slack_bonus = np.maximum(remaining_budget, 0.0)
    
    # Combine: marginal gain weighted by slack
    # If remaining budget is high, the move is "safe" and we weight the gain more
    # If remaining budget is low, we still consider it but with less emphasis
    weight = 1.0 + 0.5 * slack_bonus
    
    heuristic = marginal_gain * weight
    
    # Only allow feasible moves (positive remaining budget)
    feasible_mask = remaining_budget > 0
    heuristic = np.where(feasible_mask, heuristic, eps)
    
    # Row-normalize to ensure probabilities sum reasonably
    # Use max normalization to preserve relative ordering better
    row_max = np.max(heuristic, axis=1, keepdims=True)
    row_max = np.maximum(row_max, eps)
    heuristic = heuristic / row_max
    
    # Set diagonal to very small (can't move to self)
    np.fill_diagonal(heuristic, eps)
    
    # Ensure positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
