"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    
    # Deadline guard: keep a safety margin before the 300s limit
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Precompute key arrays
    d_from_0 = D[0, :]  # shape (n,)
    d_to_0 = D[:, 0]    # shape (n,)
    
    # For each node j, compute:
    # 1. prize[j]: the prize at node j
    # 2. cost_to_reach[j] = d[0, j]: minimum cost to reach j from start
    # 3. cost_to_return[j] = d[j, 0]: cost to return from j to start
    # 4. total_min_cost[j] = d[0, j] + d[j, 0]: minimum round-trip cost for j
    
    prize = prizes.copy()
    cost_reach = d_from_0.copy()
    cost_return = d_to_0.copy()
    
    # Structural change: Use a greedy constructive heuristic that models
    # the orienteering problem as a prize-to-travel-cost ratio with 
    # feasibility-aware weighting.
    
    # The key insight: an ant at node i considering node j should value:
    # - High prize at j
    # - Low incremental cost (d[i,j])
    # - Feasibility: the path 0->...->i->j->0 must fit in max_len
    
    # Since we don't know the exact path length to i, we use the minimum 
    # possible path length to i (which is d[0,i] for direct, but actually
    # it's the shortest path from 0 to i in the graph).
    
    # For a better heuristic, we model the expected remaining budget.
    # If an ant is at node i, the minimum path length so far is at least d[0,i].
    # The maximum additional distance it can travel before returning is:
    # max_len - d[0,i] - d[i,0] (but this is a lower bound on remaining budget)
    
    # Actually, let's think differently. The heuristic should guide ants 
    # toward nodes that offer good prize-to-cost ratios while being feasible.
    
    # Compute for each pair (i, j):
    # - prize[j] / d[i,j]: prize density of moving from i to j
    # - feasibility factor: how much budget remains after i->j->0
    
    # The minimum path length if ant went 0->i->j->0 directly is:
    # d[0,i] + d[i,j] + d[j,0]
    # This must be <= max_len for feasibility.
    
    # So the "slack" for transition i->j is:
    # slack[i,j] = max_len - d[0,i] - d[i,j] - d[j,0]
    
    # But this is a lower bound on actual slack (since the ant may have taken
    # a longer path to i). Still, it's a reasonable proxy.
    
    d_0i = d_from_0[:, np.newaxis]  # shape (n, 1)
    d_j0 = d_to_0[np.newaxis, :]    # shape (1, n)
    
    slack = max_len - d_0i - D - d_j0  # shape (n, n)
    
    # Prize density: prize[j] / d[i,j]
    d_safe = D + 1e-9
    prize_density = prize[np.newaxis, :] / d_safe  # shape (n, n)
    
    # Structural change: Use a multiplicative model where the heuristic is
    # prize_density * exp(-alpha * (1 - slack/max_len)) for feasible transitions,
    # and a small value for infeasible ones.
    
    # This creates a smooth transition from highly attractive (high slack) to
    # less attractive (low slack) while maintaining feasibility awareness.
    
    # Alternative: Use a power-law weighting on slack
    # H[i,j] = prize[j] / d[i,j] * (max(0, slack[i,j]))^gamma
    
    # Let's try gamma = 1.0 first (linear in slack)
    slack_clipped = np.clip(slack, 0.0, None)
    
    # Normalize slack to [0, 1] for better numerical stability
    max_slack = max_slack_val = np.max(slack_clipped) if np.any(slack_clipped > 0) else 1.0
    if max_slack_val > 0:
        slack_norm = slack_clipped / max_slack_val
    else:
        slack_norm = np.zeros_like(slack_clipped)
    
    # Heuristic: prize density weighted by normalized slack
    H = prize_density * slack_norm
    
    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(H, 0.0)
    
    # Ensure node 0 column is zero (ants start at 0, don't go back to 0 mid-path
    # in a way that's captured here; but actually they can visit 0? No, 0 is start)
    # Actually, the problem says "moves to an unvisited node j", and 0 is visited at start.
    # So column 0 should be zero.
    H[:, 0] = 0.0
    
    # Ensure all values are finite
    H = np.nan_to_num(H, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # Ensure positive values (evaluator will clip non-positive to 1e-9, but let's be safe)
    H = np.where(H > 0, H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
