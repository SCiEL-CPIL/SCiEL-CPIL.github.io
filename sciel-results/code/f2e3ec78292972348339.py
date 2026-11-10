"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Hybrid Lagrangian-Greedy with Bottleneck Slack Feasibility.
    
    1. Problem Framing:
       The multiple-constraint knapsack problem requires balancing high prize
       against the consumption of 5 distinct resources. Static density ratios
       (prize/weight) fail because they ignore the discrete "blocking" effect:
       an item with a great ratio might be useless if it consumes capacity in
       a constraint that is already tight for other high-value items.
       
    2. Method:
       We combine two complementary signals:
       A) Lagrangian Dual Approximation: Estimates shadow prices (lambda) for
          each constraint. Items with high positive reduced cost (prize - 
          shadow_cost) are "profitable" after paying for resources. This
          provides a global, smooth ranking.
       B) Multi-Start Greedy Simulation: Simulates the actual packing process
          with multiple sorting criteria. Items that frequently appear in
          high-value feasible packs are "safe" and "valuable". This captures
          the discrete blocking effects.
       C) Bottleneck Slack: A direct feasibility measure. For each item, we
          compute the minimum remaining capacity ratio across all constraints
          if that item were added first: min_c (capacity - w_ic) / w_ic.
          This prioritizes items that fit easily into the tightest constraints.
       
    3. Heuristic Construction:
       - Run Lagrangian dual to get reduced costs (rc).
       - Run multiple greedy passes to get weighted selection counts (wc).
       - Compute bottleneck slack (bs).
       - Combine these signals: 
         Score_i = (rc_i^alpha) * (wc_i^beta) * (bs_i^gamma) * (p_i^delta)
       - Normalize and ensure positivity.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shapes: prizes (5, 100) -> (100,), weights (5, 100)
    if prizes.ndim == 2:
        prizes = prizes.sum(axis=0)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Time budget guard
    start_time = time.time()
    time_limit = 290.0  # Safety margin before 300s limit
    
    # --- Part 1: Lagrangian Dual Approximation ---
    lambda_c = np.zeros(n_constraints)
    n_iters_lagrangian = 30
    step0 = 1.0
    best_rc = np.zeros(n_items)
    best_value = -1.0
    
    for it in range(n_iters_lagrangian):
        if time.time() - start_time > time_limit:
            break
            
        shadow_cost = lambda_c @ weights  # shape (n_items,)
        rc = prizes - shadow_cost  # shape (n_items,)
        
        order = np.argsort(-rc)
        rem_cap = np.full(n_constraints, capacity)
        selected = np.zeros(n_items, dtype=bool)
        total_prize = 0.0
        
        for i in order:
            if rc[i] <= 0:
                break
            if np.all(weights[:, i] <= rem_cap):
                selected[i] = True
                rem_cap -= weights[:, i]
                total_prize += prizes[i]
        
        if total_prize > best_value:
            best_value = total_prize
            best_rc = rc.copy()
        
        used_cap = weights @ selected
        slack = capacity - used_cap
        step = step0 / (1.0 + it * 0.1)
        lambda_c = np.maximum(0, lambda_c + step * slack)
    
    # Normalize reduced costs to [0, 1]
    rc_max = best_rc.max()
    if rc_max > 0:
        rc_norm = best_rc / rc_max
    else:
        rc_norm = np.ones(n_items)
    rc_norm = np.maximum(rc_norm, 0.0)
    
    # --- Part 2: Multi-Start Greedy Simulation ---
    weighted_counts = np.zeros(n_items, dtype=float)
    
    # Precompute weight metrics
    sum_w = weights.sum(axis=0)
    max_w = weights.max(axis=0)
    min_w = weights.min(axis=0)
    avg_w = weights.mean(axis=0)
    
    # Compute bottleneck slack
    safe_weights = np.maximum(weights, 1e-9)
    slack_ratios = (capacity - weights) / safe_weights
    slack_ratios = np.maximum(slack_ratios, 0.0)
    bottleneck_slack = slack_ratios.min(axis=0)
    
    # Define strategies
    strategies = []
    
    # Strategy 1: Prize / Sum Weight
    ratio1 = prizes / (sum_w + 1e-9)
    strategies.append(np.argsort(-ratio1))
    
    # Strategy 2: Prize / Max Weight
    ratio2 = prizes / (max_w + 1e-9)
    strategies.append(np.argsort(-ratio2))
    
    # Strategy 3: Prize only
    strategies.append(np.argsort(-prizes))
    
    # Strategy 4: Prize / (Max Weight)^2
    ratio4 = prizes / (max_w ** 2 + 1e-9)
    strategies.append(np.argsort(-ratio4))
    
    # Strategy 5: Prize / Min Weight
    ratio5 = prizes / (min_w + 1e-9)
    strategies.append(np.argsort(-ratio5))
    
    # Strategy 6: Prize / Avg Weight
    ratio6 = prizes / (avg_w + 1e-9)
    strategies.append(np.argsort(-ratio6))
    
    # Strategy 7: Bottleneck Slack * Prize
    slack_prize = bottleneck_slack * prizes
    strategies.append(np.argsort(-slack_prize))
    
    # Strategy 8: Random (seeded)
    rng = np.random.RandomState(42)
    strategies.append(np.array(rng.permutation(n_items)))
    
    # Strategy 9: Random (seeded)
    rng2 = np.random.RandomState(123)
    strategies.append(np.array(rng2.permutation(n_items)))
    
    # Strategy 10: Combined (Slack + 1) * Prize / (Max W + 1)
    combo10 = (bottleneck_slack + 1.0) * prizes / (max_w + 1.0)
    strategies.append(np.argsort(-combo10))
    
    # Strategy 11: Reduced cost from Lagrangian
    strategies.append(np.argsort(-rc_norm))
    
    # Strategy 12: Prize * (1 + bottleneck_slack) / (max_w + 1)
    combo12 = prizes * (1.0 + bottleneck_slack) / (max_w + 1.0)
    strategies.append(np.argsort(-combo12))
    
    # Run greedy simulations
    for order in strategies:
        if time.time() - start_time > time_limit:
            break
            
        remaining_capacity = np.full(n_constraints, capacity, dtype=float)
        selected = []
        total_prize = 0.0
        
        for idx in order:
            if np.all(weights[:, idx] <= remaining_capacity):
                remaining_capacity -= weights[:, idx]
                selected.append(idx)
                total_prize += prizes[idx]
        
        if total_prize > 0:
            for idx in selected:
                weighted_counts[idx] += total_prize
    
    # Normalize weighted counts
    wc_max = weighted_counts.max()
    if wc_max > 0:
        wc_norm = weighted_counts / wc_max
    else:
        wc_norm = np.ones(n_items)
    
    # Normalize bottleneck slack
    bs_max = bottleneck_slack.max()
    if bs_max > 0:
        bs_norm = bottleneck_slack / bs_max
    else:
        bs_norm = np.ones(n_items)
    
    # Normalize prizes
    p_max = prizes.max()
    if p_max > 0:
        p_norm = prizes / p_max
    else:
        p_norm = np.ones(n_items)
    
    # --- Part 3: Combine Signals ---
    # We want to emphasize items that are:
    # 1. High reduced cost (Lagrangian profitable)
    # 2. Frequently selected in high-value greedy packs
    # 3. Have high bottleneck slack (fit easily)
    # 4. Have high prize
    
    # Use a product form with exponents to tune sensitivity
    # rc_norm^1.0 * wc_norm^1.5 * bs_norm^1.0 * p_norm^1.0
    scores = (rc_norm ** 1.0) * (wc_norm ** 1.5) * (bs_norm ** 1.0) * (p_norm ** 1.0)
    
    # Add a small base to avoid zeros
    scores = scores + 1e-9
    
    # Normalize to [0, 1]
    s_max = scores.max()
    if s_max > 0:
        scores = scores / s_max
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
