"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 290
_POLICY_CACHE = {}


def _compute_policy(params):
    """
    Compute the optimal policy via backward induction (DP) over the
    discrete (time, inventory) state space.
    
    Returns (opt_bid, opt_ask) arrays of shape (steps, 2*Q+1) or None on timeout.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Depth grid: we search over a finite set of depths
    # The fill probability is 1 - exp(-A * exp(-k*depth) * dt)
    # For very small depths, fill prob ~ 1 (too aggressive)
    # For very large depths, fill prob ~ 0 (too passive)
    # We want to capture the tradeoff. Let's use a log-spaced or linear grid.
    
    # The optimal depth should be around 1/k to a few units.
    # Let's use a grid from 0.01 to maybe 5/k, with enough resolution.
    # Actually, let's think about the scale. k is 0.75-1.5, so 1/k is 0.67-1.33.
    # Depths that matter are probably in [0.01, 3.0].
    
    # Number of depth levels to search over
    N_depths = 80
    depth_min = 0.005
    depth_max = 3.0
    depth_grid = np.linspace(depth_min, depth_max, N_depths)
    
    # Precompute fill probabilities for each depth
    # p_fill(d) = 1 - exp(-A * exp(-k * d) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Value function: V[n, q] = optimal expected reward from step n with inventory q
    # q ranges from -Q to Q, so 2Q+1 values
    num_q = 2 * Q + 1
    
    # V[n, q] for n = 0..steps, q = -Q..Q
    # We store V as shape (steps+1, num_q)
    V = np.zeros((steps + 1, num_q), dtype=np.float64)
    
    # opt_bid[n, q] and opt_ask[n, q] store the optimal depths (indices into depth_grid)
    opt_bid_idx = np.zeros((steps, num_q), dtype=np.int32)
    opt_ask_idx = np.zeros((steps, num_q), dtype=np.int32)
    
    # Backward induction from n = steps-1 down to n = 0
    for n in range(steps - 1, -1, -1):
        remaining = steps - n
        time_left = remaining * dt
        
        for qi in range(num_q):
            q = qi - Q  # actual inventory value
            
            # Penalty for current inventory
            penalty = (gamma / 2.0) * sigma * sigma * q * q * dt
            
            # Try all combinations of bid and ask depths
            # But we can decouple: for each bid depth, find best ask depth, and vice versa
            # Actually, the fill events are independent, so we can think of it as:
            # Expected reward = -penalty + E[reward from fills + V[next_state]]
            
            # The mid moves by sigma*sqrt(dt)*N(0,1), but this doesn't affect the
            # fill probabilities or the inventory transition, only the absolute price.
            # Since the objective is in terms of depths earned (not absolute prices),
            # and the penalty depends only on inventory, the mid movement doesn't
            # affect the optimal policy! The depth is relative to mid, and the
            # reward from a fill is the depth itself.
            
            # So the problem is purely about inventory management.
            
            # For a given bid_depth and ask_depth:
            # p_bid = fill_probs[bid_idx]
            # p_ask = fill_probs[ask_idx]
            # 
            # Four cases:
            # 1. No fill: prob (1-p_bid)*(1-p_ask), next_q = q
            # 2. Bid fill only: prob p_bid*(1-p_ask), next_q = q-1 (if q > -Q), else no fill
            # 3. Ask fill only: prob (1-p_bid)*p_ask, next_q = q+1 (if q < Q), else no fill
            # 4. Both fill: prob p_bid*p_ask, next_q = q (bid and ask cancel)
            
            # Wait, let me re-read: "A fill earns its depth and moves inventory one unit"
            # Bid fill: you sell at bid, so inventory decreases by 1 (if you're long)
            # Actually, in market making:
            # - Bid fill: you buy at bid price, inventory increases by 1
            # - Ask fill: you sell at ask price, inventory decreases by 1
            
            # Let me re-check: "bid hit with probability... A fill earns its depth and moves inventory one unit"
            # "no bid fill at q = Q" means if inventory is at max, can't buy more
            # "no ask fill at q = -Q" means if inventory is at min, can't sell more
            
            # So:
            # Bid fill: q -> q+1 (buy), reward = bid_depth
            # Ask fill: q -> q-1 (sell), reward = ask_depth
            
            # But wait, "earns its depth" - this is the profit from the trade.
            # If you buy at bid (which is mid - bid_depth), and the mid is the reference,
            # then you "earn" bid_depth relative to mid. Similarly for ask.
            
            # Actually, I think the reward is simply the depth, representing the
            # spread captured. Let me just use:
            # Bid fill: reward += bid_depth, q -> q+1
            # Ask fill: reward += ask_depth, q -> q-1
            
            best_val = -np.inf
            best_bid_idx = 0
            best_ask_idx = 0
            
            for bi in range(N_depths):
                p_bid = fill_probs[bi]
                bid_d = depth_grid[bi]
                
                # If q == Q, no bid fill possible
                if q >= Q:
                    p_bid = 0.0
                
                for ai in range(N_depths):
                    p_ask = fill_probs[ai]
                    ask_d = depth_grid[ai]
                    
                    # If q == -Q, no ask fill possible
                    if q <= -Q:
                        p_ask = 0.0
                    
                    # Compute expected next value
                    # Case 1: no fill
                    p_none = (1.0 - p_bid) * (1.0 - p_ask)
                    v_none = V[n + 1, qi]
                    
                    # Case 2: bid fill only
                    if q < Q:
                        p_bid_only = p_bid * (1.0 - p_ask)
                        v_bid_only = bid_d + V[n + 1, qi + 1]
                    else:
                        p_bid_only = 0.0
                        v_bid_only = 0.0
                    
                    # Case 3: ask fill only
                    if q > -Q:
                        p_ask_only = (1.0 - p_bid) * p_ask
                        v_ask_only = ask_d + V[n + 1, qi - 1]
                    else:
                        p_ask_only = 0.0
                        v_ask_only = 0.0
                    
                    # Case 4: both fill
                    p_both = p_bid * p_ask
                    # Both fill: bid increases q by 1, ask decreases q by 1, net 0
                    # Reward: bid_d + ask_d
                    v_both = bid_d + ask_d + V[n + 1, qi]
                    
                    expected_next = (p_none * v_none +
                                     p_bid_only * v_bid_only +
                                     p_ask_only * v_ask_only +
                                     p_both * v_both)
                    
                    total_val = -penalty + expected_next
                    
                    if total_val > best_val:
                        best_val = total_val
                        best_bid_idx = bi
                        best_ask_idx = ai
            
            V[n, qi] = best_val
            opt_bid_idx[n, qi] = best_bid_idx
            opt_ask_idx[n, qi] = best_ask_idx
        
        # Check deadline
        if time.time() > _DEADLINE:
            return None
    
    # Convert indices to actual depths
    opt_bid = depth_grid[opt_bid_idx]  # shape (steps, num_q)
    opt_ask = depth_grid[opt_ask_idx]  # shape (steps, num_q)
    
    return opt_bid, opt_ask


def _get_policy(params):
    """Get or compute the optimal policy for given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy(params)
    
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Quote using precomputed optimal policy via DP.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']
    T = params['T']
    
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    q = inventory.astype(np.int32)
    
    policy = _get_policy(params)
    
    if policy is not None:
        opt_bid, opt_ask = policy
        # q + Q maps [-Q, Q] to [0, 2Q]
        bid_depths = opt_bid[n, q + Q].copy()
        ask_depths = opt_ask[n, q + Q].copy()
    else:
        # Fallback heuristic if DP timed out
        base_depth = 1.0 / k
        time_fraction = (T - t) / T
        beta = 0.02 * (1.0 + 0.5 * sigma) * (0.5 + 0.5 * time_fraction)
        q_scaled = q.astype(np.float64) / Q
        q_adjust = np.tanh(1.5 * q_scaled)
        bid_depths = base_depth + beta * q_adjust
        ask_depths = base_depth - beta * q_adjust
        bid_depths = np.maximum(bid_depths, 0.01)
        ask_depths = np.maximum(ask_depths, 0.01)
    
    # Boundary: no bid at q=Q, no ask at q=-Q
    bid_depths[q >= Q] = np.inf
    ask_depths[q <= -Q] = np.inf
    
    # Sanitize: no NaN, no values below -3/k
    min_allowed = -3.0 / k
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    bid_depths = np.where(bid_depths < min_allowed, np.inf, bid_depths)
    ask_depths = np.where(ask_depths < min_allowed, np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
