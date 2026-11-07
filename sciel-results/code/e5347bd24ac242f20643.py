import numpy as np
from functools import lru_cache

POLICY_CACHE = {}

def _compute_policy(params):
    """
    Computes the optimal bid and ask depths for all time steps and inventory levels
    using a discrete-action Dynamic Programming approach.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    N_q = 2 * Q + 1
    q_min = -Q
    q_max = Q
    
    # Define a log-spaced grid of candidate depths
    # Min depth small enough to have high fill prob, max depth large enough to have low prob
    d_min = 1e-4
    d_max = 5.0 / k
    N_d = 50
    depths = np.logspace(np.log10(d_min), np.log10(d_max), N_d)
    
    # Pre-compute probabilities for all candidate depths
    # p = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = np.exp(-k * depths)
    fill_probs = 1.0 - np.exp(-A * exp_term * dt)
    
    # V_next[q_idx] is the value at time n+1 for inventory q
    # q_idx = q - q_min
    V_next = np.zeros(N_q)
    
    # Policies to store: shape (steps, N_q)
    bid_policy = np.zeros((steps, N_q))
    ask_policy = np.zeros((steps, N_q))
    
    # Iterate backwards in time
    for n in range(steps - 1, -1, -1):
        # Risk cost for each inventory level at step n
        q_values = np.arange(q_min, q_max + 1)
        risk = 0.5 * gamma * sigma**2 * q_values**2 * dt
        
        # We need V_next for q-1, q, q+1
        # V_next is indexed by q - q_min
        # q-1 index: (q-1) - q_min = (q - q_min) - 1
        # q+1 index: (q+1) - q_min = (q - q_min) + 1
        
        # Pad V_next to handle boundaries
        # q = -Q (idx 0): no bid fill (q-1 is -Q-1, invalid). 
        # Model says: "no bid fill at q = Q" (wait, "no bid fill at q = Q"? 
        # "no bid fill at q = Q, no ask fill at q = -Q"
        # Wait, if q=Q (max inventory), we are fully long. 
        # Bid fill increases inventory? 
        # "A fill earns its depth and moves inventory one unit"
        # Usually:
        # Bid hit: Buy. Inventory increases? Or decreases?
        # "Inventory" usually means long position.
        # If I bid, I am offering to buy. If hit, I buy. My long position increases.
        # So Bid hit -> q -> q+1.
        # Ask hit: Sell. Inventory decreases. q -> q-1.
        # Let's check the constraint: "no bid fill at q = Q".
        # If q=Q, we can't increase further. So no bid fill.
        # "no ask fill at q = -Q".
        # If q=-Q, we can't decrease further (short limit). So no ask fill.
        
        # So:
        # Bid hit: q -> q+1. Reward: -b. (Buy at Mid-b, cost b relative to mid).
        # Ask hit: q -> q-1. Reward: +a. (Sell at Mid+a, gain a relative to mid).
        
        # Vectorized computation over q and depths
        # V_next shape (N_q,)
        # We need V_next at q+1, q, q-1 for each q.
        
        V_next_q = V_next  # V(q)
        V_next_q_plus_1 = np.roll(V_next, -1)
        V_next_q_minus_1 = np.roll(V_next, 1)
        
        # Fix boundaries
        # For q = Q (last index), bid fill (q->Q+1) is forbidden. 
        # So term involving V_next_q_plus_1 at last index should be 0? 
        # Or rather, the probability of that transition is 0.
        # We will set the probability of the forbidden transition to 0.
        
        # Probabilities are the same for all q, but the *validity* of the transition depends on q.
        # p_bid is prob of bid fill. p_ask is prob of ask fill.
        
        # Create meshes for q and depths
        # q_idx: 0..N_q-1
        # d_idx: 0..N_d-1
        
        # To vectorize over q and d simultaneously:
        # V_next_q: (N_q, 1, 1)
        # V_next_q_plus_1: (N_q, 1, 1)
        # V_next_q_minus_1: (N_q, 1, 1)
        
        # Fill probs: (N_d,)
        # We need p_b for each b, p_a for each a.
        # p_b: (1, N_d, 1)
        # p_a: (1, 1, N_d)
        
        V_q = V_next_q[:, np.newaxis, np.newaxis]
        V_qp1 = V_next_q_plus_1[:, np.newaxis, np.newaxis]
        V_qm1 = V_next_q_minus_1[:, np.newaxis, np.newaxis]
        
        p_b = fill_probs[np.newaxis, :, np.newaxis]
        p_a = fill_probs[np.newaxis, np.newaxis, :]
        
        # Probabilities of transitions:
        # 1. No fill: (1-p_b)(1-p_a) * V_q
        # 2. Bid only: p_b(1-p_a) * V_qp1
        # 3. Ask only: (1-p_b)p_a * V_qm1
        # 4. Both: p_b p_a * V_q (Net change 0)
        
        # Immediate Rewards:
        # Bid reward: -b (if bid hits)
        # Ask reward: +a (if ask hits)
        # Total Immediate: p_b*(-b) + p_a*(a)
        
        # b: (1, N_d, 1)
        # a: (1, 1, N_d)
        b_vals = depths[np.newaxis, :, np.newaxis]
        a_vals = depths[np.newaxis, np.newaxis, :]
        
        # Expected Next Value
        E_V_next = (1-p_b)*(1-p_a)*V_q + \
                   p_b*(1-p_a)*V_qp1 + \
                   (1-p_b)*p_a*V_qm1 + \
                   p_b*p_a*V_q
        
        # Immediate Reward
        Reward = p_b * (-b_vals) + p_a * (a_vals)
        
        # Total Value Candidate
        V_cand = -risk[:, np.newaxis, np.newaxis] + Reward + E_V_next
        
        # Handle Boundaries:
        # At q = Q (last index, idx N_q-1): Bid fill is forbidden.
        # So we must set the contribution of Bid fill transitions to 0 for that q.
        # This means we should only consider actions where p_b is effectively 0?
        # No, the model says "no bid fill". It doesn't say we can't quote bid.
        # But if we quote bid and it's forbidden, does it fill? No.
        # So the probability of bid fill is 0 at q=Q.
        # Similarly, prob of ask fill is 0 at q=-Q (idx 0).
        
        # We can mask the probabilities based on q.
        # Create a mask for valid bid fills: all True, except last q.
        # valid_bid: (N_q, 1, 1)
        valid_bid = np.ones((N_q, 1, 1))
        valid_bid[-1, :, :] = 0.0
        
        # valid_ask: (N_q, 1, 1)
        valid_ask = np.ones((N_q, 1, 1))
        valid_ask[0, :, :] = 0.0
        
        # Apply masks to p_b and p_a
        p_b_masked = p_b * valid_bid
        p_a_masked = p_a * valid_ask
        
        # Recalculate with masked probs
        # If p_b is 0, then bid fill doesn't happen.
        # The reward term p_b * (-b) becomes 0.
        # The E_V_next terms involving p_b become 0.
        
        E_V_next_masked = (1-p_b_masked)*(1-p_a_masked)*V_q + \
                          p_b_masked*(1-p_a_masked)*V_qp1 + \
                          (1-p_b_masked)*p_a_masked*V_qm1 + \
                          p_b_masked*p_a_masked*V_q
        
        Reward_masked = p_b_masked * (-b_vals) + p_a_masked * (a_vals)
        
        V_cand_masked = -risk[:, np.newaxis, np.newaxis] + Reward_masked + E_V_next_masked
        
        # Find optimal b and a for each q
        # V_cand_masked shape: (N_q, N_d, N_d)
        max_vals, max_indices = np.max(V_cand_masked, axis=(1, 2))
        
        # max_indices are flat indices into (N_d, N_d)
        # We need to unpack them to b_idx and a_idx
        b_indices = max_indices // N_d
        a_indices = max_indices % N_d
        
        # Store optimal depths
        bid_policy[n, :] = depths[b_indices]
        ask_policy[n, :] = depths[a_indices]
        
        # Update V_next for next iteration
        V_next = max_vals
        
    return bid_policy, ask_policy

def quote(t, inventory, mid, params):
    """
    Returns bid and ask depths for each path.
    """
    # Hash params for caching
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    
    if key not in POLICY_CACHE:
        POLICY_CACHE[key] = _compute_policy(params)
    
    bid_policy, ask_policy = POLICY_CACHE[key]
    
    steps = params['steps']
    Q = params['Q']
    dt = params['T'] / steps
    
    # Determine current step index
    # t is n * dt
    n = int(round(t / dt))
    # Clamp n to valid range
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    # Map inventory to index: q in [-Q, Q] -> idx in [0, 2Q]
    # idx = q + Q
    idx = inventory + Q
    
    # Get depths
    bid_depths = bid_policy[n, idx]
    ask_depths = ask_policy[n, idx]
    
    return bid_depths, ask_depths

