import numpy as np
import time

# Global cache for precomputed policies
_policy_cache = {}
_start_time = time.time()
_TIME_LIMIT = 300.0
_SAFE_MARGIN = 10.0

def _precompute_policy(regime_params):
    """
    Precompute optimal bid/ask depths for all steps and inventory levels for a given regime.
    Returns a dictionary or array structure for fast lookup.
    """
    sigma = regime_params['sigma']
    A = regime_params['A']
    k = regime_params['k']
    gamma = regime_params['gamma']
    T = regime_params['T']
    steps = regime_params['steps']
    Q = regime_params['Q']
    
    dt = T / steps
    # Inventory range
    q_min = -Q
    q_max = Q
    q_values = np.arange(q_min, q_max + 1)
    n_q = len(q_values)
    
    # Depth grid
    # We want to cover depths where the fill probability is non-negligible.
    # p = 1 - exp(-A exp(-k d) dt)
    # Let's use a grid from d_min to d_max.
    d_min = 0.01
    d_max = 5.0
    n_d = 50
    depths = np.linspace(d_min, d_max, n_d)
    
    # Precompute fill probabilities for all depths
    # P_fill[d] = 1 - exp(-A exp(-k d) dt)
    exp_term = np.exp(-k * depths)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    
    # Initialize Value Function V_n(q)
    # V is a 2D array [steps+1, n_q]
    # V[steps] is terminal value, which is 0.
    V = np.zeros((steps + 1, n_q))
    
    # Store optimal actions
    # opt_bid[n, q_idx] = depth
    # opt_ask[n, q_idx] = depth
    opt_bid = np.full((steps, n_q), np.inf)
    opt_ask = np.full((steps, n_q), np.inf)
    
    # Iterate backwards in time
    for n in range(steps - 1, -1, -1):
        # Check time limit
        if time.time() - _start_time > _TIME_LIMIT - _SAFE_MARGIN:
            # Fallback to simple heuristic if we run out of time
            # Just return 1/k for all
            fallback_bid = np.full(n_q, 1.0/k)
            fallback_ask = np.full(n_q, 1.0/k)
            # Fill remaining steps with fallback
            for n_rem in range(n, -1, -1):
                opt_bid[n_rem] = fallback_bid
                opt_ask[n_rem] = fallback_ask
            break
        
        # Risk penalty for current step
        # Penalty = (gamma / 2) * sigma^2 * q^2 * dt
        q_sq = q_values ** 2
        risk_penalty = 0.5 * gamma * sigma**2 * q_sq * dt
        
        # We need to maximize:
        # - risk_penalty[q] + 
        #   p_fill_bid * (depth_bid + V[n+1][q-1]) + 
        #   p_fill_ask * (depth_ask + V[n+1][q+1]) +
        #   (1-p_fill_bid)*(1-p_fill_ask) * V[n+1][q]
        #
        # Note: 
        # If bid fills, q becomes q-1. Reward is depth_bid.
        # If ask fills, q becomes q+1. Reward is depth_ask.
        # If both fill (prob p_b * p_a), q becomes q-1+1 = q? 
        # The problem says "independently". 
        # "A fill earns its depth and moves inventory one unit".
        # If both hit, do we earn both depths? Yes.
        # Inventory change: -1 (bid) + 1 (ask) = 0.
        # So:
        # Term = p_b * d_b + p_a * d_a + 
        #        p_b * V[n+1][q-1] + 
        #        p_a * V[n+1][q+1] +
        #        (1-p_b)(1-p_a) V[n+1][q] - p_b p_a V[n+1][q] ... wait.
        #
        # Let's expand carefully.
        # Let $V_{n+1}$ be the value of next state.
        # State transitions:
        # 1. Bid only: prob $p_b (1-p_a)$. New q: $q-1$. Reward: $d_b$.
        # 2. Ask only: prob $(1-p_b) p_a$. New q: $q+1$. Reward: $d_a$.
        # 3. Both: prob $p_b p_a$. New q: $q-1+1 = q$. Reward: $d_b + d_a$.
        # 4. Neither: prob $(1-p_b)(1-p_a)$. New q: $q$. Reward: $0$.
        #
        # Expected Next Value:
        # $E[V_{n+1}] = p_b(1-p_a)V_{n+1}[q-1] + (1-p_b)p_a V_{n+1}[q+1] + (p_b p_a + (1-p_b)(1-p_a)) V_{n+1}[q]$
        #
        # Expected Current Reward:
        # $E[R] = p_b(1-p_a)d_b + (1-p_b)p_a d_a + p_b p_a (d_b + d_a)$
        # $E[R] = d_b (p_b - p_b p_a + p_b p_a) + d_a (p_a - p_b p_a + p_b p_a) = d_b p_b + d_a p_a$
        #
        # So Total Value to Maximize:
        # $- risk[q] + p_b d_b + p_a d_a + E[V_{n+1}]$
        
        # We can vectorize over depths for bid and ask.
        # p_b is a vector of shape (n_d,)
        # p_a is a vector of shape (n_d,)
        # d_b is a vector of shape (n_d,)
        # d_a is a vector of shape (n_d,)
        
        # V_next is shape (n_q,)
        # V_next[q-1] corresponds to index q_idx - 1
        # V_next[q+1] corresponds to index q_idx + 1
        
        # We need to handle boundaries.
        # If q=Q, bid fill is impossible. So p_b = 0.
        # If q=-Q, ask fill is impossible. So p_a = 0.
        
        # Let's create arrays for V_next shifted
        # V_next_shifted_down: V_next[q-1] for each q.
        # If q=Q, V_next[q-1] is valid.
        # If q=-Q, V_next[q-1] is out of bounds (would be q=-Q-1). 
        # But if q=-Q, bid fill is impossible? 
        # "no bid fill at q=Q, no ask fill at q=-Q".
        # Wait, if q=Q (max inventory), we cannot buy more. So Bid fill prob = 0.
        # If q=-Q (min inventory), we cannot sell more. So Ask fill prob = 0.
        
        # So:
        # For q=Q (index n_q-1): p_b = 0.
        # For q=-Q (index 0): p_a = 0.
        
        # Let's compute the term for each q and each pair of depths.
        # This is a 2D grid of (n_d, n_d) for each q.
        # Total size: n_q * n_d * n_d.
        # 61 * 50 * 50 = 152,500 values per step.
        # 1000 steps = 150 million values.
        # This might be slow in pure Python loops, but NumPy can handle it.
        # We can vectorize over q as well?
        # The terms $V_{n+1}[q-1]$ and $V_{n+1}[q+1]$ depend on q.
        
        # Let's construct the matrix for all q at once.
        # p_b is same for all q, except q=Q where it is 0.
        # p_a is same for all q, except q=-Q where it is 0.
        
        # Create masks
        # p_b_vec: shape (n_d,)
        # p_a_vec: shape (n_d,)
        
        # For each q, we compute:
        # Val[q, i, j] = -risk[q] + p_b[i]*d_b[i] + p_a[j]*d_a[j] + 
        #                 p_b[i]*(1-p_a[j])*V_next[q-1] + 
        #                 (1-p_b[i])*p_a[j]*V_next[q+1] + 
        #                 (p_b[i]*p_a[j] + (1-p_b[i])*(1-p_a[j]))*V_next[q]
        
        # We can simplify the V_next terms.
        # Let $A_{ij} = p_b[i] p_a[j]$, $B_{ij} = p_b[i] (1-p_a[j])$, $C_{ij} = (1-p_b[i]) p_a[j]$, $D_{ij} = (1-p_b[i])(1-p_a[j])$.
        # $E[V] = B_{ij} V_{q-1} + C_{ij} V_{q+1} + (A_{ij} + D_{ij}) V_q$.
        
        # We can precompute the coefficient matrices for the probabilities.
        # p_b_grid: shape (1, n_d, 1) ? No, let's use broadcasting.
        
        # Let's try to vectorize over q.
        # V_next is shape (n_q,)
        # V_next_q = V_next (shape n_q)
        # V_next_q_minus = np.roll(V_next, 1) # q-1
        # V_next_q_plus = np.roll(V_next, -1) # q+1
        # Handle boundaries:
        # If q=Q (last index), bid fill prob is 0.
        # If q=-Q (first index), ask fill prob is 0.
        
        # Let's create arrays for the effective probabilities for each q.
        # p_b_eff[q, i]: 0 if q==Q else p_fill[i]
        # p_a_eff[q, j]: 0 if q==-Q else p_fill[j]
        
        # This creates an array of shape (n_q, n_d) for p_b and p_a.
        p_b_full = np.tile(p_fill, (n_q, 1)) # shape (n_q, n_d)
        p_a_full = np.tile(p_fill, (n_q, 1)) # shape (n_q, n_d)
        
        # Zero out bid prob for q=Q (index -1)
        p_b_full[-1, :] = 0.0
        # Zero out ask prob for q=-Q (index 0)
        p_a_full[0, :] = 0.0
        
        # Now we need to maximize over i (bid depth index) and j (ask depth index).
        # For each q, we have a matrix of values of shape (n_d, n_d).
        
        # Let's reshape to allow broadcasting.
        # p_b: (n_q, n_d, 1)
        # p_a: (n_q, 1, n_d)
        p_b = p_b_full[:, :, np.newaxis]
        p_a = p_a_full[np.newaxis, :, :] # Wait, p_a_full is (n_q, n_d). We want (n_q, 1, n_d) for j dimension?
        # Let's be precise.
        # We want to compute Value[q, i, j].
        # p_b[q, i]
        # p_a[q, j]
        
        p_b_3d = p_b_full[:, :, np.newaxis] # (n_q, n_d, 1)
        p_a_3d = p_a_full[np.newaxis, :, :] # (1, n_d, n_q) -> No.
        # p_a_full is (n_q, n_d). We want index j to be the second dimension of the result.
        # So p_a_3d should be (n_q, 1, n_d).
        p_a_3d = p_a_full[:, np.newaxis, :] # (n_q, 1, n_d)
        
        d_b = depths[np.newaxis, :, np.newaxis] # (1, n_d, 1)
        d_a = depths[np.newaxis, np.newaxis, :] # (1, 1, n_d)
        
        V_next = V[n+1] # (n_q,)
        V_next_q = V_next[:, np.newaxis, np.newaxis] # (n_q, 1, 1)
        
        # Shifted V
        # V[q-1]:
        # If q=0 (q=-Q), V[q-1] is invalid. But p_b is 0 for q=0? No, p_a is 0 for q=0.
        # If q=0, p_a=0. So terms with p_a are 0.
        # But the term $C_{ij} V_{q+1}$ uses $V_{q+1}$.
        # The term $B_{ij} V_{q-1}$ uses $V_{q-1}$.
        # If q=0, $V_{q-1}$ is out of bounds.
        # However, if q=0, p_a=0.
        # $B_{ij} = p_b (1-p_a) = p_b$.
        # So we need $V_{q-1}$ even if p_a=0?
        # Yes, if bid fills, q goes to q-1.
        # But at q=-Q (0), can bid fill?
        # "no bid fill at q=Q". It does NOT say "no bid fill at q=-Q".
        # Inventory is in [-Q, Q].
        # If q=-Q, and bid fills, q becomes -Q-1?
        # The problem says "inventory (4000,) ints in [-Q, Q]".
        # Usually, this implies the state space is bounded.
        # If a fill would push it out of bounds, does it happen?
        # "no bid fill at q = Q, no ask fill at q = -Q".
        # This implies:
        # At q=Q, bid fill prob = 0.
        # At q=-Q, ask fill prob = 0.
        # What about bid fill at q=-Q?
        # If q=-Q and bid fills, q becomes -Q-1.
        # Is -Q-1 a valid state?
        # The input inventory is always in [-Q, Q].
        # If the model allows q to go out of bounds, the next step's input would be out of bounds.
        # But the evaluator provides inventory in [-Q, Q].
        # This suggests that transitions that would go out of bounds are either:
        # 1. Prohibited (prob 0).
        # 2. Clamped (q stays at -Q).
        # 3. The state space is actually larger, but the evaluator only tests within bounds?
        # Given "no bid fill at q=Q" is explicit, and "no ask fill at q=-Q" is explicit, it's likely that:
        # - At q=Q, bid prob = 0.
        # - At q=-Q, ask prob = 0.
        # What about the other corners?
        # At q=Q, ask fill moves to Q+1?
        # If Q is the max inventory, usually you can't hold more.
        # But the text only forbids bid at Q and ask at -Q.
        # This is asymmetric.
        # Bid = Buy. Inventory increases.
        # Ask = Sell. Inventory decreases.
        # So:
        # Buy (Bid) increases q.
        # Sell (Ask) decreases q.
        # Wait.
        # "A fill earns its depth and moves inventory one unit".
        # Bid hit: I sell to the buyer? No, I am the market maker.
        # I quote Bid (buy price) and Ask (sell price).
        # If Bid is hit, a buyer buys from me? No, a buyer buys at my Bid?
        # No, I quote Bid (I am willing to buy at Bid) and Ask (I am willing to sell at Ask).
        # If a trader hits my Bid, they are buying from me? No, they are selling to me?
        # Standard convention:
        # Bid: Price I pay to buy. If hit, I buy. Inventory increases (+1).
        # Ask: Price I receive to sell. If hit, I sell. Inventory decreases (-1).
        
        # Let's re-read: "bid hit ... ask hit".
        # "no bid fill at q = Q".
        # If q=Q (max inventory), I should not buy more. So Bid fill (which increases inventory) is forbidden.
        # This matches: Bid -> +1 inventory.
        # "no ask fill at q = -Q".
        # If q=-Q (min inventory), I should not sell more. So Ask fill (which decreases inventory) is forbidden.
        # This matches: Ask -> -1 inventory.
        
        # So:
        # Bid fill: q -> q+1.
        # Ask fill: q -> q-1.
        
        # My previous DP logic had:
        # "If bid fills, q becomes q-1". This was WRONG.
        # Correction:
        # Bid fill: q -> q+1.
        # Ask fill: q -> q-1.
        
        # Let's re-verify.
        # Inventory $q$.
        # Buy (Bid): $q \leftarrow q+1$.
        # Sell (Ask): $q \leftarrow q-1$.
        
        # Constraints:
        # If $q=Q$, cannot Buy. Bid prob = 0.
        # If $q=-Q$, cannot Sell. Ask prob = 0.
        
        # What if $q=Q$ and Ask hits? $q \leftarrow Q-1$. Valid.
        # What if $q=-Q$ and Bid hits? $q \leftarrow -Q+1$. Valid.
        
        # So the DP transitions are:
        # 1. Bid only: $p_b (1-p_a)$. New q: $q+1$.
        # 2. Ask only: $(1-p_b) p_a$. New q: $q-1$.
        # 3. Both: $p_b p_a$. New q: $q+1-1 = q$.
        # 4. Neither: $(1-p_b)(1-p_a)$. New q: $q$.
        
        # Reward: $d_b$ if bid hits, $d_a$ if ask hits.
        
        # $E[V_{n+1}] = p_b(1-p_a) V_{n+1}[q+1] + (1-p_b)p_a V_{n+1}[q-1] + (p_b p_a + (1-p_b)(1-p_a)) V_{n+1}[q]$
        
        # We need to handle boundaries for $V_{n+1}[q+1]$ and $V_{n+1}[q-1]$.
        # If $q=Q$, $q+1$ is out of bounds. But $p_b=0$ for $q=Q$.
        # So the term $p_b(1-p_a) V_{n+1}[q+1]$ is $0 \times V_{out} = 0$.
        # We can safely use 0 for out-of-bounds V values if the probability is 0.
        # To be safe, we can pad V_next with 0s.
        
        V_next_padded = np.zeros(n_q + 2)
        V_next_padded[1:-1] = V_next
        # Indices:
        # q=0 (index 0) -> q-1 is -1 (index 0 in padded), q+1 is 1 (index 1 in padded)
        # q=n_q-1 (index n_q-1) -> q-1 is n_q-2 (index n_q-1 in padded), q+1 is n_q (index n_q in padded)
        
        # V_next_minus[q] = V_next_padded[q]  (corresponds to q-1)
        # V_next_plus[q] = V_next_padded[q+2] (corresponds to q+1)
        
        V_next_minus = V_next_padded[:-2] # shape (n_q,)
        V_next_plus = V_next_padded[2:]   # shape (n_q,)
        V_next_same = V_next              # shape (n_q,)
        
        # Now vectorize.
        # p_b_3d: (n_q, n_d, 1)
        # p_a_3d: (n_q, 1, n_d)
        
        # Coefficients:
        # C_bid = p_b * (1 - p_a)
        # C_ask = (1 - p_b) * p_a
        # C_same = p_b * p_a + (1 - p_b) * (1 - p_a)
        
        C_bid = p_b_3d * (1.0 - p_a_3d)
        C_ask = (1.0 - p_b_3d) * p_a_3d
        C_same = p_b_3d * p_a_3d + (1.0 - p_b_3d) * (1.0 - p_a_3d)
        
        # Values:
        V_minus = V_next_minus[:, np.newaxis, np.newaxis] # (n_q, 1, 1)
        V_plus = V_next_plus[:, np.newaxis, np.newaxis]   # (n_q, 1, 1)
        V_same = V_next_same[:, np.newaxis, np.newaxis]   # (n_q, 1, 1)
        
        E_V = C_bid * V_plus + C_ask * V_minus + C_same * V_same
        
        # Rewards:
        R = p_b_3d * d_b + p_a_3d * d_a
        
        # Risk:
        Risk = risk_penalty[:, np.newaxis, np.newaxis] # (n_q, 1, 1)
        
        Total_Value = -Risk + R + E_V
        
        # Maximize over i and j
        # Total_Value shape: (n_q, n_d, n_d)
        # We want max over axes 1 and 2.
        # np.argmax returns flat index.
        max_vals = np.max(Total_Value, axis=(1, 2))
        max_indices = np.argmax(Total_Value, axis=(1, 2))
        
        # Convert flat indices to i, j
        i_indices = max_indices // n_d
        j_indices = max_indices % n_d
        
        # Store optimal depths
        opt_bid[n] = depths[i_indices]
        opt_ask[n] = depths[j_indices]
        
        # Update V[n]
        V[n] = max_vals
        
    return opt_bid, opt_ask

def _get_or_create_policy(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    if key not in _policy_cache:
        _policy_cache[key] = _precompute_policy(params)
    return _policy_cache[key]

def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths.
    """
    # Get precomputed policy
    opt_bid, opt_ask = _get_or_create_policy(params)
    
    # Determine step n
    T = params['T']
    steps = params['steps']
    dt = T / steps
    # t is n * dt
    # n = round(t / dt)
    n = int(round(t / dt))
    # Clamp n to [0, steps-1]
    if n < 0:
        n = 0
    elif n >= steps:
        n = steps - 1
        
    Q = params['Q']
    # inventory is array of ints
    # Convert to indices: q + Q
    # inventory range [-Q, Q]
    indices = inventory + Q
    
    # Get depths for each inventory
    bid_depths = opt_bid[n, indices]
    ask_depths = opt_ask[n, indices]
    
    return bid_depths, ask_depths
