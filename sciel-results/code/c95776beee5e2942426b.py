<<<<<<< SEARCH
"""
Optimal market making via pre-computed DP policy lookup.
The state space is (step, inventory) with 1000 steps and 61 inventory levels (-30..30).
We solve the Bellman equation backwards to find optimal bid/ask depths for each state.
Results are cached per regime so only the first call pays the DP cost.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 325.0  # safety margin before 330s kill


def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via backward DP.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) - optimal bid depth for each state
        ask_table: array of shape (steps, 2*Q+1) - optimal ask depth for each state
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    n_q = 2 * Q + 1  # 61 inventory levels
    q_range = np.arange(-Q, Q + 1)
    
    # We need to search over possible depths. The optimal depth is bounded.
    # From the seed, 1/k is the risk-neutral depth. Let's search in a reasonable range.
    # The probability of a fill is 1 - exp(-A * exp(-k * depth) * dt).
    # For very large depth, exp(-k*depth) -> 0, so fill prob -> 0.
    # For very small depth (near 0), fill prob is high.
    # Let's search over a grid of depths.
    
    # Reasonable depth range: from a small positive value to some upper bound
    # The key trade-off is between earning the depth and the inventory risk cost.
    # Let's use a grid of depths.
    
    # We'll use a fine grid for depth search
    # The optimal depth is likely around 1/k or slightly adjusted for inventory
    # Let's search from 0.01 to 5.0/k with fine resolution
    
    depth_min = 0.005
    depth_max = 5.0 / k
    n_depths = 200  # number of depth candidates to search
    
    depth_grid = np.linspace(depth_min, depth_max, n_depths)
    
    # Value function: V[step, q_index] = max expected future reward from step to end
    # at inventory level q
    # V[steps, q] = 0 (no terminal charge)
    
    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))
    
    # Backward DP
    for n in range(steps - 1, -1, -1):
        # Check time limit
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            # Not enough time, use a simple heuristic for remaining steps
            for qi in range(n_q):
                q = q_range[qi]
                # Simple inventory-adjusted depth
                adj = 1.0 / k - gamma * sigma**2 * q * dt / (2.0 * A * np.exp(-k / k) * dt)
                adj = max(adj, depth_min)
                bid_table[n, qi] = adj
                ask_table[n, qi] = adj
            # Fill remaining steps with same heuristic
            for n2 in range(n - 1, -1, -1):
                for qi in range(n_q):
                    q = q_range[qi]
                    adj = 1.0 / k - gamma * sigma**2 * q * dt / (2.0 * A * np.exp(-k / k) * dt)
                    adj = max(adj, depth_min)
                    bid_table[n2, qi] = adj
                    ask_table[n2, qi] = adj
            break
        
        for qi in range(n_q):
            q = q_range[qi]
            
            # Inventory cost for this step
            inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            best_val = -np.inf
            best_bid = depth_min
            best_ask = depth_min
            
            # For each candidate bid depth and ask depth, compute the expected value
            # To speed up, we can vectorize over depths
            # p_bid(d) = 1 - exp(-A * exp(-k*d) * dt)
            # p_ask(d) = 1 - exp(-A * exp(-k*d) * dt)
            
            # Compute fill probabilities for all candidate depths
            exp_term = np.exp(-k * depth_grid)
            p_fill = 1.0 - np.exp(-A * exp_term * dt)
            
            # For each combination of (bid_depth, ask_depth), compute expected next state
            # This is O(n_depths^2) per state which is 40000 * 61 * 1000 = too slow
            # We need a better approach.
            
            # Actually, let's think about this more carefully.
            # The expected value is:
            # E[V[n+1, q']] - inv_cost + bid_depth * p_bid + ask_depth * p_ask
            # where q' depends on which fills occur:
            # - bid fill: q+1 (if q < Q)
            # - ask fill: q-1 (if q > -Q)
            # - both: q (net zero)
            # - neither: q
            
            # p_both = p_bid * p_ask
            # p_bid_only = p_bid * (1 - p_ask)
            # p_ask_only = (1 - p_bid) * p_ask
            # p_neither = (1 - p_bid) * (1 - p_ask)
            
            # V_next = p_both * V[n+1, q] + p_bid_only * V[n+1, q+1] + p_ask_only * V[n+1, q-1] + p_neither * V[n+1, q]
            # (with boundary conditions: no bid fill at q=Q, no ask fill at q=-Q)
            
            # Let's vectorize over bid depths for a fixed ask depth, then optimize.
            # Actually, let's just do a 2D search but with fewer candidates.
            
            # Use coarser grid for speed
            n_search = 50
            search_depths = np.linspace(depth_min, depth_max, n_search)
            exp_s = np.exp(-k * search_depths)
            p_s = 1.0 - np.exp(-A * exp_s * dt)
            
            # For each bid depth candidate, compute the best ask depth
            # Vectorize: for each bid depth i, compute expected value for each ask depth j
            
            # V[n+1, q] is known
            Vn1_q = V[n + 1, qi]
            
            # Boundary conditions
            if q < Q:
                Vn1_q_plus = V[n + 1, qi + 1]  # q+1
            else:
                Vn1_q_plus = Vn1_q  # no bid fill possible, but we handle via p=0
            
            if q > -Q:
                Vn1_q_minus = V[n + 1, qi - 1]  # q-1
            else:
                Vn1_q_minus = Vn1_q  # no ask fill possible
            
            # p_bid[i], p_ask[j]
            # p_both[i,j] = p_bid[i] * p_ask[j]
            # p_bid_only[i,j] = p_bid[i] * (1 - p_ask[j])
            # p_ask_only[i,j] = (1 - p_bid[i]) * p_ask[j]
            # p_neither[i,j] = (1 - p_bid[i]) * (1 - p_ask[j])
            
            # V_next[i,j] = p_both * Vn1_q + p_bid_only * Vn1_q_plus + p_ask_only * Vn1_q_minus + p_neither * Vn1_q
            # = p_bid[i]*p_ask[j]*Vn1_q + p_bid[i]*(1-p_ask[j])*Vn1_q_plus + (1-p_bid[i])*p_ask[j]*Vn1_q_minus + (1-p_bid[i])*(1-p_ask[j])*Vn1_q
            
            # Let's simplify:
            # = p_bid[i] * (p_ask[j]*Vn1_q + (1-p_ask[j])*Vn1_q_plus) + (1-p_bid[i]) * (p_ask[j]*Vn1_q_minus + (1-p_ask[j])*Vn1_q)
            
            # For boundary cases, we need to adjust.
            # If q == Q, bid fills don't happen, so p_bid should be 0 effectively.
            # If q == -Q, ask fills don't happen, so p_ask should be 0 effectively.
            
            if q == Q:
                # No bid fills
                p_bid_eff = np.zeros(n_search)
            else:
                p_bid_eff = p_s.copy()
            
            if q == -Q:
                # No ask fills
                p_ask_eff = np.zeros(n_search)
            else:
                p_ask_eff = p_s.copy()
            
            # Compute V_next for all (i, j) pairs
            # V_next[i,j] = p_bid_eff[i] * (p_ask_eff[j]*Vn1_q + (1-p_ask_eff[j])*Vn1_q_plus) + (1-p_bid_eff[i]) * (p_ask_eff[j]*Vn1_q_minus + (1-p_ask_eff[j])*Vn1_q)
            
            # Let's define:
            # term1[j] = p_ask_eff[j]*Vn1_q + (1-p_ask_eff[j])*Vn1_q_plus
            # term2[j] = p_ask_eff[j]*Vn1_q_minus + (1-p_ask_eff[j])*Vn1_q
            # V_next[i,j] = p_bid_eff[i] * term1[j] + (1-p_bid_eff[i]) * term2[j]
            
            term1 = p_ask_eff * Vn1_q + (1 - p_ask_eff) * Vn1_q_plus
            term2 = p_ask_eff * Vn1_q_minus + (1 - p_ask_eff) * Vn1_q
            
            # V_next[i,j] = p_bid_eff[i] * term1[j] + (1-p_bid_eff[i]) * term2[j]
            # = p_bid_eff[i] * (term1[j] - term2[j]) + term2[j]
            
            # Reward[i,j] = search_depths[i] * p_bid_eff[i] + search_depths[j] * p_ask_eff[j]
            
            # Total[i,j] = V_next[i,j] + Reward[i,j] - inv_cost
            
            # We want to maximize over (i,j)
            
            # Vectorize:
            # diff = term1 - term2  (shape n_search)
            # V_next[i,j] = p_bid_eff[i] * diff[j] + term2[j]
            # Total[i,j] = p_bid_eff[i] * diff[j] + term2[j] + search_depths[i] * p_bid_eff[i] + search_depths[j] * p_ask_eff[j] - inv_cost
            # = p_bid_eff[i] * (diff[j] + search_depths[i]) + term2[j] + search_depths[j] * p_ask_eff[j] - inv_cost
            # = p_bid_eff[i] * (diff[j] + search_depths[i]) + (term2[j] + search_depths[j] * p_ask_eff[j]) - inv_cost
            
            diff = term1 - term2
            ask_term = term2 + search_depths * p_ask_eff
            
            # Total[i,j] = p_bid_eff[i] * (diff[j] + search_depths[i]) + ask_term[j] - inv_cost
            # For fixed i, this is linear in j through (diff[j] + ... ) but diff[j] depends on j.
            # Actually for fixed i: Total[i,j] = p_bid_eff[i] * diff[j] + p_bid_eff[i]*search_depths[i] + ask_term[j] - inv_cost
            # = (p_bid_eff[i] * diff[j] + ask_term[j]) + (p_bid_eff[i]*search_depths[i] - inv_cost)
            # The first part depends on j, the second is constant for fixed i.
            
            # So for each i, find j that maximizes (p_bid_eff[i] * diff[j] + ask_term[j])
            
            # Matrix form: M[i,j] = p_bid_eff[i] * diff[j] + ask_term[j]
            # M = np.outer(p_bid_eff, diff) + ask_term[None, :]
            
            M = np.outer(p_bid_eff, diff) + ask_term[None, :]
            idx_max = np.argmax(M, axis=1)
            
            # For each i, the best j is idx_max[i]
            best_j = idx_max
            best_val_i = M[np.arange(n_search), best_j] + p_bid_eff * search_depths - inv_cost
            
            best_i = np.argmax(best_val_i)
            best_val = best_val_i[best_i]
            best_bid = search_depths[best_i]
            best_ask = search_depths[best_j[best_i]]
            
            V[n, qi] = best_val
            bid_table[n, qi] = best_bid
            ask_table[n, qi] = best_ask
    
    return bid_table, ask_table
=======
"""
Optimal market making via pre-computed DP policy lookup.
The state space is (step, inventory) with 1000 steps and 61 inventory levels (-30..30).
We solve the Bellman equation backwards to find optimal bid/ask depths for each state.
Results are cached per regime so only the first call pays the DP cost.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 325.0  # safety margin before 330s kill


def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via backward DP.
    Uses a non-uniform depth grid concentrated near the expected optimum (~1/k),
    and vectorizes across inventory levels where possible.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_q = 2 * Q + 1  # 61 inventory levels
    q_range = np.arange(-Q, Q + 1)

    # Non-uniform depth grid: concentrate resolution near 1/k where the optimum
    # is expected. Use a log-spaced grid from a small floor to an upper bound.
    depth_min = 0.002
    depth_max = 8.0 / k
    # Use 80 points: 40 in [depth_min, 2/k] and 40 in [2/k, depth_max]
    n_search = 80
    half = n_search // 2
    d1 = np.linspace(depth_min, 2.0 / k, half)
    d2 = np.linspace(2.0 / k, depth_max, half)
    search_depths = np.concatenate([d1, d2[1:]])  # avoid duplicate at 2/k
    n_search = len(search_depths)

    # Precompute fill probabilities for all candidate depths
    exp_s = np.exp(-k * search_depths)
    p_s = 1.0 - np.exp(-A * exp_s * dt)

    # Value function
    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    # Precompute inventory costs for all q
    inv_costs = (gamma / 2.0) * sigma**2 * q_range**2 * dt  # shape (n_q,)

    # Precompute boundary masks
    no_bid_mask = np.zeros(n_q, dtype=bool)  # True where q == Q
    no_ask_mask = np.zeros(n_q, dtype=bool)  # True where q == -Q
    no_bid_mask[Q + Q] = True   # index for q=Q is Q+Q=2Q
    no_ask_mask[0] = True       # index for q=-Q is 0

    # Backward DP
    for n in range(steps - 1, -1, -1):
        # Check time limit
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            # Fallback: use a simple inventory-adjusted heuristic
            base_depth = 1.0 / k
            # Adjust: larger inventory (positive) -> widen bid, narrow ask
            # Adjust: smaller inventory (negative) -> narrow bid, widen ask
            # Simple linear adjustment
            adj = gamma * sigma**2 * q_range * dt / (2.0 * A * dt)
            bid_fallback = np.clip(base_depth + adj, depth_min, depth_max)
            ask_fallback = np.clip(base_depth - adj, depth_min, depth_max)
            for n2 in range(n, -1, -1):
                bid_table[n2, :] = bid_fallback
                ask_table[n2, :] = ask_fallback
            break

        Vn1 = V[n + 1, :]  # shape (n_q,)

        # For each q, compute the optimal (bid, ask)
        # We can vectorize partially across q
        # Vn1_q_plus: V[n+1, q+1] with boundary handling
        Vn1_q_plus = np.zeros(n_q)
        Vn1_q_minus = np.zeros(n_q)
        # q < Q: Vn1_q_plus = V[n+1, q+1]
        idx_plus = np.arange(n_q)
        valid_plus = (q_range < Q)  # shape (n_q,)
        Vn1_q_plus[valid_plus] = Vn1[idx_plus[valid_plus] + 1]
        Vn1_q_plus[~valid_plus] = Vn1[~valid_plus]  # at Q, stays at Q

        # q > -Q: Vn1_q_minus = V[n+1, q-1]
        valid_minus = (q_range > -Q)
        Vn1_q_minus[valid_minus] = Vn1[idx_plus[valid_minus] - 1]
        Vn1_q_minus[~valid_minus] = Vn1[~valid_minus]  # at -Q, stays at -Q

        # Effective fill probabilities
        # p_bid_eff[qi] = p_s if q < Q else 0
        p_bid_eff = np.where(no_bid_mask[:, None], 0.0, p_s[None, :])  # shape (n_q, n_search)
        p_ask_eff = np.where(no_ask_mask[:, None], 0.0, p_s[None, :])  # shape (n_q, n_search)

        # term1[j] for each q: p_ask_eff[j]*Vn1_q + (1-p_ask_eff[j])*Vn1_q_plus
        # term2[j] for each q: p_ask_eff[j]*Vn1_q_minus + (1-p_ask_eff[j])*Vn1_q
        # Shapes: (n_q, n_search)
        Vn1_q_b = Vn1[:, None]  # (n_q, 1)
        Vn1_q_plus_b = Vn1_q_plus[:, None]
        Vn1_q_minus_b = Vn1_q_minus[:, None]

        term1 = p_ask_eff * Vn1_q_b + (1.0 - p_ask_eff) * Vn1_q_plus_b  # (n_q, n_search)
        term2 = p_ask_eff * Vn1_q_minus_b + (1.0 - p_ask_eff) * Vn1_q_b  # (n_q, n_search)

        diff = term1 - term2  # (n_q, n_search)
        ask_term = term2 + search_depths[None, :] * p_ask_eff  # (n_q, n_search)

        # M[qi, i, j] = p_bid_eff[qi, i] * diff[qi, j] + ask_term[qi, j]
        # We want, for each qi and i, the j that maximizes M[qi, i, j]
        # M = p_bid_eff[..., :, None] * diff[..., None, :] + ask_term[..., None, :]
        # Shape: (n_q, n_search, n_search)
        M = p_bid_eff[:, :, None] * diff[:, None, :] + ask_term[:, None, :]

        # For each (qi, i), find best j
        best_j = np.argmax(M, axis=2)  # (n_q, n_search)
        best_val_j = np.take_along_axis(M, best_j[:, :, None], axis=2).squeeze(-1)  # (n_q, n_search)

        # Total[qi, i] = best_val_j[qi, i] + search_depths[i] * p_bid_eff[qi, i] - inv_costs[qi]
        total = best_val_j + p_bid_eff * search_depths[None, :] - inv_costs[:, None]  # (n_q, n_search)

        # Best i for each qi
        best_i = np.argmax(total, axis=1)  # (n_q,)
        best_val = total[np.arange(n_q), best_i]  # (n_q,)

        V[n, :] = best_val
        bid_table[n, :] = search_depths[best_i]
        ask_table[n, :] = search_depths[np.arange(n_q), best_j[np.arange(n_q), best_i]]

    return bid_table, ask_table
>>>>>>> REPLACE

