"""Market-making with inventory risk: DP over inventory grid with exact policy lookup."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 290.0

def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)

# Cache optimal policies per parameter regime
_POLICY_CACHE = {}

def _solve_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Solves the discrete DP to find optimal bid and ask depths for each inventory level
    at each time step.
    Returns:
        opt_bids: array of shape (steps, 2*Q+1)
        opt_asks: array of shape (steps, 2*Q+1)
    """
    if _remaining_budget() < 10.0:
        return None

    dt = T / steps
    inv_k = 1.0 / k
    max_depth = 3.0 * inv_k
    q_min = -Q
    q_max = Q
    nq = 2 * Q + 1
    qs = np.arange(q_min, q_max + 1)
    
    # Precompute constants
    risk_coeff = 0.5 * gamma * sigma * sigma * dt
    
    # V_next is V at step n+1. We start with V at step steps (terminal) = 0
    V_next = np.zeros(nq)
    
    # Storage for optimal depths
    # opt_bids[n] corresponds to the action taken at step n (from t_n to t_{n+1})
    # This action is determined by V_{n+1}
    opt_bids = np.zeros((steps, nq))
    opt_asks = np.zeros((steps, nq))
    
    # Iterate backwards from n = steps-1 down to 0
    for n in range(steps - 1, -1, -1):
        # V_next is currently V_{n+1}
        # We need to find optimal b, a for each q to maximize V_n(q)
        
        # For each q, we need V_{n+1}(q-1), V_{n+1}(q), V_{n+1}(q+1)
        # Indices:
        # q_idx = q - q_min
        # q-1 -> q_idx - 1
        # q+1 -> q_idx + 1
        
        idx = np.arange(nq)
        idx_lo = np.clip(idx - 1, 0, nq - 1)
        idx_hi = np.clip(idx + 1, 0, nq - 1)
        
        V_lo = V_next[idx_lo]
        V_mid = V_next[idx]
        V_hi = V_next[idx_hi]
        
        # Delta V for bid fill (q -> q-1): V_lo - V_mid
        # Delta V for ask fill (q -> q+1): V_hi - V_mid
        dV_bid = V_lo - V_mid
        dV_ask = V_hi - V_mid
        
        # Optimal depths derived from:
        # b* = (1/k) * ln( k * A * dt * dV_bid )  if dV_bid > 0
        # a* = (1/k) * ln( k * A * dt * (-dV_ask) ) if dV_ask < 0
        # If dV_bid <= 0, we don't want to buy, so b -> inf (or max_depth)
        # If dV_ask >= 0, we don't want to sell, so a -> inf (or max_depth)
        
        # Note: The approximation p ~ A e^{-kb} dt is valid for small p.
        # If the optimal b is very small (large prob), the linear approx fails.
        # But for market making, depths are usually such that prob is moderate.
        
        # Calculate optimal b
        # term_b = k * A * dt * dV_bid
        term_b = k * A * dt * dV_bid
        # If term_b <= 0, log is invalid, set b to max_depth (don't quote or quote far)
        # Actually if dV_bid <= 0, buying increases cost or has no benefit, so we should not quote bid.
        # Setting b to max_depth minimizes fill prob.
        valid_b = term_b > 0
        b_opt = np.full(nq, max_depth)
        b_opt[valid_b] = (1.0 / k) * np.log(term_b[valid_b])
        b_opt = np.clip(b_opt, 0.0, max_depth)
        
        # Calculate optimal a
        term_a = k * A * dt * (-dV_ask)
        valid_a = term_a > 0
        a_opt = np.full(nq, max_depth)
        a_opt[valid_a] = (1.0 / k) * np.log(term_a[valid_a])
        a_opt = np.clip(a_opt, 0.0, max_depth)
        
        # Store optimal actions for step n
        opt_bids[n] = b_opt
        opt_asks[n] = a_opt
        
        # Compute V_n using these optimal actions
        # p_b = 1 - exp(-A exp(-k b) dt)
        # p_a = 1 - exp(-A exp(-k a) dt)
        
        # Boundary conditions:
        # At q = Q (idx = nq-1), no bid fill (can't buy more). p_b = 0.
        # At q = -Q (idx = 0), no ask fill (can't sell more). p_a = 0.
        
        p_b = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
        p_a = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
        
        # Apply boundary constraints
        p_b[nq - 1] = 0.0
        p_a[0] = 0.0
        
        # Reward: depths earned - risk charge
        # Risk charge uses q^2
        q_arr = qs
        reward = (b_opt + a_opt) - risk_coeff * q_arr**2
        
        # Continuation value
        # E[V_{n+1}(q')] = p_b V_{n+1}(q-1) + p_a V_{n+1}(q+1) + (1-p_b-p_a) V_{n+1}(q)
        # Note: If p_b > 0 but q=Q, we set p_b=0, so V_lo is not used for that term.
        # The formula holds.
        
        cont = p_b * V_lo + p_a * V_hi + (1.0 - p_b - p_a) * V_mid
        
        V_curr = reward + cont
        V_next = V_curr
        
        if n % 100 == 0 and _remaining_budget() < 5.0:
            return None
            
    return opt_bids, opt_asks

def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _POLICY_CACHE:
        result = _solve_policy(sigma, A, k, gamma, T, steps, Q)
        _POLICY_CACHE[key] = result
        
    data = _POLICY_CACHE[key]
    
    if data is None:
        # Fallback to analytic shift if DP failed/timed out
        inv_k = 1.0 / k
        max_depth = 3.0 * inv_k
        shift = gamma * sigma * sigma * 0.5 / (A * k)
        q = inventory.astype(float)
        bid = np.clip(inv_k + q * shift, 0.0, max_depth)
        ask = np.clip(inv_k - q * shift, 0.0, max_depth)
        bid = np.nan_to_num(bid, nan=inv_k, posinf=max_depth, neginf=0.0)
        ask = np.nan_to_num(ask, nan=inv_k, posinf=max_depth, neginf=0.0)
        return bid, ask
        
    opt_bids, opt_asks = data
    
    # Determine step index
    dt = T / steps
    n = int(t / dt)
    # Clamp n to [0, steps-1]
    n = max(0, min(n, steps - 1))
    
    # Map inventory to index
    # inventory is in [-Q, Q]
    # idx = q - (-Q) = q + Q
    q = inventory
    idx = q + Q
    
    # Clip indices just in case of any float issues, though inventory is int
    idx = np.clip(idx, 0, 2 * Q)
    
    # Lookup depths
    bid_depths = opt_bids[n][idx]
    ask_depths = opt_asks[n][idx]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
