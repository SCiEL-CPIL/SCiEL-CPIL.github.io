"""Market-making with inventory risk: Exact DP with analytic policy extraction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard (300 s hard limit).
_T0 = time.time()
_DEADLINE = 290.0  # seconds of wall clock budget for the whole run.

def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)

# Cache optimal policies per parameter regime
_POLICY_CACHE = {}

def _solve_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Solves the discrete DP to find optimal bid and ask depths for each
    inventory level at each time step.
    Returns a dictionary with 'bids' and 'asks' arrays of shape (steps+1, 2*Q+1).
    bids[n, idx] is the optimal bid depth for inventory q at step n (0-indexed from start).
    Actually, we compute backwards. V[n] is value with n steps remaining.
    Policy for step n (0 to steps-1) uses V[n+1] to determine depths.
    We store policies for step n (time n*dt) which uses V[n+1] (value after step).
    So policy[n, q] is the depth to quote at time n*dt given inventory q.
    """
    if _remaining_budget() < 10.0:
        return None

    dt = T / steps
    inv_k = 1.0 / k
    max_depth = 3.0 * inv_k
    min_depth = -3.0 / k
    
    # Use a finer grid: q_step = 1
    q_step = 1.0
    q_min = float(-Q)
    q_max = float(Q)
    nq = int(round((q_max - q_min) / q_step)) + 1
    qs = q_min + q_step * np.arange(nq)
    
    risk_coeff = 0.5 * gamma * sigma * sigma * dt

    # V[n] is the value function with n steps remaining.
    # V[0] = 0
    # V[n] = max_{b,a} (b + a - risk*q^2 + E[V[n-1]])
    # We iterate n from 1 to steps.
    
    V_prev = np.zeros(nq)  # V[0]
    
    # Policies: policy_bid[n-1, q] is the optimal bid for step n-1 (using V[n-1] as next state)
    # Wait, let's be precise.
    # At step n (0-indexed), we are at time t_n. Inventory q.
    # We choose b, a.
    # We receive reward b + a - risk*q^2.
    # Inventory becomes q'.
    # We get V_remaining[n] where n is steps remaining after this step.
    # Total steps = steps.
    # At start (t=0), steps_remaining = steps.
    # At t=(steps-1)dt, steps_remaining = 1.
    # At t=steps*dt, steps_remaining = 0.
    
    # Let V_rem[r] be the value function with r steps remaining.
    # V_rem[0] = 0.
    # V_rem[r] = max_{b,a} (b + a - risk*q^2 + E[V_rem[r-1]])
    
    # We want the policy for the step where steps_remaining = r.
    # This policy is determined by V_rem[r-1].
    # Let's store policies for r = 1..steps.
    # policy_bid[r-1, q] is the bid depth to use when steps_remaining = r.
    
    opt_bids = np.zeros((steps, nq))
    opt_asks = np.zeros((steps, nq))
    
    # Precompute indices for q-1 and q+1
    # q_idx = 0..nq-1 corresponds to q = -Q, -Q+1, ..., Q
    # q-1 is at q_idx - 1
    # q+1 is at q_idx + 1
    idx = np.arange(nq)
    idx_lo = np.clip(idx - 1, 0, nq - 1)
    idx_hi = np.clip(idx + 1, 0, nq - 1)
    
    # V_prev is V_rem[r-1]. Initially V_rem[0] = 0.
    V_next = np.zeros(nq)
    
    for r in range(1, steps + 1):
        # Calculate dV_bid and dV_ask for each q
        # dV_bid(q) = V_rem[r-1](q-1) - V_rem[r-1](q)
        # dV_ask(q) = V_rem[r-1](q+1) - V_rem[r-1](q)
        
        # V_prev is V_rem[r-1]
        dV_bid = V_prev[idx_lo] - V_prev
        dV_ask = V_prev[idx_hi] - V_prev
        
        # Optimal bid depth
        # Term in objective: b + (1 - A dt e^{-kb}) dV_bid
        # = b + dV_bid - A dt e^{-kb} dV_bid
        # If dV_bid < 0, let C = -dV_bid > 0.
        # Obj = b - C + A dt C e^{-kb}
        # Deriv: 1 - A dt C k e^{-kb} = 0 => e^{-kb} = 1 / (A dt C k)
        # b = -1/k ln(1 / (A dt (-dV_bid) k))
        
        # If dV_bid >= 0, we want to minimize fill prob (make b large) to avoid reducing value?
        # If dV_bid > 0, term is b + C + A dt (-C) e^{-kb} = b + C - A dt C e^{-kb}.
        # Deriv: 1 + A dt C k e^{-kb} > 0. Function is increasing in b.
        # So max b = 3/k.
        
        C_bid = np.maximum(-dV_bid, 0.0)
        # If C_bid is 0, b = max_depth
        # Else b = -1/k ln(1 / (A dt C_bid k))
        
        # Handle numerical issues
        # If A dt C_bid k is very small, ln is large negative, b is large positive.
        # If A dt C_bid k is very large, ln is large positive, b is large negative.
        # b must be in [min_depth, max_depth].
        
        arg_bid = A * dt * C_bid * k
        # b = -1/k * log(1/arg) = 1/k * log(arg)
        # Wait: e^{-kb} = 1/arg => -kb = -ln(arg) => b = 1/k ln(arg).
        
        # If arg <= 1, ln(arg) <= 0, b <= 0.
        # If arg > 1, ln(arg) > 0, b > 0.
        
        # We need to be careful with arg=0 (C_bid=0).
        # If C_bid=0, we set b = max_depth.
        
        b_opt = np.full(nq, max_depth)
        mask_bid = C_bid > 1e-12
        if np.any(mask_bid):
            arg_b = arg_bid[mask_bid]
            # Clip arg_b to avoid log(0) or inf
            arg_b = np.clip(arg_b, 1e-300, 1e300)
            b_vals = (1.0 / k) * np.log(arg_b)
            # Clip to valid range
            b_vals = np.clip(b_vals, min_depth, max_depth)
            b_opt[mask_bid] = b_vals
            
        # Optimal ask depth
        # Symmetric logic
        C_ask = np.maximum(-dV_ask, 0.0)
        a_opt = np.full(nq, max_depth)
        mask_ask = C_ask > 1e-12
        if np.any(mask_ask):
            arg_a = A * dt * C_ask * k
            arg_a = np.clip(arg_a, 1e-300, 1e300)
            a_vals = (1.0 / k) * np.log(arg_a)
            a_vals = np.clip(a_vals, min_depth, max_depth)
            a_opt[mask_ask] = a_vals
            
        # Store policies for step r-1 (0-indexed)
        opt_bids[r-1] = b_opt
        opt_asks[r-1] = a_opt
        
        # Update V_rem[r]
        # V_rem[r](q) = b_opt + a_opt - risk*q^2 + E[V_rem[r-1]]
        
        p_bid = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
        p_ask = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
        
        # Boundary conditions:
        # At q = Q (max inventory, most long), we cannot buy more (bid fill increases q).
        # Wait, inventory q. Bid fill: buy 1 unit. q -> q+1.
        # If q = Q, q+1 > Q. Is this allowed?
        # Task says: "no bid fill at q = Q, no ask fill at q = -Q".
        # So if q = Q, p_bid = 0.
        # If q = -Q, p_ask = 0.
        
        q_arr = qs
        p_bid = np.where(q_arr >= q_max, 0.0, p_bid)
        p_ask = np.where(q_arr <= q_min, 0.0, p_ask)
        
        # Expected next value
        # V_next(q) = p_bid * V_prev(q+1) + p_ask * V_prev(q-1) + (1 - p_bid - p_ask) * V_prev(q)
        # Note: My previous definition of dV was:
        # dV_bid = V(q-1) - V(q). This implies bid fill moves to q-1?
        # Let's re-read the model.
        # "A fill earns its depth and moves inventory one unit"
        # "bid hit ... moves inventory one unit"
        # Usually, bid hit means we sell at bid? No, in market making:
        # Bid quote: we are the buyer. If hit, we buy. Inventory increases.
        # Ask quote: we are the seller. If hit, we sell. Inventory decreases.
        
        # Let's check the objective: "depths earned".
        # If I buy at bid depth b (price Mid - b), I pay Mid - b.
        # If I sell at ask depth a (price Mid + a), I receive Mid + a.
        # "depths earned" usually means the spread captured.
        # If I buy at b, I "earn" b relative to mid?
        # The objective is: "depths earned minus ...".
        # This implies the reward is b + a.
        # This is consistent with capturing the spread.
        
        # Inventory dynamics:
        # Bid fill (buy): q -> q + 1.
        # Ask fill (sell): q -> q - 1.
        
        # Let's re-evaluate the dV terms.
        # V(q) = b + a - R + p_b V(q+1) + p_a V(q-1) + (1-p_b-p_a) V(q)
        # V(q) - V(q) = b + a - R + p_b (V(q+1) - V(q)) + p_a (V(q-1) - V(q))
        # Let dV_plus = V(q+1) - V(q).
        # Let dV_minus = V(q-1) - V(q).
        # Term with b: b + p_b dV_plus.
        # p_b = 1 - A dt e^{-kb}.
        # b + (1 - A dt e^{-kb}) dV_plus = b + dV_plus - A dt e^{-kb} dV_plus.
        # Deriv wrt b: 1 + A dt dV_plus k e^{-kb}.
        # Set to 0: e^{-kb} = -1 / (A dt k dV_plus).
        # This requires dV_plus < 0.
        # If dV_plus > 0, deriv > 0, so b should be max.
        
        # So:
        # Bid depth depends on dV_plus = V(q+1) - V(q).
        # Ask depth depends on dV_minus = V(q-1) - V(q).
        
        # My previous code used:
        # dV_bid = V(q-1) - V(q). This is dV_minus.
        # So I was calculating the optimal ASK depth using the BID logic?
        # Or did I mix up the indices?
        
        # Let's re-derive carefully.
        # Bid side:
        # We choose b.
        # Reward: b.
        # Prob of fill: p_b.
        # Next state: q+1.
        # Value change: V(q+1) - V(q) = dV_plus.
        # Contribution: b + p_b dV_plus.
        # Optimal b* satisfies e^{-k b*} = -1 / (A dt k dV_plus) if dV_plus < 0.
        # Else b* = max_depth.
        
        # Ask side:
        # We choose a.
        # Reward: a.
        # Prob of fill: p_a.
        # Next state: q-1.
        # Value change: V(q-1) - V(q) = dV_minus.
        # Contribution: a + p_a dV_minus.
        # Optimal a* satisfies e^{-k a*} = -1 / (A dt k dV_minus) if dV_minus < 0.
        # Else a* = max_depth.
        
        # So:
        # b_opt uses dV_plus = V(q+1) - V(q).
        # a_opt uses dV_minus = V(q-1) - V(q).
        
        # In my code:
        # dV_bid = V_prev[idx_lo] - V_prev  => V(q-1) - V(q) = dV_minus.
        # dV_ask = V_prev[idx_hi] - V_prev  => V(q+1) - V(q) = dV_plus.
        
        # So I calculated:
        # b_opt based on dV_minus. (WRONG, should be dV_plus)
        # a_opt based on dV_plus. (WRONG, should be dV_minus)
        
        # I need to swap them.
        
        # Recalculate with correct dVs
        dV_plus = V_prev[idx_hi] - V_prev  # For Bid
        dV_minus = V_prev[idx_lo] - V_prev # For Ask
        
        C_bid = np.maximum(-dV_plus, 0.0)
        b_opt = np.full(nq, max_depth)
        mask_bid = C_bid > 1e-12
        if np.any(mask_bid):
            arg_b = A * dt * C_bid[mask_bid] * k
            arg_b = np.clip(arg_b, 1e-300, 1e300)
            b_vals = (1.0 / k) * np.log(arg_b)
            b_vals = np.clip(b_vals, min_depth, max_depth)
            b_opt[mask_bid] = b_vals
            
        C_ask = np.maximum(-dV_minus, 0.0)
        a_opt = np.full(nq, max_depth)
        mask_ask = C_ask > 1e-12
        if np.any(mask_ask):
            arg_a = A * dt * C_ask[mask_ask] * k
            arg_a = np.clip(arg_a, 1e-300, 1e300)
            a_vals = (1.0 / k) * np.log(arg_a)
            a_vals = np.clip(a_vals, min_depth, max_depth)
            a_opt[mask_ask] = a_vals
            
        # Update stored policies (overwrite the wrong ones)
        opt_bids[r-1] = b_opt
        opt_asks[r-1] = a_opt
        
        # Update V
        p_bid = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
        p_ask = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
        
        p_bid = np.where(q_arr >= q_max, 0.0, p_bid)
        p_ask = np.where(q_arr <= q_min, 0.0, p_ask)
        
        V_next = b_opt + a_opt - risk_coeff * q_arr**2 + \
                 p_bid * V_prev[idx_hi] + p_ask * V_prev[idx_lo] + \
                 (1.0 - p_bid - p_ask) * V_prev
        
        V_prev = V_next
        
        if r % 50 == 0 and _remaining_budget() < 15.0:
            return None

    return {
        'bids': opt_bids,
        'asks': opt_asks,
        'qs': qs,
        'q_min': q_min,
        'q_max': q_max,
        'q_step': q_step
    }


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
        policy = _solve_policy(sigma, A, k, gamma, T, steps, Q)
        _POLICY_CACHE[key] = policy
    policy = _POLICY_CACHE[key]

    inv_k = 1.0 / k
    max_depth = 3.0 * inv_k
    min_depth = -3.0 / k

    if policy is None:
        # Fallback to analytic shift if DP failed
        shift = gamma * sigma * sigma * 0.5 / (A * k)
        q = inventory.astype(float)
        bid = np.clip(inv_k + q * shift, min_depth, max_depth)
        ask = np.clip(inv_k - q * shift, min_depth, max_depth)
        return bid, ask

    # Determine step index
    # t = n * dt. n = t / dt.
    # dt = T / steps.
    dt = T / steps
    n = int(np.clip(t / dt, 0, steps - 1))
    
    # Inventory interpolation
    q = inventory.astype(float)
    q_min = policy['q_min']
    q_step = policy['q_step']
    
    # Map q to grid index
    idx_float = (q - q_min) / q_step
    i0 = np.floor(idx_float).astype(int)
    i1 = i0 + 1
    i0 = np.clip(i0, 0, len(policy['qs']) - 1)
    i1 = np.clip(i1, 0, len(policy['qs']) - 1)
    frac = idx_float - np.floor(idx_float)
    
    bids_table = policy['bids']
    asks_table = policy['asks']
    
    # Get depths for step n
    bid_n = bids_table[n]
    ask_n = asks_table[n]
    
    # Interpolate
    bid_depths = (1.0 - frac) * bid_n[i0] + frac * bid_n[i1]
    ask_depths = (1.0 - frac) * ask_n[i0] + frac * ask_n[i1]
    
    # Ensure valid values
    bid_depths = np.nan_to_num(bid_depths, nan=inv_k, posinf=max_depth, neginf=min_depth)
    ask_depths = np.nan_to_num(ask_depths, nan=inv_k, posinf=max_depth, neginf=min_depth)
    
    # Final clip to valid range
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
