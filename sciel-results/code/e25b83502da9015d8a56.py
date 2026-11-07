"""Optimal market making with inventory risk: Exact Joint HJB Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin

_policy_cache = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precomputes the optimal policy using exact 2D backward induction.
    For each state (n, q), we maximize over (d_bid, d_ask) the joint objective:
    R(d_b, d_a) + E[V_next(q')]
    where q' depends on both fill events.
    """
    q_min = -Q
    q_max = Q
    n_q = q_max - q_min + 1
    
    dt = T / steps
    
    # Depth grid
    # We need a grid fine enough to capture the optimum.
    # Max depth: 5.0 is sufficient.
    max_depth = 5.0
    n_d = 200  # Reduced from 1000 to allow exact 2D search within time budget
    d_grid = np.linspace(0.0, max_depth, n_d)
    
    # Precompute fill probabilities
    # p[d] = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = np.exp(-k * d_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    
    # V_next[q_idx]
    V_next = np.zeros(n_q)
    
    # Store optimal depths
    optimal_bid = np.full((steps, n_q), np.inf)
    optimal_ask = np.full((steps, n_q), np.inf)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Time check
        if time.time() > _DEADLINE - 5.0:
            # If we run out of time, we must stop. 
            # The quote function will handle the partial cache or fallback.
            # But since we compute backwards, we have the early steps (n near steps-1) 
            # but not the early time steps (n near 0). 
            # This is a problem because quote() is called from n=0.
            # We need to ensure we finish n=0.
            # Let's estimate time per step.
            # 61 states * (200x200 ops) = 61 * 40000 = 2.44M ops per time step.
            # 1000 steps * 2.44M = 2.44B ops.
            # NumPy can do ~100M-1B ops/s. This might be too slow for exact 2D.
            # Let's try to optimize the 2D search.
            
            # Objective for state q:
            # max_{i,j} [ p_b[i] * d_b[i] + p_a[j] * d_a[j] 
            #              + p_b[i] * p_a[j] * V_next[q]
            #              + p_b[i] * (1-p_a[j]) * V_next[q+1]
            #              + (1-p_b[i]) * p_a[j] * V_next[q-1]
            #              + (1-p_b[i]) * (1-p_a[j]) * V_next[q]
            #              - hold_cost ]
            
            # Let's simplify the E[V_next] term:
            # E[V] = V_next[q] + p_b[i] * (V_next[q+1] - V_next[q]) * (1-p_a[j]) 
            #                   + p_a[j] * (V_next[q-1] - V_next[q]) * (1-p_b[i])
            # This is not separable.
            
            # However, we can vectorize the 2D grid for each q.
            # Create meshgrid of p_b and p_a.
            # This creates a (n_d, n_d) array.
            
            V_curr = np.full(n_q, -np.inf)
            
            # Precompute differences for all q
            # diff_plus[q] = V_next[q+1] - V_next[q]
            # diff_minus[q] = V_next[q-1] - V_next[q]
            # We handle boundaries separately.
            
            for q_idx in range(n_q):
                q = q_idx + q_min
                hold_cost = (gamma / 2.0) * (sigma ** 2) * (q ** 2) * dt
                
                if q == Q:
                    # No bid fill. p_b is effectively 0 for the transition, but we can still quote?
                    # "no bid fill at q=Q". This means if we quote, it doesn't fill.
                    # So p_b = 0 in the transition logic.
                    # But the reward from quoting a bid? "A fill earns its depth". No fill, no reward.
                    # So objective depends only on ask.
                    # E[V] = V_next[q] + p_a[j] * (V_next[q-1] - V_next[q])
                    # Obj = p_a[j] * d_a[j] + p_a[j] * (V_next[q-1] - V_next[q]) - hold_cost
                    # This is 1D optimization over j.
                    val_diff_ask = V_next[q_idx - 1] - V_next[q_idx]
                    obj_ask = p_fill * (d_grid + val_diff_ask)
                    max_idx_ask = np.argmax(obj_ask)
                    ask_depth = d_grid[max_idx_ask]
                    ask_val = obj_ask[max_idx_ask]
                    
                    V_curr[q_idx] = ask_val - hold_cost
                    optimal_bid[n, q_idx] = np.inf
                    optimal_ask[n, q_idx] = ask_depth
                    
                elif q == -Q:
                    # No ask fill.
                    val_diff_bid = V_next[q_idx + 1] - V_next[q_idx]
                    obj_bid = p_fill * (d_grid + val_diff_bid)
                    max_idx_bid = np.argmax(obj_bid)
                    bid_depth = d_grid[max_idx_bid]
                    bid_val = obj_bid[max_idx_bid]
                    
                    V_curr[q_idx] = bid_val - hold_cost
                    optimal_bid[n, q_idx] = bid_depth
                    optimal_ask[n, q_idx] = np.inf
                    
                else:
                    # Normal case: 2D optimization
                    # We need to maximize over i (bid) and j (ask)
                    # Term = p_b[i]*d_b[i] + p_a[j]*d_a[j] + p_b[i]*(1-p_a[j])*V[q+1] + (1-p_b[i])*p_a[j]*V[q-1] + (1-p_b[i])*(1-p_a[j])*V[q]
                    
                    # Let's expand:
                    # = V[q] + p_b[i]*(d_b[i] + V[q+1] - V[q]) - p_b[i]*p_a[j]*(V[q+1] - V[q])
                    #         + p_a[j]*(d_a[j] + V[q-1] - V[q]) - p_b[i]*p_a[j]*(V[q-1] - V[q])
                    # Wait, let's do it carefully.
                    # E[V] = p_b p_a V_q + p_b(1-p_a) V_{q+1} + (1-p_b)p_a V_{q-1} + (1-p_b)(1-p_a) V_q
                    # = V_q (p_b p_a + 1 - p_b - p_a + p_a) 
                    # = V_q (1 - p_b + p_b p_a) 
                    # No, (1-p_b)(1-p_a) + p_b p_a = 1 - p_b - p_a + 2 p_b p_a
                    # This is getting messy. Let's just compute the matrix directly.
                    
                    # p_b: (n_d,), p_a: (n_d,)
                    # P_b: (n_d, 1), P_a: (1, n_d)
                    P_b = p_fill[:, np.newaxis]
                    P_a = p_fill[np.newaxis, :]
                    D_b = d_grid[:, np.newaxis]
                    D_a = d_grid[np.newaxis, :]
                    
                    V_q = V_next[q_idx]
                    V_qp1 = V_next[q_idx + 1]
                    V_qm1 = V_next[q_idx - 1]
                    
                    # Transition probabilities
                    # P_bb = P_b * P_a  (both fill) -> q
                    # P_ba = P_b * (1-P_a) (bid fill, ask no) -> q+1
                    # P_ab = (1-P_b) * P_a (bid no, ask fill) -> q-1
                    # P_aa = (1-P_b) * (1-P_a) (both no) -> q
                    
                    E_V = (P_b * P_a * V_q + 
                           P_b * (1.0 - P_a) * V_qp1 + 
                           (1.0 - P_b) * P_a * V_qm1 + 
                           (1.0 - P_b) * (1.0 - P_a) * V_q)
                    
                    Reward = P_b * D_b + P_a * D_a
                    
                    Total_Obj = Reward + E_V - hold_cost
                    
                    max_idx = np.argmax(Total_Obj)
                    i, j = np.unravel_index(max_idx, Total_Obj.shape)
                    
                    bid_depth = d_grid[i]
                    ask_depth = d_grid[j]
                    V_curr[q_idx] = Total_Obj[i, j]
                    
                    optimal_bid[n, q_idx] = bid_depth
                    optimal_ask[n, q_idx] = ask_depth
                    
        V_next = V_curr
    
    return {
        'bid_depths': optimal_bid,
        'ask_depths': optimal_ask,
        'q_min': q_min
    }

def _get_policy_for_params(params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key in _policy_cache:
        return _policy_cache[key]
    
    if time.time() > _DEADLINE - 10.0:
        return None
    
    policy = _precompute_regime(sigma, A, k, gamma, T, steps, Q)
    _policy_cache[key] = policy
    return policy

def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    policy = _get_policy_for_params(params)
    
    if policy is None:
        # Fallback heuristic
        dt_rem = T - t
        R = gamma * (sigma ** 2) * inventory * dt_rem
        delta_star = 1.0 / k
        
        bid_depths = R + delta_star
        ask_depths = -R + delta_star
        
        min_depth = 1e-6
        bid_depths = np.maximum(bid_depths, min_depth)
        ask_depths = np.maximum(ask_depths, min_depth)
        
        return bid_depths, ask_depths
    
    q_min = policy['q_min']
    bid_table = policy['bid_depths']
    ask_table = policy['ask_depths']
    
    idxs = inventory - q_min
    idxs = np.clip(idxs, 0, 2 * Q)
    
    bid_depths = bid_table[n, idxs]
    ask_depths = ask_table[n, idxs]
    
    # Ensure validity
    min_valid = -3.0 / k
    bid_depths = np.maximum(bid_depths, min_valid)
    ask_depths = np.maximum(ask_depths, min_valid)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
