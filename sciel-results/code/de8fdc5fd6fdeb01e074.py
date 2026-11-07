"""Optimal market making with inventory risk: Joint 2D HJB Solver with Coarse-to-Fine Refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

_policy_cache = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precomputes the optimal policy using Joint 2D backward induction.
    
    Unlike the separable approximation, this solver accounts for the coupling
    between bid and ask fills in both the immediate reward and the future value
    expectation. It uses a coarse-to-fine grid search to efficiently find the
    optimal (bid_depth, ask_depth) pair for each state.
    """
    q_min = -Q
    q_max = Q
    n_q = q_max - q_min + 1
    
    dt = T / steps
    
    # Depth grid definitions
    # Max depth 5/k is sufficient as prob of fill becomes negligible
    max_depth = 5.0 / k
    
    # Coarse grid for initial search
    n_coarse = 60
    d_coarse = np.linspace(0.0, max_depth, n_coarse)
    
    # Fine grid size for local refinement
    n_fine = 30
    
    # Precompute p_fill for coarse grid
    p_coarse = 1.0 - np.exp(-A * np.exp(-k * d_coarse) * dt)
    
    # V_next: shape (n_q,)
    V_next = np.zeros(n_q)
    
    # Optimal depths tables
    optimal_bid = np.full((steps, n_q), 0.0)
    optimal_ask = np.full((steps, n_q), 0.0)
    
    # Helper to compute fine grid p_fill on the fly
    # We will store the best depths found
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_q)
        
        for q_idx in range(n_q):
            q = q_idx + q_min
            
            hold_cost = (gamma / 2.0) * (sigma ** 2) * (q ** 2) * dt
            base_val = V_next[q_idx]
            
            # Determine valid Deltas
            # Delta_plus = V_next[q+1] - V_next[q] (impact of bid fill)
            # Delta_minus = V_next[q-1] - V_next[q] (impact of ask fill)
            
            # Boundaries
            if q < Q:
                delta_plus = V_next[q_idx + 1] - V_next[q_idx]
                bid_allowed = True
            else:
                delta_plus = 0.0 # Not used
                bid_allowed = False
                
            if q > -Q:
                delta_minus = V_next[q_idx - 1] - V_next[q_idx]
                ask_allowed = True
            else:
                delta_minus = 0.0 # Not used
                ask_allowed = False
            
            best_val = -np.inf
            best_db = 0.0
            best_da = 0.0
            
            if bid_allowed and ask_allowed:
                # 2D Optimization
                # Coarse Pass
                # p_b shape (n_c,), p_a shape (n_c,)
                # p_b[:, None] shape (n_c, 1)
                # p_a[None, :] shape (1, n_c)
                
                pb = p_coarse[:, None]
                pa = p_coarse[None, :]
                
                # Immediate Reward:
                # d_b * p_b (bid fills) + d_a * p_a (ask fills)
                # Note: if both fill, we get d_b + d_a.
                # Total immediate = d_b * p_b * (1 - p_a) + d_a * p_a * (1 - p_b) + (d_b + d_a) * p_b * p_a
                # Simplifies to: d_b * p_b + d_a * p_a
                # Let's verify:
                # Bid only: d_b
                # Ask only: d_a
                # Both: d_b + d_a
                # Exp: (1-pb)(1-pa)*0 + pb(1-pa)*db + (1-pb)pa*da + pb*pa*(db+da)
                # = pb*db - pb*pa*db + pa*da - pb*pa*da + pb*pa*db + pb*pa*da
                # = pb*db + pa*da
                # Yes, immediate reward is simply pb*db + pa*da.
                
                d_b_grid = d_coarse[:, None]
                d_a_grid = d_coarse[None, :]
                
                immediate = pb * d_b_grid + pa * d_a_grid
                
                # Future Value Expectation:
                # V_next[q] + pb(1-pa)Delta_plus + pa(1-pb)Delta_minus
                # Note: if both fill, q' = q, so change is 0.
                # Term: V_next[q] + pb*pa*0 + pb(1-pa)Delta_plus + (1-pb)pa*Delta_minus
                future_delta = pb * (1.0 - pa) * delta_plus + (1.0 - pb) * pa * delta_minus
                
                total_obj = immediate + future_delta
                
                # Find max in coarse grid
                max_idx_flat = np.argmax(total_obj)
                max_i, max_j = np.unravel_index(max_idx_flat, total_obj.shape)
                
                # Coarse best
                coarse_db = d_coarse[max_i]
                coarse_da = d_coarse[max_j]
                coarse_val = total_obj[max_i, max_j]
                
                # Fine Pass
                # Create local fine grids around coarse best
                # Ensure indices are within bounds
                i_lo = max(0, max_i - 2)
                i_hi = min(n_coarse, max_i + 3)
                j_lo = max(0, max_j - 2)
                j_hi = min(n_coarse, max_j + 3)
                
                # Create fine grids
                # We want a fine grid covering [d_coarse[i_lo], d_coarse[i_hi]]
                # and [d_coarse[j_lo], d_coarse[j_hi]]
                d_b_fine = np.linspace(d_coarse[i_lo], d_coarse[i_hi], n_fine)
                d_a_fine = np.linspace(d_coarse[j_lo], d_coarse[j_hi], n_fine)
                
                pb_fine = 1.0 - np.exp(-A * np.exp(-k * d_b_fine) * dt)
                pa_fine = 1.0 - np.exp(-A * np.exp(-k * d_a_fine) * dt)
                
                pb_f = pb_fine[:, None]
                pa_f = pa_fine[None, :]
                
                d_b_f_grid = d_b_fine[:, None]
                d_a_f_grid = d_a_fine[None, :]
                
                immediate_f = pb_f * d_b_f_grid + pa_f * d_a_f_grid
                future_delta_f = pb_f * (1.0 - pa_f) * delta_plus + (1.0 - pb_f) * pa_f * delta_minus
                
                total_obj_f = immediate_f + future_delta_f
                
                max_idx_flat_f = np.argmax(total_obj_f)
                max_i_f, max_j_f = np.unravel_index(max_idx_flat_f, total_obj_f.shape)
                
                best_db = d_b_fine[max_i_f]
                best_da = d_a_fine[max_j_f]
                best_val = total_obj_f[max_i_f, max_j_f]
                
            elif bid_allowed:
                # Only bid allowed (ask is inf)
                # Maximize: pb * db + pb * delta_plus
                obj = p_coarse * (d_coarse + delta_plus)
                max_idx = np.argmax(obj)
                best_db = d_coarse[max_idx]
                best_da = np.inf
                best_val = obj[max_idx]
                
            elif ask_allowed:
                # Only ask allowed
                obj = p_coarse * (d_coarse + delta_minus)
                max_idx = np.argmax(obj)
                best_db = np.inf
                best_da = d_coarse[max_idx]
                best_val = obj[max_idx]
                
            else:
                # No quotes allowed
                best_db = np.inf
                best_da = np.inf
                best_val = 0.0
            
            optimal_bid[n, q_idx] = best_db
            optimal_ask[n, q_idx] = best_da
            V_curr[q_idx] = base_val - hold_cost + best_val
            
        V_next = V_curr
        
        # Time check
        if time.time() > _DEADLINE - 20.0:
            # If we run out of time, we can't finish the precomputation.
            # We should signal this or handle it in quote.
            # For now, we break and return partial policy, but quote needs to handle missing steps.
            # Better to just let it run if possible, but 300s is tight.
            # 1000 steps * 61 q * (60*60 + 30*30) ops ~ 1000 * 61 * 4500 ~ 275M ops.
            # NumPy vectorized 2D arrays are fast, but 275M floating point ops might take a few seconds.
            # Actually, 60x60 = 3600 elements. 1000*61 = 61000 states.
            # 61000 * (3600 + 900) ~ 275 million operations.
            # At 1 GFLOP/s, this is 0.275s. Should be fine.
            pass

    return {
        'bid_depths': optimal_bid,
        'ask_depths': optimal_ask,
        'q_min': q_min
    }

def _get_policy_for_params(params):
    """
    Retrieves or computes the policy for the given parameters.
    """
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
    
    # Check time budget
    if time.time() > _DEADLINE - 10.0:
        return None
    
    # Precompute
    policy = _precompute_regime(sigma, A, k, gamma, T, steps, Q)
    _policy_cache[key] = policy
    return policy

def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed HJB policy.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    # Get step index
    dt = T / steps
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    # Get policy
    policy = _get_policy_for_params(params)
    
    if policy is None:
        # Fallback: Myopic Avellaneda-Stoikov approximation
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
    
    # Map inventory to indices
    idxs = inventory - q_min
    idxs = np.clip(idxs, 0, 2 * Q)
    
    # Retrieve depths
    bid_depths = bid_table[n, idxs]
    ask_depths = ask_table[n, idxs]
    
    # Ensure validity
    # The problem states values below -3/k are invalid.
    # Our grid starts at 0, so depths are non-negative.
    # However, if the optimal strategy involves "no quote", we return np.inf.
    # The problem allows np.inf.
    
    # Check for any NaNs or infs that might have slipped through (shouldn't happen)
    # And ensure no depth is < -3/k (which is negative, so 0 is safe)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
