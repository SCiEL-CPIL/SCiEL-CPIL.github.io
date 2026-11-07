import numpy as np
import time
import hashlib

_POLICY_CACHE = {}
_DEADLINE = None
_DEADLINE_SET = False

def _get_deadline():
    global _DEADLINE, _DEADLINE_SET
    if not _DEADLINE_SET:
        _DEADLINE = time.time() + 320.0
        _DEADLINE_SET = True
    return _DEADLINE

def _remaining_time():
    return _get_deadline() - time.time()

def _params_key(params):
    return (params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q'])

def _compute_policy(params, deadline=None):
    if deadline is None:
        deadline = _remaining_time()
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Discretize inventory: -Q to Q inclusive
    n_states = 2 * Q + 1
    q_min = -Q
    q_max = Q
    
    # Depth grid: log-spaced from small to large
    # Optimal depths are typically around 1/k, maybe range from 0.1/k to 5/k
    depth_min = 0.05 / k
    depth_max = 5.0 / k
    n_depths = 200
    depths = np.geomspace(depth_min, depth_max, n_depths)
    
    # Precompute transition probabilities
    # bid hit prob: 1 - exp(-A * exp(-k * depth) * dt)
    # ask hit prob: same formula
    exp_neg_k_depths = np.exp(-k * depths)
    hit_probs = 1.0 - np.exp(-A * exp_neg_k_depths * dt)
    # hit_probs shape: (n_depths,)
    
    # Value function: V[i, q_idx] where i is step index, q_idx is inventory index
    # V[i, q] = optimal expected value from step i onward starting with inventory q
    # At step i, we choose bid_depth and ask_depth
    # Rewards: depth_bid if bid fills, depth_ask if ask fills
    # Inventory changes: q - 1 if bid fills, q + 1 if ask fills
    # No bid fill at q = Q, no ask fill at q = -Q
    
    # Terminal: V[steps, q] = 0 (no terminal charge)
    V = np.zeros((steps + 1, n_states))
    
    # For each step, compute optimal depths for each inventory level
    # bid_opt[i, q_idx] = optimal bid depth index
    # ask_opt[i, q_idx] = optimal ask depth index
    
    bid_opt = np.zeros((steps, n_states), dtype=np.int32)
    ask_opt = np.zeros((steps, n_states), dtype=np.int32)
    
    q_indices = np.arange(n_states)
    q_values = q_indices - Q  # actual inventory values
    
    # Backward induction
    for i in range(steps - 1, -1, -1):
        if time.time() > deadline:
            # Return what we have so far
            break
        
        # For each inventory level, find optimal bid and ask depths
        # V[i, q] = max over bid_depth, ask_depth of:
        #   hit_prob_bid * (bid_depth + V[i+1, q-1]) + (1-hit_prob_bid) * V[i+1, q]
        #   + hit_prob_ask * (ask_depth + V[i+1, q+1]) + (1-hit_prob_ask) * V[i+1, q]
        #   - (gamma/2) * sigma^2 * q^2 * dt
        #   - hit_prob_bid * hit_prob_ask * (correction for double fill?)
        
        # Wait, the model says bid and ask hit independently. If both hit, what happens?
        # "A fill earns its depth and moves inventory one unit"
        # If both fill: bid fill decreases inventory by 1, ask fill increases by 1, net 0
        # Earn both depths
        # Probability of both: hit_bid * hit_ask
        # Probability of only bid: hit_bid * (1-hit_ask)
        # Probability of only ask: (1-hit_bid) * hit_ask
        # Probability of neither: (1-hit_bid) * (1-hit_ask)
        
        # V[i, q] = max over b, a of:
        #   p_b * p_a * (b_depth + a_depth + V[i+1, q])
        #   + p_b * (1-p_a) * (b_depth + V[i+1, q-1])
        #   + (1-p_b) * p_a * (a_depth + V[i+1, q+1])
        #   + (1-p_b) * (1-p_a) * V[i+1, q]
        #   - (gamma/2) * sigma^2 * q^2 * dt
        # Where p_b = hit_prob[b], p_a = hit_prob[a]
        # q-1 clipped to not go below -Q (if q=-Q, no ask fill means q stays)
        # Actually: "no bid fill at q = Q, no ask fill at q = -Q"
        # So at q=Q, bid can't fill, only ask can
        # At q=-Q, ask can't fill, only bid can
        
        # Let's handle boundaries carefully
        # For q > -Q: ask fill moves to q+1 (up to Q)
        # For q < Q: bid fill moves to q-1 (down to -Q)
        
        # Vectorize over inventory levels
        # For each q_idx, compute expected value for each (b, a) pair
        # This is a 2D optimization over (b, a) for each q
        
        # To speed up, note that the value is separable-ish but not exactly
        # Let's use a grid search with vectorization
        
        # For efficiency, precompute V[i+1, :] which is shape (n_states,)
        V_next = V[i + 1, :]
        
        # Handle boundaries
        # q_values shape: (n_states,)
        # Can bid fill: q < Q
        # Can ask fill: q > -Q
        
        # For each q, the optimal (b, a) might be found by:
        # Expanding: 
        # EV = p_b * p_a * (d_b + d_a) + p_b * (1-p_a) * d_b + (1-p_b) * p_a * d_a
        #    + p_b * p_a * V[i+1, q] + p_b*(1-p_a)*V[i+1,q-1] + (1-p_b)*p_a*V[i+1,q+1] + (1-p_b)*(1-p_a)*V[i+1,q]
        #    - cost
        # = p_b * p_a * (d_b + d_a + V_next[q])
        #   + p_b * (1-p_a) * (d_b + V_next[q-1])
        #   + (1-p_b) * p_a * (d_a + V_next[q+1])
        #   + (1-p_b) * (1-p_a) * V_next[q]
        #   - cost
        
        # Let me define for each q:
        # V_next[q], V_next[q-1] (or V_next[q] if q=-Q, since no ask fill... wait)
        
        # At q = -Q: ask can't fill, so only bid matters
        # At q = Q: bid can't fill, so only ask matters
        # At interior: both can fill
        
        # Let's handle interior first, then boundaries
        
        cost = (gamma / 2.0) * sigma**2 * q_values**2 * dt  # shape (n_states,)
        
        # For interior q (-Q < q < Q):
        # Let A_q = V_next[q-1], B_q = V_next[q], C_q = V_next[q+1]
        # EV(b, a) = p_b*p_a*(d_b+d_a+B_q) + p_b*(1-p_a)*(d_b+A_q) + (1-p_b)*p_a*(d_a+C_q) + (1-p_b)*(1-p_a)*B_q - cost_q
        # = p_b*p_a*d_b + p_b*p_a*d_a + p_b*(1-p_a)*d_b + (1-p_b)*p_a*d_a
        #   + p_b*p_a*B_q + p_b*(1-p_a)*A_q + (1-p_b)*p_a*C_q + (1-p_b)*(1-p_a)*B_q
        #   - cost_q
        # = d_b*(p_b*p_a + p_b*(1-p_a)) + d_a*(p_b*p_a + (1-p_b)*p_a)
        #   + p_b*p_a*B_q + p_b*(1-p_a)*A_q + (1-p_b)*p_a*C_q + (1-p_b)*(1-p_a)*B_q
        #   - cost_q
        # = d_b*p_b + d_a*p_a
        #   + p_b*p_a*B_q + p_b*(1-p_a)*A_q + (1-p_b)*p_a*C_q + (1-p_b)*(1-p_a)*B_q
        #   - cost_q
        # = d_b*p_b + d_a*p_a
        #   + p_b*(p_a*B_q + (1-p_a)*A_q) + (1-p_b)*(p_a*C_q + (1-p_a)*B_q)
        #   - cost_q
        
        # This is nice! For fixed q, it's:
        # = [d_b*p_b + p_b*(p_a*B_q + (1-p_a)*A_q)] + [(1-p_b)*p_a*C_q + (1-p_b)*(1-p_a)*B_q + d_a*p_a]
        # Hmm, still coupled through p_a in the b part and p_b in the a part
        
        # Alternative: for each b, find best a, then find best b
        # Or just do a 2D grid search vectorized
        
        # Let's do: for each b (n_depths options), compute the best a
        # Then for each q, find best b
        
        # Precompute:
        # d_b * p_b: shape (n_depths,)
        # d_a * p_a: shape (n_depths,)
        
        d_b = depths  # shape (n_depths,)
        d_a = depths
        p_b_arr = hit_probs  # shape (n_depths,)
        p_a_arr = hit_probs
        
        # For interior q values:
        interior_mask = (q_values > -Q) & (q_values < Q)
        interior_indices = np.where(interior_mask)[0]
        
        if len(interior_indices) > 0:
            A_q = V_next[interior_indices - 1]  # V[i+1, q-1]
            B_q = V_next[interior_indices]       # V[i+1, q]
            C_q = V_next[interior_indices + 1]   # V[i+1, q+1]
            
            # For each b_idx and a_idx, compute EV for each interior q
            # EV[b, a, q_idx] = d_b[b]*p_b[b] + d_a[a]*p_a[a]
            #   + p_b[b]*(p_a[a]*B_q[q_idx] + (1-p_a[a])*A_q[q_idx])
            #   + (1-p_b[b])*(p_a[a]*C_q[q_idx] + (1-p_a[a])*B_q[q_idx])
            #   - cost[q_idx]
            
            # Vectorize: 
            # p_b_col: (n_depths, 1)
            # p_a_row: (1, n_depths)
            # d_b_p_b: (n_depths,)
            # d_a_p_a: (n_depths,)
            
            pb_col = p_b_arr[:, np.newaxis]  # (n_depths, 1)
            pa_row = p_a_arr[np.newaxis, :]  # (1, n_depths)
            dbpb = d_b * p_b_arr  # (n_depths,)
            dap_a = d_a * p_a_arr  # (n_depths,)
            
            # For each (b, a) pair and each q:
            # term1 = dbpb[b]  -> broadcast to (n_depths_b, 1, n_q)
            # term2 = dapa[a]  -> broadcast to (1, n_depths_a, n_q)
            # term3 = pb[b] * (pa[a] * B_q + (1-pa[a]) * A_q)
            #        = pb_col * (pa_row * B_q[None,None,:] + (1-pa_row) * A_q[None,None,:])
            # term4 = (1-pb[b]) * (pa[a] * C_q + (1-pa[a]) * B_q)
            #        = (1-pb_col) * (pa_row * C_q[None,None,:] + (1-pa_row) * B_q[None,None,:])
            # term5 = -cost[None,None,:]
            
            Bq_3d = B_q[np.newaxis, np.newaxis, :]  # (1, 1, n_interior)
            Cq_3d = C_q[np.newaxis, np.newaxis, :]
            Aq_3d = A_q[np.newaxis, np.newaxis, :]
            cost_3d = cost[interior_indices][np.newaxis, np.newaxis, :]
            
            term1 = dbpb[:, np.newaxis, np.newaxis]  # (n_b, 1, 1)
            term2 = dap_a[np.newaxis, :, np.newaxis]  # (1, n_a, 1)
            term3 = pb_col[:, np.newaxis, np.newaxis] * (pa_row[np.newaxis, :, np.newaxis] * Bq_3d + (1 - pa_row[np.newaxis, :, np.newaxis]) * Aq_3d)
            term4 = (1 - pb_col[:, np.newaxis, np.newaxis]) * (pa_row[np.newaxis, :, np.newaxis] * Cq_3d + (1 - pa_row[np.newaxis, :, np.newaxis]) * Bq_3d)
            term5 = -cost_3d
            
            EV = term1 + term2 + term3 + term4 + term5  # (n_b, n_a, n_interior)
            
            # Find best (b, a) for each q
            # argmax over (b, a) axis
            best_flat = np.argmax(EV, axis=(0, 1))  # (n_interior,)
            best_b = best_flat // n_depths
            best_a = best_flat % n_depths
            
            bid_opt[i, interior_indices] = best_b
            ask_opt[i, interior_indices] = best_a
            V[i, interior_indices] = np.max(EV, axis=(0, 1))
        
        # Boundary: q = Q (only ask can fill)
        q_idx_Q = 2 * Q
        # EV(b, a) = only ask matters, bid doesn't fill
        # = p_a * (d_a + V_next[q]) + (1-p_a) * V_next[q] - cost
        # = p_a * d_a + V_next[q] - cost
        # Optimal b: doesn't matter, set to 0
        # Optimal a: argmax of p_a * d_a
        if steps - i > 0:
            best_a_Q = np.argmax(p_a_arr * d_a)
            ask_opt[i, q_idx_Q] = best_a_Q
            bid_opt[i, q_idx_Q] = 0
            V[i, q_idx_Q] = p_a_arr[best_a_Q] * d_a[best_a_Q] + V_next[q_idx_Q] - cost[q_idx_Q]
        
        # Boundary: q = -Q (only bid can fill)
        q_idx_nQ = 0
        # EV(b, a) = only bid matters
        # = p_b * d_b + V_next[q] - cost
        # Optimal a: doesn't matter, set to 0
        # Optimal b: argmax of p_b * d_b
        best_b_nQ = np.argmax(p_b_arr * d_b)
        bid_opt[i, q_idx_nQ] = best_b_nQ
        ask_opt[i, q_idx_nQ] = 0
        V[i, q_idx_nQ] = p_b_arr[best_b_nQ] * d_b[best_b_nQ] + V_next[q_idx_nQ] - cost[q_idx_nQ]
    
    return depths, bid_opt, ask_opt, V

def quote(t, inventory, mid, params):
    key = _params_key(params)
    
    if key not in _POLICY_CACHE:
        deadline = _remaining_time() - 5.0  # safety margin
        if deadline < 1.0:
            # Not enough time, use simple heuristic
            depth = 1.0 / params['k']
            depths = np.full(len(inventory), depth)
            return depths, depths.copy()
        
        try:
            depths_grid, bid_opt, ask_opt, V = _compute_policy(params, deadline)
            _POLICY_CACHE[key] = (depths_grid, bid_opt, ask_opt, V)
        except Exception:
            depth = 1.0 / params['k']
            depths = np.full(len(inventory), depth)
            return depths, depths.copy()
    
    depths_grid, bid_opt, ask_opt, V = _POLICY_CACHE[key]
    
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step from t
    dt = params['T'] / steps
    step_idx = int(round(t / dt))
    step_idx = max(0, min(step_idx, steps - 1))
    
    # Inventory to index
    q_idx = np.clip(inventory, -Q, Q) + Q
    q_idx = q_idx.astype(np.int32)
    
    bid_depth_idx = bid_opt[step_idx, q_idx]
    ask_depth_idx = ask_opt[step_idx, q_idx]
    
    bid_depths = depths_grid[bid_depth_idx]
    ask_depths = depths_grid[ask_depth_idx]
    
    return bid_depths, ask_depths

