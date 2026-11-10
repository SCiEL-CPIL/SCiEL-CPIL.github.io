"""Market making with inventory risk - DP-based optimal policy."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache

# Precomputed optimal policies for each regime
_policy_cache = {}


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via backward DP.
    
    State: (step, inventory)
    Action: (bid_depth, ask_depth)
    
    For each state, we find the optimal depths that maximize expected reward.
    The reward structure:
    - Fill earns depth (positive for ask, negative for bid in terms of P&L)
    - Actually, "depths earned" means: bid fill gives -bid_depth (we buy at bid),
      ask fill gives +ask_depth (we sell at ask)
    - Inventory cost: (gamma/2) * sigma^2 * q^2 * dt per step
    
    Wait, re-reading: "depths earned minus sum of (gamma/2) sigma^2 q^2 dt"
    
    So the objective is: sum of (depths from fills) - sum of (gamma/2)*sigma^2*q^2*dt
    
    A bid fill: we buy at mid - bid_depth, so we earn -bid_depth (cost)
    An ask fill: we sell at mid + ask_depth, so we earn +ask_depth (revenue)
    
    Actually, in market making, the "depth earned" typically means the spread captured.
    Let me re-read: "A fill earns its depth"
    
    So bid fill earns bid_depth (positive), ask fill earns ask_depth (positive).
    This makes sense as the market maker's profit from the spread.
    
    Inventory cost: (gamma/2) * sigma^2 * q^2 * dt where q is inventory BEFORE fills.
    """
    dt = T / steps
    n_inv = 2 * Q + 1  # 61 states: -Q to Q
    
    # Value function: V[step, inv_index]
    # We compute backward from step=steps to step=0
    # At step=steps (terminal), V = 0 (no terminal charge)
    
    # For efficiency, precompute fill probabilities for a range of depths
    # Fill prob for bid: 1 - exp(-A * exp(-k * d) * dt)
    # For small A*exp(-k*d)*dt, this is approximately A*exp(-k*d)*dt
    
    # The optimal depth for a single side (ignoring inventory interaction) 
    # balances the marginal benefit of higher depth (more fills) vs cost (more P&L per fill)
    
    # Let's use a grid of possible depths and find the optimal one
    # Depth range: from very small to large
    # For k=1.5, 1/k ≈ 0.67; for k=0.75, 1/k ≈ 1.33
    
    # Maximum reasonable depth: where fill prob becomes negligible
    # A*exp(-k*d)*dt << 1 => d >> log(A*dt)/k
    # For A=140, dt=0.001, k=1.5: log(0.14)/1.5 ≈ -0.09, so depths up to ~5 are relevant
    # For A=50, dt=0.001, k=0.75: log(0.05)/0.75 ≈ -3.06, so depths up to ~8
    
    max_depth = 10.0
    n_depths = 200  # number of depth grid points
    depth_grid = np.linspace(0.001, max_depth, n_depths)
    
    # Precompute fill probabilities and rewards for each depth
    # fill_prob[d] = 1 - exp(-A * exp(-k*d) * dt)
    # reward from a fill at depth d: d (the depth earned)
    
    # For the DP, at each state (step, inv), we need to choose (bid_depth, ask_depth)
    # to maximize:
    #   fill_prob_bid * bid_depth + fill_prob_ask * ask_depth 
    #   - (gamma/2)*sigma^2*inv^2*dt
    #   + E[V[step+1, inv']]
    # 
    # where inv' depends on whether fills occur:
    #   No fill: inv' = inv
    #   Bid fill: inv' = inv + 1 (if inv < Q)
    #   Ask fill: inv' = inv - 1 (if inv > -Q)
    #   Both fills: inv' = inv (bid +1, ask -1)
    
    # The mid move doesn't affect the state (inventory is the only state variable
    # besides time, and the fill probabilities don't depend on mid)
    
    # So the DP is:
    # V[step, inv] = max_{bid_d, ask_d} [
    #   p_bid(bid_d) * bid_d + p_ask(ask_d) * ask_d
    #   - (gamma/2)*sigma^2*inv^2*dt
    #   + (1-p_bid)(1-p_ask) * V[step+1, inv]
    #   + p_bid*(1-p_ask) * V[step+1, min(inv+1, Q)]
    #   + (1-p_bid)*p_ask * V[step+1, max(inv-1, -Q)]
    #   + p_bid*p_ask * V[step+1, inv]
    # ]
    
    # where p_bid(d) = 1 - exp(-A*exp(-k*d)*dt) and similarly for ask.
    
    # Note: "no bid fill at q=Q" means if inv==Q, bid fill is impossible
    # "no ask fill at q=-Q" means if inv==-Q, ask fill is impossible
    
    # For efficiency, since the optimal bid and ask depths might be separable
    # in some approximation, but in general they're coupled through the value function.
    
    # Let's do a full 2D search over (bid_depth, ask_depth) at each state.
    # With 200x200 = 40000 combinations per state, and 1000*61 = 61000 states,
    # that's 2.4 billion operations - too slow.
    
    # Better approach: for each state, find optimal bid and ask separately
    # using the fact that the value function is concave in inventory.
    
    # Actually, let's reduce the depth grid and use smarter search.
    # The optimal depth for a side is where marginal benefit = marginal cost.
    
    # Alternative: use a smaller grid and vectorize.
    
    # Let's try: for each step and inventory, compute the optimal bid and ask depths.
    # We can vectorize over inventory for a given step.
    
    # Key insight: the fill probability and reward depend only on depth, not on inv.
    # So we can precompute arrays of fill probs and rewards for the depth grid.
    
    exp_neg_kd = np.exp(-k * depth_grid)
    fill_prob = 1.0 - np.exp(-A * exp_neg_kd * dt)
    # Reward per fill is just the depth
    fill_reward = depth_grid  # depth earned
    
    # For each (step, inv), the objective for choosing (bid_idx, ask_idx) is:
    # fill_prob[bid_idx]*depth_grid[bid_idx] + fill_prob[ask_idx]*depth_grid[ask_idx]
    # - (gamma/2)*sigma^2*inv^2*dt
    # + (1-fill_prob[bid_idx])*(1-fill_prob[ask_idx]) * V_next[inv]
    # + fill_prob[bid_idx]*(1-fill_prob[ask_idx]) * V_next[inv+1 if inv<Q else Q]
    # + (1-fill_prob[bid_idx])*fill_prob[ask_idx] * V_next[inv-1 if inv>-Q else -Q]
    # + fill_prob[bid_idx]*fill_prob[ask_idx] * V_next[inv]
    
    # The inventory cost term is constant for a given (step, inv), so it doesn't affect the argmax.
    
    # Let's restructure: for a given step, V_next[inv] is known for all inv.
    # For each inv, we need to find argmax over (bid_idx, ask_idx).
    
    # This is still expensive. Let's think about whether bid and ask can be optimized separately.
    
    # The coupling is through the value function terms. If V is linear in inv (which it 
    # approximately is for small inventory), then the cross terms simplify.
    
    # For now, let's use a reduced depth grid and accept some approximation.
    
    # Actually, let's try a different approach: use the analytical first-order condition.
    
    # For a single side (say bid), ignoring the other side's impact:
    # Maximize: p(d)*d + (1-p(d)) * V(inv+1) + p(d) * V(inv)
    # = p(d)*d + V(inv) + (1-p(d))*(V(inv+1) - V(inv))
    
    # where p(d) = 1 - exp(-A*exp(-k*d)*dt)
    
    # Let delta_V = V(inv+1) - V(inv)
    # Objective: p(d)*d + V(inv) + (1-p(d))*delta_V
    # = p(d)*(d - delta_V) + V(inv) + delta_V
    
    # So we maximize p(d)*(d - delta_V)
    # = (1 - exp(-A*exp(-k*d)*dt)) * (d - delta_V)
    
    # Taking derivative w.r.t. d and setting to 0:
    # This gives us an implicit equation for the optimal d.
    
    # For the two-sided case, it's more complex but similar.
    
    # Let's just do the DP with a moderate grid size.
    
    # Use 50 depth points for speed
    n_depths = 50
    depth_grid = np.linspace(0.01, max_depth, n_depths)
    exp_neg_kd = np.exp(-k * depth_grid)
    fill_prob = 1.0 - np.exp(-A * exp_neg_kd * dt)
    fill_reward = depth_grid
    
    # V[step] is an array of shape (n_inv,)
    # V[steps] = zeros
    V = np.zeros((steps + 1, n_inv))
    optimal_bid = np.zeros((steps, n_inv))
    optimal_ask = np.zeros((steps, n_inv))
    
    inv_values = np.arange(-Q, Q + 1)  # -Q, -Q+1, ..., Q
    
    # Precompute fill_prob and fill_reward as 1D arrays
    # For each step, compute V[step] from V[step+1]
    
    # For each step (backward), for each inventory level, find optimal bid and ask depths.
    # To speed up, we can vectorize over inventory.
    
    # For a given step, V_next = V[step+1]
    # For each inv index i (0 to n_inv-1), inv_val = inv_values[i]
    
    # The objective matrix for (bid_idx, ask_idx) at inventory i:
    # M[i, b, a] = fill_prob[b]*fill_reward[b] + fill_prob[a]*fill_reward[a]
    #              + (1-fill_prob[b])*(1-fill_prob[a]) * V_next[i]
    #              + fill_prob[b]*(1-fill_prob[a]) * V_next[min(i+1, n_inv-1)]
    #              + (1-fill_prob[b])*fill_prob[a] * V_next[max(i-1, 0)]
    #              + fill_prob[b]*fill_prob[a] * V_next[i]
    
    # This is a 3D array of shape (n_inv, n_depths, n_depths) = (61, 50, 50) = 152500
    # Per step, that's manageable.
    
    # Let's precompute the parts that don't depend on (b, a):
    # For inventory i:
    #   base = V_next[i]
    #   V_plus = V_next[min(i+1, n_inv-1)]  # after bid fill
    #   V_minus = V_next[max(i-1, 0)]        # after ask fill
    
    # M[i,b,a] = fill_prob[b]*fill_reward[b] + fill_prob[a]*fill_reward[a]
    #            + (1-fill_prob[b])*(1-fill_prob[a]) * base
    #            + fill_prob[b]*(1-fill_prob[a]) * V_plus
    #            + (1-fill_prob[b])*fill_prob[a] * V_minus
    #            + fill_prob[b]*fill_prob[a] * base
    
    # Let's expand:
    # = fill_prob[b]*fill_reward[b] + fill_prob[a]*fill_reward[a]
    #   + base*(1-fill_prob[b])*(1-fill_prob[a]) 
    #   + V_plus*fill_prob[b]*(1-fill_prob[a])
    #   + V_minus*(1-fill_prob[b])*fill_prob[a]
    #   + base*fill_prob[b]*fill_prob[a]
    
    # Group terms by fill_prob[b] and fill_prob[a]:
    # Let pb = fill_prob[b], pa = fill_prob[a], rb = fill_reward[b], ra = fill_reward[a]
    
    # = pb*rb + pa*ra
    #   + base*(1-pb)*(1-pa)
    #   + V_plus*pb*(1-pa)
    #   + V_minus*(1-pb)*pa
    #   + base*pb*pa
    
    # Expand (1-pb)*(1-pa) = 1 - pb - pa + pb*pa:
    # = pb*rb + pa*ra
    #   + base*(1 - pb - pa + pb*pa)
    #   + V_plus*pb - V_plus*pb*pa
    #   + V_minus*pa - V_minus*pb*pa
    #   + base*pb*pa
    
    # Group by pb*pa:
    # pb*pa terms: base - V_plus - V_minus + base = 2*base - V_plus - V_minus
    # Wait, let me redo:
    
    # pb*pa coefficient:
    #   from base*(1-pb)*(1-pa): +base
    #   from V_plus*pb*(1-pa): -V_plus
    #   from V_minus*(1-pb)*pa: -V_minus
    #   from base*pb*pa: +base
    # Total: 2*base - V_plus - V_minus
    
    # pb coefficient (without pa):
    #   from pb*rb: rb
    #   from base*(1-pb)*(1-pa): -base
    #   from V_plus*pb*(1-pa): +V_plus
    #   from V_minus*(1-pb)*pa: 0 (has pa)
    #   from base*pb*pa: 0 (has pa)
    # Total: rb - base + V_plus
    
    # pa coefficient (without pb):
    #   from pa*ra: ra
    #   from base*(1-pb)*(1-pa): -base
    #   from V_plus*pb*(1-pa): 0 (has pb)
    #   from V_minus*(1-pb)*pa: +V_minus
    #   from base*pb*pa: 0 (has pb)
    # Total: ra - base + V_minus
    
    # Constant (no pb, no pa):
    #   from base*(1-pb)*(1-pa): +base
    # Total: base
    
    # So M[i,b,a] = base 
    #   + pb*(rb - base + V_plus)
    #   + pa*(ra - base + V_minus)
    #   + pb*pa*(2*base - V_plus - V_minus)
    
    # This is a bilinear form in (pb, pa)! We can write it as:
    # M[i,b,a] = base + pb*c1[i] + pa*c2[i] + pb*pa*c3[i]
    # where c1[i] = rb - base + V_plus (but rb depends on b!)
    
    # Oops, rb depends on b, so it's not separable in that simple way.
    
    # Let me just compute the 3D array directly for each step.
    
    import time
    start_time = time.time()
    
    for step in range(steps - 1, -1, -1):
        V_next = V[step + 1]
        
        # For each inventory index i
        # We need to find argmax over (b, a) of M[i, b, a]
        
        # Vectorize: compute M for all i, b, a at once
        # M has shape (n_inv, n_depths, n_depths)
        
        # pb has shape (n_depths,), pa has shape (n_depths,)
        # For each i:
        #   base_i = V_next[i]
        #   V_plus_i = V_next[min(i+1, n_inv-1)]
        #   V_minus_i = V_next[max(i-1, 0)]
        
        base = V_next  # shape (n_inv,)
        V_plus = np.minimum(np.arange(n_inv) + 1, n_inv - 1)
        V_plus_vals = V_next[V_plus]  # shape (n_inv,)
        V_minus = np.maximum(np.arange(n_inv) - 1, 0)
        V_minus_vals = V_next[V_minus]  # shape (n_inv,)
        
        # M[i, b, a] = base[i]
        #   + fill_prob[b] * (fill_reward[b] - base[i] + V_plus_vals[i])
        #   + fill_prob[a] * (fill_reward[a] - base[i] + V_minus_vals[i])
        #   + fill_prob[b] * fill_prob[a] * (2*base[i] - V_plus_vals[i] - V_minus_vals[i])
        
        # Let's compute this efficiently.
        # Term 1: base[i] -> shape (n_inv, 1, 1)
        # Term 2: fill_prob[b] * (fill_reward[b] - base[i] + V_plus_vals[i])
        #   = fill_prob[b] * fill_reward[b] - fill_prob[b] * base[i] + fill_prob[b] * V_plus_vals[i]
        # Term 3: similar for ask
        # Term 4: fill_prob[b] * fill_prob[a] * (2*base[i] - V_plus_vals[i] - V_minus_vals[i])
        
        # Let's compute:
        # A_b = fill_prob * fill_reward  # shape (n_depths,)
        # For term 2: A_b[b] - fill_prob[b]*base[i] + fill_prob[b]*V_plus_vals[i]
        #   = A_b[b] + fill_prob[b]*(V_plus_vals[i] - base[i])
        
        # Similarly for term 3:
        # A_a = fill_prob * fill_reward
        # A_a[a] + fill_prob[a]*(V_minus_vals[i] - base[i])
        
        # Term 4: fill_prob[b]*fill_prob[a]*(2*base[i] - V_plus_vals[i] - V_minus_vals[i])
        
        A_side = fill_prob * fill_reward  # shape (n_depths,)
        
        # For each i, b, a:
        # M = base[i] 
        #   + A_side[b] + fill_prob[b]*(V_plus_vals[i] - base[i])
        #   + A_side[a] + fill_prob[a]*(V_minus_vals[i] - base[i])
        #   + fill_prob[b]*fill_prob[a]*(2*base[i] - V_plus_vals[i] - V_minus_vals[i])
        
        # Let's use broadcasting:
        # base_3d = base[:, None, None]  # (n_inv, 1, 1)
        # fp_b = fill_prob[None, :, None]  # (1, n_depths, 1)
        # fp_a = fill_prob[None, None, :]  # (1, 1, n_depths)
        # A_b = A_side[None, :, None]  # (1, n_depths, 1)
        # A_a = A_side[None, None, :]  # (1, 1, n_depths)
        # Vp = V_plus_vals[:, None, None]  # (n_inv, 1, 1)
        # Vm = V_minus_vals[:, None, None]  # (n_inv, 1, 1)
        
        # M = base_3d + A_b + A_a 
        #   + fp_b*(Vp - base_3d) + fp_a*(Vm - base_3d)
        #   + fp_b*fp_a*(2*base_3d - Vp - Vm)
        
        # This creates a (n_inv, n_depths, n_depths) array = (61, 50, 50) = 152500 floats
        # Per step, that's fine.
        
        base_3d = base[:, None, None]
        fp_b = fill_prob[None, :, None]
        fp_a = fill_prob[None, None, :]
        A_b = A_side[None, :, None]
        A_a = A_side[None, None, :]
        Vp = V_plus_vals[:, None, None]
        Vm = V_minus_vals[:, None, None]
        
        M = base_3d + A_b + A_a 
        M = M + fp_b * (Vp - base_3d) + fp_a * (Vm - base_3d)
        M = M + fp_b * fp_a * (2 * base_3d - Vp - Vm)
        
        # Find argmax over (b, a) for each i
        # M has shape (n_inv, n_depths, n_depths)
        # We want for each i, the (b, a) that maximizes M[i, b, a]
        
        # Reshape to (n_inv, n_depths*n_depths)
        M_flat = M.reshape(n_inv, n_depths * n_depths)
        best_idx = np.argmax(M_flat, axis=1)  # shape (n_inv,)
        
        best_b = best_idx // n_depths
        best_a = best_idx % n_depths
        
        optimal_bid[step] = depth_grid[best_b]
        optimal_ask[step] = depth_grid[best_a]
        V[step] = M_flat[np.arange(n_inv), best_idx]
        
        # Check time
        if time.time() - start_time > 250:
            # Fallback: use remaining steps with current best estimate
            for s in range(step - 1, -1, -1):
                V[s] = V[step]
                optimal_bid[s] = optimal_bid[step]
                optimal_ask[s] = optimal_ask[step]
            break
    
    return V, optimal_bid, optimal_ask, depth_grid


def _get_policy(params):
    """Get or compute the optimal policy for the given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_optimal_policy(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    
    Args:
        t: current time
        inventory: array of shape (4000,) with inventory levels
        mid: array of shape (4000,) with mid prices
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): tuple of two arrays of shape (4000,)
    """
    V, opt_bid, opt_ask, depth_grid = _get_policy(params)
    
    steps = params['steps']
    Q = params['Q']
    n_paths = len(inventory)
    
    # Current step index
    dt = params['T'] / steps
    step_idx = int(round(t / dt))
    step_idx = min(max(step_idx, 0), steps - 1)
    
    # Map inventory to index: inv in [-Q, Q] -> index in [0, 2Q]
    inv = np.asarray(inventory, dtype=np.int32)
    inv = np.clip(inv, -Q, Q)
    inv_idx = inv + Q  # 0 to 2Q
    
    # Get optimal depths for this step and each inventory level
    bid_depths = opt_bid[step_idx][inv_idx]
    ask_depths = opt_ask[step_idx][inv_idx]
    
    # Ensure no bid fill at Q, no ask fill at -Q
    # At Q, bid_depth should be inf (no quote)
    # At -Q, ask_depth should be inf (no quote)
    bid_depths = np.where(inv_idx == 2 * Q, np.inf, bid_depths)
    ask_depths = np.where(inv_idx == 0, np.inf, ask_depths)
    
    # Ensure depths are valid: > -3/k, not -inf, not NaN
    k = params['k']
    min_depth = -3.0 / k
    bid_depths = np.where(bid_depths < min_depth, min_depth + 1e-9, bid_depths)
    ask_depths = np.where(ask_depths < min_depth, min_depth + 1e-9, ask_depths)
    bid_depths = np.where(np.isnan(bid_depths), 1.0 / k, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), 1.0 / k, ask_depths)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

# EVOLVE-BLOCK-END
