import numpy as np
import time
import threading

# Cache for precomputed tables
_TABLE_CACHE = {}
_TABLE_LOCK = threading.Lock()

# Global deadline tracking
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill

def _check_deadline():
    return time.time() >= _DEADLINE

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for all steps and inventories.
    Uses vectorized interpolation for the optimality condition to speed up
    the fixed-point iteration.
    """
    dt = T / steps
    N = steps
    max_q = Q
    min_q = -Q
    n_states = 2 * Q + 1
    
    V_next = np.zeros(n_states)
    opt_bid = np.zeros((N, n_states))
    opt_ask = np.zeros((N, n_states))
    
    # Precompute interpolation grid for the optimality condition
    # We want to solve for x in: g(x) - x * h(x) = C
    # where g(x) = 1 - exp(-A exp(-kx) dt)
    # and h(x) = -g'(x) = k A dt exp(-kx) exp(-A exp(-kx) dt)
    # This is equivalent to phi(x) = g(x)/h(x) - x = C/h(x)? No.
    # The FOC is: g'(x)(x + C) + g(x) = 0  => -h(x)(x+C) + g(x) = 0 => g(x)/h(x) - x = C.
    # So we define phi(x) = g(x)/h(x) - x.
    # phi(x) is decreasing. We can invert it using interpolation.
    
    x_max = 10.0 / k  # Reasonable upper bound for depth
    n_grid = 4000     # High resolution for accuracy
    x_grid = np.linspace(0, x_max, n_grid)
    
    # Calculate phi(x) on the grid
    exp_kx = np.exp(-k * x_grid)
    exp_Au = np.exp(-A * exp_kx * dt)
    g = 1.0 - exp_Au
    h = k * A * dt * exp_kx * exp_Au
    
    # Handle x=0 where h might be small?
    # At x=0, h = k A dt exp(-A dt). If A dt is small, h is small.
    # But g is also small. Limit of g/h as x->0 is 1/(k A dt).
    # Let's just compute it. If h is 0, phi is inf.
    # For x > 0, h > 0.
    
    # Avoid division by zero
    h_safe = np.where(h < 1e-15, 1e-15, h)
    phi = g / h_safe - x_grid
    
    # phi is decreasing.
    # We want to find x given C such that phi(x) = C.
    # Since phi is decreasing, -phi is increasing.
    # np.interp requires xp to be increasing.
    # So we use xp = -phi, fp = x_grid.
    # We want to find x for a given C.
    # Equation: phi(x) = C  =>  -phi(x) = -C.
    # So we look up -C in -phi to get x.
    
    neg_phi = -phi
    
    # Ensure neg_phi is strictly increasing for interp stability
    # phi is strictly decreasing, so neg_phi is strictly increasing.
    # However, numerical noise might cause issues.
    # We can just use it as is, np.interp handles non-strictly increasing by taking the last value, 
    # but for a monotonic function it should be fine.
    
    V_curr = np.zeros(n_states)
    
    # Precompute q values for risk calculation
    q_vals = np.arange(n_states) - Q
    risk = (gamma / 2.0) * (sigma**2) * (q_vals**2) * dt
    
    # Precompute shifts for V_next
    # V_next_plus[i] = V_next[i+1] for i < 2Q, 0 otherwise
    # V_next_minus[i] = V_next[i-1] for i > 0, 0 otherwise
    V_next_plus = np.zeros(n_states)
    V_next_minus = np.zeros(n_states)
    V_next_plus[:-1] = V_next[1:]
    V_next_minus[1:] = V_next[:-1]
    
    # Initialize b and a
    b = np.zeros(n_states)
    a = np.zeros(n_states)
    
    # Precompute constants for delta calculations
    # delta_plus[i] = V_next[i+1] - V_next[i]
    # delta_minus[i] = V_next[i-1] - V_next[i]
    
    for n in range(N - 1, -1, -1):
        if _check_deadline():
            # Return what we have so far, fill remaining with a reasonable default
            depth_default = 1.0 / k
            for n_fill in range(n, N):
                opt_bid[n_fill, :] = depth_default
                opt_ask[n_fill, :] = depth_default
            return opt_bid, opt_ask
        
        # Update deltas based on current V_next
        delta_plus = np.zeros(n_states)
        delta_minus = np.zeros(n_states)
        delta_plus[:-1] = V_next[1:] - V_next[:-1]
        delta_minus[1:] = V_next[:-1] - V_next[1:]
        
        # Fixed point iteration for b and a
        # Max iterations
        max_iter = 50
        for _iter in range(max_iter):
            if _check_deadline():
                break
            
            # Compute p_a from current a
            p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
            
            # Compute C_b for each q
            # C_b = (1 - p_a) * delta_plus - p_a * delta_minus
            # Note: delta_plus and delta_minus are 0 at boundaries where moves are impossible,
            # but the formula holds generally if we define the "missing" V as 0 or handle boundaries separately.
            # Actually, the derivation:
            # V = p_b(1-p_a)V_{q+1} + (1-p_b)p_a V_{q-1} + ...
            # Term for p_b: p_b [ (1-p_a)V_{q+1} - (1-p_b... wait.
            # Let's re-verify C_b.
            # E[V] = p_b(1-p_a)V_{q+1} + (1-p_b)p_a V_{q-1} + (1-p_b)(1-p_a)V_q
            # = p_b V_{q+1} - p_b p_a V_{q+1} + p_a V_{q-1} - p_b p_a V_{q-1} + V_q - p_b V_q - p_a V_q + p_b p_a V_q
            # = V_q + p_b (V_{q+1} - V_q) - p_a (V_q - V_{q-1}) - p_b p_a (V_{q+1} - V_{q-1})
            # Let Delta_+ = V_{q+1} - V_q, Delta_- = V_q - V_{q-1}.
            # E[V] = V_q + p_b Delta_+ - p_a Delta_- - p_b p_a (Delta_+ + Delta_-)
            # We maximize: Reward + E[V]
            # Reward = p_b b + p_a a
            # Objective for b: p_b b + p_b Delta_+ - p_b p_a (Delta_+ + Delta_-)
            # = p_b [ b + Delta_+ (1 - p_a) - p_a Delta_- ]
            # So C_b = Delta_+ (1 - p_a) - p_a Delta_-
            # My previous formula: C_b = (1 - p_a) * delta_plus - p_a * delta_minus.
            # Here delta_plus = V_{q+1} - V_q = Delta_+.
            # Here delta_minus = V_{q-1} - V_q = -Delta_-.
            # So C_b = (1 - p_a) * delta_plus - p_a * (-delta_minus) = (1 - p_a) delta_plus + p_a delta_minus?
            # Let's check the sign of delta_minus in the code.
            # Code: delta_minus[1:] = V_next[:-1] - V_next[1:]  => V_{i-1} - V_i = -Delta_-.
            # So Delta_- = -delta_minus.
            # Formula: C_b = (1 - p_a) Delta_+ - p_a Delta_- = (1 - p_a) delta_plus - p_a (-delta_minus) = (1 - p_a) delta_plus + p_a delta_minus.
            
            # Let's re-derive carefully.
            # E[V] = p_b (1-p_a) V_{q+1} + (1-p_b) p_a V_{q-1} + (1-p_b)(1-p_a) V_q
            # = p_b V_{q+1} - p_b p_a V_{q+1} + p_a V_{q-1} - p_b p_a V_{q-1} + V_q - p_b V_q - p_a V_q + p_b p_a V_q
            # = V_q + p_b (V_{q+1} - V_q) + p_a (V_{q-1} - V_q) - p_b p_a (V_{q+1} - V_{q-1})
            # Let D_+ = V_{q+1} - V_q. Let D_- = V_{q-1} - V_q.
            # E[V] = V_q + p_b D_+ + p_a D_- - p_b p_a (D_+ + D_-)
            # Term with p_b: p_b [ D_+ - p_a (D_+ + D_-) ] = p_b [ D_+ (1 - p_a) - p_a D_- ]
            # So C_b = D_+ (1 - p_a) - p_a D_-
            
            # In code variables:
            # delta_plus = D_+
            # delta_minus = D_-
            # So C_b = delta_plus * (1 - p_a) - p_a * delta_minus
            
            C_b = delta_plus * (1.0 - p_a) - p_a * delta_minus
            
            # Solve for b: phi(b) = C_b
            # b = interp(-C_b, neg_phi, x_grid)
            b_new = np.interp(-C_b, neg_phi, x_grid)
            
            # Update p_b
            p_b = 1.0 - np.exp(-A * np.exp(-k * b_new) * dt)
            
            # Compute C_a
            # Symmetry:
            # Term with p_a: p_a [ D_- (1 - p_b) - p_b D_+ ]
            # So C_a = delta_minus * (1 - p_b) - p_b * delta_plus
            
            C_a = delta_minus * (1.0 - p_b) - p_b * delta_plus
            
            a_new = np.interp(-C_a, neg_phi, x_grid)
            
            # Convergence check
            if np.allclose(b_new, b, atol=1e-8) and np.allclose(a_new, a, atol=1e-8):
                b = b_new
                a = a_new
                break
            
            b = b_new
            a = a_new
        
        # Store optimal depths
        opt_bid[n, :] = b
        opt_ask[n, :] = a
        
        # Compute V_curr
        # p_b, p_a are already computed from final b, a
        
        # Transitions
        prob_q_plus = p_b * (1.0 - p_a)
        prob_q_minus = (1.0 - p_b) * p_a
        
        # Handle boundaries
        # q = Q (idx = 2Q): no bid fill. prob_q_plus = 0.
        # q = -Q (idx = 0): no ask fill. prob_q_minus = 0.
        prob_q_plus[-1] = 0.0
        prob_q_minus[0] = 0.0
        
        # Reward
        reward = p_b * b + p_a * a
        # At boundaries, the "impossible" fill probability is 0, so the reward term for that side is 0.
        # The formula reward = p_b b + p_a a is correct because p_b[-1] is not necessarily 0?
        # Wait, at q=Q, we don't quote bid? Or we quote but it doesn't fill?
        # "no bid fill at q = Q".
        # If we quote bid depth b, p_b is the probability of fill.
        # But the model says "no bid fill". This implies the transition to Q+1 is forbidden.
        # Does it mean we cannot quote bid? Or if we quote, p_b=0?
        # Usually in market making, if you are at max inventory, you might still quote bid but it won't fill?
        # Or you just don't quote?
        # The problem statement: "no bid fill at q = Q, no ask fill at q = -Q".
        # This usually means the state space is bounded. If a fill would push it out, it doesn't happen.
        # So, if q=Q, and a bid hit occurs, inventory stays at Q? Or the hit is ignored?
        # "A fill earns its depth and moves inventory one unit... no bid fill at q=Q".
        # This implies the event "bid fill" has probability 0 at q=Q.
        # So p_b should be 0 at q=Q?
        # If p_b=0, then we don't earn the depth b.
        # So we should set b=0 or p_b=0?
        # If we set b=0, p_b is small but not 0.
        # The optimal policy would set b=0 (or very large) to avoid the fill?
        # But if the fill is impossible, the cost of quoting is 0 (no inventory change) and the reward is 0 (no fill).
        # So it doesn't matter what we quote?
        # Actually, if the fill is impossible, the term p_b * b is 0 because p_b=0.
        # So we should enforce p_b=0 at q=Q.
        # How? By setting b such that p_b=0? Or just masking?
        # The interpolation gives a b. If we mask the reward and transition, it's fine.
        # But if p_b is non-zero in the calculation, it affects C_a.
        # If a bid fill is impossible, it shouldn't affect the ask decision?
        # The ask decision depends on the probability of *not* having a bid fill?
        # If bid fill is impossible, p_b=0.
        # Then C_a = delta_minus * (1 - 0) - 0 * delta_plus = delta_minus.
        # This seems correct.
        # So we should force p_b=0 at q=Q and p_a=0 at q=-Q for the calculation of the other side's C?
        # Let's modify p_b and p_a arrays after calculation.
        
        p_b_effective = p_b.copy()
        p_a_effective = p_a.copy()
        p_b_effective[-1] = 0.0  # No bid fill at max inventory
        p_a_effective[0] = 0.0   # No ask fill at min inventory
        
        # Recalculate probabilities with effective p's
        prob_q_plus = p_b_effective * (1.0 - p_a_effective)
        prob_q_minus = (1.0 - p_b_effective) * p_a_effective
        # Boundaries already handled by p_b_eff/p_a_eff being 0, but explicit 0 is safer
        prob_q_plus[-1] = 0.0
        prob_q_minus[0] = 0.0
        
        # Reward
        # If p_b_eff=0, reward from bid is 0.
        reward = p_b_effective * b + p_a_effective * a
        
        # Expectation
        # V_next_plus and V_next_minus are precomputed shifts.
        # We need to be careful at boundaries.
        # At q=Q, prob_q_plus=0, so V_next_plus value doesn't matter.
        # At q=-Q, prob_q_minus=0, so V_next_minus value doesn't matter.
        
        exp_val = prob_q_plus * V_next_plus + prob_q_minus * V_next_minus
        
        # The "stay" probability term
        prob_q = 1.0 - prob_q_plus - prob_q_minus
        exp_val += prob_q * V_next
        
        V_curr = reward - risk + exp_val
        
        V_next = V_curr
        # Update shifts for next iteration
        V_next_plus[:-1] = V_next[1:]
        V_next_minus[1:] = V_next[:-1]

    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    # Determine step index
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    
    with _TABLE_LOCK:
        if key not in _TABLE_CACHE:
            # Precompute
            start_time = time.time()
            opt_bid, opt_ask = _precompute_regime(
                params['sigma'], params['A'], params['k'], params['gamma'], 
                params['T'], params['steps'], params['Q']
            )
            _TABLE_CACHE[key] = (opt_bid, opt_ask)
            
            # Time guard (safety)
            if time.time() - start_time > 290:
                 depth = 1.0 / params["k"]
                 return np.full(len(inventory), depth), np.full(len(inventory), depth)

    opt_bid, opt_ask = _TABLE_CACHE[key]
    
    q_indices = np.asarray(inventory) + params['Q']
    
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    # Ensure valid outputs: no NaN, no -inf, no values below -3/k
    min_allowed = -3.0 / params['k']
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf) | (bid_depths < min_allowed), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf) | (ask_depths < min_allowed), np.inf, ask_depths)
    
    return bid_depths, ask_depths
