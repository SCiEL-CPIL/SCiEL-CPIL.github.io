import numpy as np
import time

# Cache for precomputed DP tables per regime
_dp_cache = {}
_start_time = time.time()


def _precompute_dp(params):
    """Precompute optimal bid/ask depth tables for a given regime."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key in _dp_cache:
        return _dp_cache[key]

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    dt = T / steps
    n_inv = 2 * Q + 1
    # inventory index: i=0 -> q=-Q, i=n_inv-1 -> q=Q
    # q = i - Q

    # V[i] holds value at current time for inventory q = i - Q
    # We compute backwards from t=T to t=0
    # At t=T, V = 0 (no terminal charge)

    # We'll store depth tables for each time step
    # bid_depth_table[step, i] = optimal bid depth at time step, inventory index i
    # ask_depth_table[step, i] = optimal ask depth

    bid_depth_table = np.full((steps, n_inv), np.inf, dtype=np.float64)
    ask_depth_table = np.full((steps, n_inv), np.inf, dtype=np.float64)

    V = np.zeros(n_inv, dtype=np.float64)

    # Precompute e^{-k*d} for a range of depths? No, we solve analytically.
    # The Bellman equation:
    # V(q,t) = V(q,t+dt) + max over d_b, d_a of:
    #   A*dt * [ (1-exp(-k*d_b)) * (d_b + V(q-1,t+dt) - V(q,t+dt)) 
    #           + (1-exp(-k*d_a)) * (d_a + V(q+1,t+dt) - V(q,t+dt)) ]
    #   - (gamma/2)*sigma^2*q^2*dt
    #
    # Wait, the problem says: "bid hit with probability 1 - exp(-A exp(-k bid_depth) dt)"
    # So the probability is p_b = 1 - exp(-A * exp(-k * d_b) * dt)
    # For small A*exp(-k*d_b)*dt, p_b ≈ A*exp(-k*d_b)*dt
    # But let's use the exact expression.
    #
    # Expected reward from bid: p_b * (d_b + ΔV_b) where ΔV_b = V(q-1,t+dt) - V(q,t+dt)
    # Similarly for ask: p_a * (d_a + ΔV_a) where ΔV_a = V(q+1,t+dt) - V(q,t+dt)
    #
    # The two sides are independent, so the expected total is:
    # E[reward] = p_b*(d_b + ΔV_b) + p_a*(d_a + ΔV_a)
    #
    # We need to maximize over d_b >= 0 and d_a >= 0 (or d=inf means no quote)
    #
    # For each side independently, maximize:
    # f(d) = (1 - exp(-A*exp(-k*d)*dt)) * (d + C)
    # where C = ΔV (for bid) or ΔV (for ask)
    #
    # This is a 1D optimization. Let x = A*exp(-k*d)*dt, so exp(-k*d) = x/(A*dt)
    # d = -(1/k)*ln(x/(A*dt)) = (1/k)*ln(A*dt/x)
    # f as function of x: (1-exp(-x))*(d(x) + C)
    # This is still not trivial. Let's just do a numerical 1D optimization.
    #
    # Actually, for typical parameters, A*exp(-k*d)*dt is small, so we can use
    # the approximation p ≈ A*exp(-k*d)*dt. Then:
    # f(d) ≈ A*dt*exp(-k*d)*(d + C)
    # f'(d) = A*dt*[-k*exp(-k*d)*(d+C) + exp(-k*d)] = A*dt*exp(-k*d)*(1 - k*(d+C))
    # Setting to 0: 1 - k*(d+C) = 0 => d* = 1/k - C
    # If d* < 0, optimum is at d=0.
    # f(0) = A*dt*(0+C) = A*dt*C
    # f(d*) = A*dt*exp(-k*(1/k-C))*(1/k) = A*dt*exp(C-1)*(1/k)
    #
    # Let's use the exact formula but solve numerically with a few iterations.
    # Given the small dt, the approximation should be very good.

    for step in range(steps - 1, -1, -1):
        V_next = V.copy()
        for i in range(n_inv):
            q = i - Q
            # Inventory penalty
            inv_penalty = (gamma / 2.0) * sigma * sigma * q * q * dt

            if i == 0:
                # q = -Q, cannot ask (would go to -Q-1)
                # Only bid side
                C_b = V_next[i + 1] - V_next[i]  # V(q+1) - V(q) ... wait
                # q = i - Q. If we get a bid fill, inventory changes by +1 (we sell? or buy?)
                # "A fill earns its depth and moves inventory one unit"
                # Bid fill: we sell at bid, so inventory decreases? Or increases?
                # In market making, if we quote bid and get filled, we buy, so inventory increases.
                # Let's re-read: "inventory (4000,) ints in [-Q, Q]"
                # "bid_depths ... distances below mid" - we're buying at bid
                # "ask_depths ... distances above mid" - we're selling at ask
                # If bid is filled, we buy 1 unit, so inventory increases by 1.
                # If ask is filled, we sell 1 unit, so inventory decreases by 1.
                # So: bid fill -> q -> q+1, ask fill -> q -> q-1
                #
                # Wait, that seems backwards from typical convention. Let me re-read.
                # "bid_depths ... distances below and above mid to quote this step"
                # bid_depths are distances below mid (we're the buyer at bid)
                # ask_depths are distances above mid (we're the seller at ask)
                # If someone hits our bid, we buy, inventory +1
                # If someone lifts our ask, we sell, inventory -1
                #
                # So: bid fill: q -> q+1, ask fill: q -> q-1
                # No bid fill at q = Q (can't buy more if already at max)
                # No ask fill at q = -Q (can't sell more if already at min)
                #
                # This means:
                # C_b = V(q+1, t+dt) - V(q, t+dt) = V_next[i+1] - V_next[i] if bid fill
                # C_a = V(q-1, t+dt) - V(q, t+dt) = V_next[i-1] - V_next[i] if ask fill

                # For i=0, q=-Q: no ask fill possible
                C_b = V_next[i + 1] - V_next[i] if i + 1 < n_inv else 0.0
                best_val_b = _max_fill_value(C_b, A, k, dt)
                V[i] = V_next[i] + best_val_b - inv_penalty
                bid_depth_table[step, i] = _optimal_depth(C_b, A, k, dt)
                ask_depth_table[step, i] = np.inf
            elif i == n_inv - 1:
                # q = Q: no bid fill possible
                C_a = V_next[i - 1] - V_next[i] if i - 1 >= 0 else 0.0
                best_val_a = _max_fill_value(C_a, A, k, dt)
                V[i] = V_next[i] + best_val_a - inv_penalty
                bid_depth_table[step, i] = np.inf
                ask_depth_table[step, i] = _optimal_depth(C_a, A, k, dt)
            else:
                C_b = V_next[i + 1] - V_next[i]
                C_a = V_next[i - 1] - V_next[i]
                best_val_b = _max_fill_value(C_b, A, k, dt)
                best_val_a = _max_fill_value(C_a, A, k, dt)
                V[i] = V_next[i] + best_val_b + best_val_a - inv_penalty
                bid_depth_table[step, i] = _optimal_depth(C_b, A, k, dt)
                ask_depth_table[step, i] = _optimal_depth(C_a, A, k, dt)

    _dp_cache[key] = (bid_depth_table, ask_depth_table)
    return _dp_cache[key]


def _max_fill_value(C, A, k, dt):
    """Maximize (1 - exp(-A*exp(-k*d)*dt)) * (d + C) over d >= 0, with d=inf giving 0."""
    # If C is very negative, it's better not to quote (d=inf, value 0)
    # If C is very positive, quote aggressively
    # Use approximation for speed: f(d) ≈ A*dt*exp(-k*d)*(d+C)
    # Optimal d* = max(0, 1/k - C)
    d_star = max(0.0, 1.0 / k - C)
    # Compute exact value at d_star and at d=0
    if d_star == 0.0:
        x0 = A * np.exp(-k * 0.0) * dt  # = A*dt
        f0 = (1.0 - np.exp(-x0)) * (0.0 + C)
        # Check if not quoting is better
        return max(f0, 0.0)
    else:
        xd = A * np.exp(-k * d_star) * dt
        fd = (1.0 - np.exp(-xd)) * (d_star + C)
        x0 = A * dt
        f0 = (1.0 - np.exp(-x0)) * C
        return max(fd, f0, 0.0)


def _optimal_depth(C, A, k, dt):
    """Find optimal depth for a side with value difference C."""
    # If not quoting is optimal, return inf
    # Compare value at d=0, d=d_star, and d=inf(0)
    d_star = max(0.0, 1.0 / k - C)
    
    # Value at d=inf is 0
    # Value at d=0
    x0 = A * dt
    f0 = (1.0 - np.exp(-x0)) * C
    
    if d_star <= 0.0:
        # Optimum at boundary d=0
        if f0 <= 0.0:
            return np.inf
        return 0.0
    else:
        xd = A * np.exp(-k * d_star) * dt
        fd = (1.0 - np.exp(-xd)) * (d_star + C)
        if fd >= f0 and fd > 0.0:
            return d_star
        elif f0 > 0.0:
            return 0.0
        else:
            return np.inf


def quote(t, inventory, mid, params):
    """Quote based on precomputed optimal DP policy."""
    steps = params['steps']
    Q = params['Q']
    
    # Get current step index
    dt = params['T'] / steps
    step_idx = int(round(t / dt))
    step_idx = min(max(step_idx, 0), steps - 1)
    
    # Precompute if needed
    bid_table, ask_table = _precompute_dp(params)
    
    n_inv = 2 * Q + 1
    
    # Map inventory to index: q in [-Q, Q] -> i in [0, n_inv-1]
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, n_inv - 1)
    
    bid_depths = bid_table[step_idx, inv_idx].astype(np.float64)
    ask_depths = ask_table[step_idx, inv_idx].astype(np.float64)
    
    # Ensure shapes and valid values
    # Values below -3/k, -inf, or NaN are invalid
    min_valid = -3.0 / params['k']
    
    # Replace any invalid values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, np.inf)
    
    return bid_depths, ask_depths
