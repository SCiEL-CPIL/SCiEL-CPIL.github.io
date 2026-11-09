"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 270.0  # safety margin before 300s limit
_CACHE = {}
_SEARCH_DONE = False


def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the exact optimal policy via backward induction.
    Returns arrays f_b[q, s] and f_a[q, s] giving optimal bid and ask depths,
    where q is in [-Q, Q] and s is the step index (0 = last step, steps-1 = first step).
    """
    dt = T / steps
    n_q = 2 * Q + 1
    n_s = steps
    
    # f_b[i, s] = optimal bid depth at inventory q = i - Q, step s
    # f_a[i, s] = optimal ask depth at inventory q = i - Q, step s
    f_b = np.zeros((n_q, n_s), dtype=np.float64)
    f_a = np.zeros((n_q, n_s), dtype=np.float64)
    
    # We do backward induction from s = steps-1 down to 0
    # V[q, s] = max value from step s onwards given inventory q before step s
    # At each step, we choose bid_depth b and ask_depth a (with constraints)
    # Then a bid fill happens with prob p_b = 1 - exp(-A * exp(-k * b) * dt)
    # And an ask fill happens with prob p_a = 1 - exp(-A * exp(-k * a) * dt)
    
    # For a finite state space, we can enumerate possible depths.
    # Since the objective is linear in depth (earn depth on fill) and the fill probability
    # depends on depth via exp(-k*depth), we can solve for the optimal depth analytically
    # or use a grid search.
    
    # Let's use a grid of candidate depths. The optimal depth should be in a reasonable range.
    # The seed uses 1/k. Let's try depths from 0.01 to maybe 5/k or so.
    
    # Actually, let's think about this more carefully.
    # The value function V[q, s] satisfies:
    # V[q, s] = max_{b, a} { 
    #     -gamma/2 * sigma^2 * q^2 * dt +
    #     p_b(b) * (b + V[q+1, s+1]) +
    #     p_a(a) * (-a + V[q-1, s+1]) +
    #     (1 - p_b(b)) * (1 - p_a(a)) * V[q, s+1]
    # }
    # where p_b(b) = 1 - exp(-A * exp(-k*b) * dt)
    
    # This is a 2D maximization. We can do it on a grid.
    
    # Grid of depths
    depth_min = 0.01
    depth_max = 5.0 / k
    n_depth = 200
    depths = np.linspace(depth_min, depth_max, n_depth)
    
    # Precompute fill probabilities for each depth
    # p_fill[d] = 1 - exp(-A * exp(-k * d) * dt)
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    # Terminal value: V[q, steps] = 0 (no terminal charge)
    V_next = np.zeros(n_q, dtype=np.float64)
    
    for s in range(n_s - 1, -1, -1):
        V_curr = np.zeros(n_q, dtype=np.float64)
        
        # Inventory cost for this step
        inv_cost = 0.5 * gamma * sigma * sigma * dt
        # q values: i - Q for i in 0..n_q-1
        q_vals = np.arange(n_q) - Q
        
        for i in range(n_q):
            q = q_vals[i]
            
            # If q == Q, no bid fill possible; if q == -Q, no ask fill possible
            if q == Q:
                # Only ask side matters
                best_val = -np.inf
                best_a = depths[0]
                for di, a in enumerate(depths):
                    pa = p_fill[di]
                    val = -inv_cost * q * q + pa * (-a + V_next[i - 1]) + (1 - pa) * V_next[i]
                    if val > best_val:
                        best_val = val
                        best_a = a
                V_curr[i] = best_val
                f_a[i, s] = best_a
                f_b[i, s] = np.inf  # no bid quote
            elif q == -Q:
                # Only bid side matters
                best_val = -np.inf
                best_b = depths[0]
                for di, b in enumerate(depths):
                    pb = p_fill[di]
                    val = -inv_cost * q * q + pb * (b + V_next[i + 1]) + (1 - pb) * V_next[i]
                    if val > best_val:
                        best_val = val
                        best_b = b
                V_curr[i] = best_val
                f_b[i, s] = best_b
                f_a[i, s] = np.inf  # no ask quote
            else:
                # Both sides possible
                # We need to maximize over (b, a). This is a 2D grid.
                # For efficiency, we can fix one and optimize the other, but let's just do 2D.
                best_val = -np.inf
                best_b = depths[0]
                best_a = depths[0]
                
                for di_b, b in enumerate(depths):
                    pb = p_fill[di_b]
                    # For this b, maximize over a
                    # val = -inv_cost*q^2 + pb*(b + V_next[i+1]) + pa*(-a + V_next[i-1]) + (1-pb)*(1-pa)*V_next[i]
                    # = -inv_cost*q^2 + pb*(b + V_next[i+1]) + pa*(V_next[i-1] - a) + (1-pb)*V_next[i] - (1-pb)*pa*V_next[i]
                    # = -inv_cost*q^2 + pb*(b + V_next[i+1]) + (1-pb)*V_next[i] + pa*(V_next[i-1] - a - (1-pb)*V_next[i])
                    # Wait, let me redo:
                    # val = -inv_cost*q^2 + pb*(b + V_next[i+1]) + pa*(-a + V_next[i-1]) + (1-pb)*(1-pa)*V_next[i]
                    # = -inv_cost*q^2 + pb*(b + V_next[i+1]) + pa*(-a + V_next[i-1]) + (1-pb)*V_next[i] - (1-pb)*pa*V_next[i]
                    # = -inv_cost*q^2 + pb*(b + V_next[i+1]) + (1-pb)*V_next[i] + pa*(-a + V_next[i-1] - (1-pb)*V_next[i])
                    
                    c1 = -inv_cost * q * q + pb * (b + V_next[i + 1]) + (1 - pb) * V_next[i]
                    c2 = V_next[i - 1] - (1 - pb) * V_next[i]
                    
                    # Maximize over pa: c1 + pa * (c2 - a)
                    # But pa depends on a: pa = 1 - exp(-A * exp(-k*a) * dt)
                    # So we need to maximize c1 + (1 - exp(-A*exp(-k*a)*dt)) * (c2 - a)
                    # over a in depths
                    
                    # For each a, compute val
                    # To speed up, note that for fixed b, we can vectorize over a
                    vals_a = c1 + p_fill * (c2 - depths)
                    best_di_a = np.argmax(vals_a)
                    val_a = vals_a[best_di_a]
                    
                    if val_a > best_val:
                        best_val = val_a
                        best_b = b
                        best_a = depths[best_di_a]
                
                V_curr[i] = best_val
                f_b[i, s] = best_b
                f_a[i, s] = best_a
        
        V_next = V_curr
    
    # f_b and f_a are indexed by [inventory_index, step]
    # step s in our loop: s = steps-1 is the last step, s = 0 is the first step
    # But the evaluator calls quote at step n = 0..999 with t = n*dt
    # So step index in the evaluator is n, and our backward induction used s where s=steps-1 corresponds to n=0?
    # Let me re-check: in the loop, s goes from n_s-1 down to 0.
    # V_next starts as V at step steps (terminal).
    # When s = n_s-1 = steps-1, we compute V for step steps-1, which is the last step.
    # So f_b[i, s] for s=steps-1 is the optimal depth at the last step.
    # The evaluator calls with step n=0 first, which corresponds to our s=steps-1.
    # So we need to reverse the step axis: f_b[i, n] = f_b_computed[i, steps-1-n]
    
    f_b = f_b[:, ::-1]
    f_a = f_a[:, ::-1]
    
    return f_b, f_a


def _get_policy(params):
    """Get or compute the optimal policy for the given parameters."""
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    if key not in _CACHE:
        if time.time() > _DEADLINE:
            # Fallback to seed if we're running out of time
            return None
        _CACHE[key] = _precompute_regime(
            params["sigma"], params["A"], params["k"], params["gamma"],
            params["T"], params["steps"], params["Q"]
        )
    return _CACHE[key]


def quote(t, inventory, mid, params):
    """Quote using the precomputed optimal policy."""
    global _SEARCH_DONE
    
    n = len(inventory)
    Q = params["Q"]
    
    policy = _get_policy(params)
    
    if policy is None:
        # Fallback: seed policy
        depth = np.full(n, 1.0 / params["k"])
        return depth, depth.copy()
    
    f_b, f_a = policy
    
    # inventory is in [-Q, Q], map to index [0, 2Q]
    idx = inventory + Q  # now in [0, 2Q]
    
    # t = n * dt, so step index n = t / dt
    dt = params["T"] / params["steps"]
    step_idx = int(round(t / dt))
    
    # Clamp step_idx to valid range
    step_idx = max(0, min(step_idx, params["steps"] - 1))
    
    bid_depths = f_b[idx, step_idx]
    ask_depths = f_a[idx, step_idx]
    
    # Handle boundary conditions: no bid at q=Q, no ask at q=-Q
    # f_b is already inf for q=Q, f_a is inf for q=-Q from the precomputation
    
    # Ensure values are within valid range
    min_depth = -3.0 / params["k"]
    bid_depths = np.where(bid_depths < min_depth, np.inf, bid_depths)
    ask_depths = np.where(ask_depths < min_depth, np.inf, ask_depths)
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
