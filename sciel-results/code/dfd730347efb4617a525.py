"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for pre-computed policy parameters
# Key: (sigma, A, k, gamma, Q, steps) -> pre-computed arrays
_policy_cache = {}
_start_time = None


def _precompute_policy(sigma, A, k, gamma, Q, steps, T):
    """
    Pre-compute the optimal quoting depths using dynamic programming.
    
    The state is (step, inventory). We work backwards from step = steps to 0.
    At each step, given inventory q, we choose bid_depth and ask_depth to maximize
    the expected value.
    
    The value function V[n, q] represents the maximum expected reward from step n
    to the end, given inventory q at the start of step n.
    
    The reward at step n with inventory q (before fills) is:
      - (gamma/2) * sigma^2 * q^2 * dt  (holding cost)
    
    Plus the expected value of the next state after fills.
    
    For a given bid_depth b and ask_depth a:
      - Bid hit probability: p_b = 1 - exp(-A * exp(-k * b) * dt)
      - Ask hit probability: p_a = 1 - exp(-A * exp(-k * a) * dt)
      - If bid hits: inventory decreases by 1, we earn b
      - If ask hits: inventory increases by 1, we earn a
      - Both can hit independently
    
    So the expected value of choosing (b, a) at state (n, q) is:
      (gamma/2)*sigma^2*q^2*dt  [this is a cost, so negative reward]
      + p_b * (b + V[n+1, q-1])  [if bid hits]
      + p_a * (a + V[n+1, q+1])  [if ask hits]
      + (1-p_b)*(1-p_a) * V[n+1, q]  [no hits]
    
    Wait, let me re-read the objective: "depths earned minus sum over steps of (gamma/2) sigma^2 q^2 dt"
    
    So the total objective is: sum of depths earned - sum of holding costs.
    
    Let me redefine: V[n, q] = max expected (future depths earned - future holding costs) from step n onwards,
    given inventory q at start of step n.
    
    V[steps, q] = 0 for all q (no terminal charge)
    
    V[n, q] = max_{b, a >= 0} {
        -(gamma/2) * sigma^2 * q^2 * dt  [holding cost at this step]
        + p_b * (b + V[n+1, q-1])  [bid hit: earn b, inventory becomes q-1]
        + p_a * (a + V[n+1, q+1])  [ask hit: earn a, inventory becomes q+1]
        + (1-p_b)*(1-p_a) * V[n+1, q]  [no hits]
    }
    
    Subject to: if q == Q, bid_depth is not allowed (no bid fill at q=Q)
                if q == -Q, ask_depth is not allowed (no ask fill at q=-Q)
    
    Actually, re-reading: "no bid fill at q = Q, no ask fill at q = -Q"
    This means when inventory is at the maximum Q, we cannot have a bid fill (which would decrease inventory).
    When inventory is at the minimum -Q, we cannot have an ask fill (which would increase inventory).
    
    So at q = Q: p_b = 0 (no bid hits possible)
    At q = -Q: p_a = 0 (no ask hits possible)
    
    For the optimization over b and a, note that b and a only appear in:
      p_b = 1 - exp(-A * exp(-k*b) * dt)
      p_a = 1 - exp(-A * exp(-k*a) * dt)
    
    And in the terms: p_b * b and p_a * a.
    
    Let's define for bid side: f(b) = p_b * (b + V[n+1, q-1]) + (1-p_b) * V[n+1, q]
    Wait, that's not quite right because p_b and p_a are independent.
    
    Let me expand more carefully:
    
    Expected value = -(gamma/2)*sigma^2*q^2*dt
      + p_b * (b + V[n+1, q-1]) * (1 - p_a)  [bid hits, ask doesn't]
      + p_a * (a + V[n+1, q+1]) * (1 - p_b)  [ask hits, bid doesn't]
      + p_b * p_a * (b + a + V[n+1, q])  [both hit, inventory unchanged]
      + (1-p_b)*(1-p_a) * V[n+1, q]  [neither hits]
    
    Wait, if both hit: bid hit decreases inventory by 1, ask hit increases by 1, net change is 0.
    So inventory stays at q. And we earn b + a.
    
    Let me verify: "A fill earns its depth and moves inventory one unit"
    Bid fill: inventory -= 1, earn bid_depth
    Ask fill: inventory += 1, earn ask_depth
    
    If both fill: inventory -= 1 + 1 = 0 net change. Earn b + a.
    
    So:
    EV = -(gamma/2)*sigma^2*q^2*dt
      + p_b*(1-p_a) * (b + V[n+1, q-1])
      + p_a*(1-p_b) * (a + V[n+1, q+1])
      + p_b*p_a * (b + a + V[n+1, q])
      + (1-p_b)*(1-p_a) * V[n+1, q]
    
    This separates into bid and ask parts:
    
    EV = -(gamma/2)*sigma^2*q^2*dt
      + p_b * [ (1-p_a)*(b + V[n+1, q-1]) + p_a*(b + a + V[n+1, q]) ]
      + p_a * [ (1-p_b)*(a + V[n+1, q+1]) + p_b*(b + a + V[n+1, q]) ]
      + (1-p_b)*(1-p_a) * V[n+1, q]
    
    This is still coupled. Let me just do a grid search over b and a for each state.
    
    Given the constraints and the 300s budget, I need to be efficient.
    Let me pre-compute on a grid of possible depths and inventory levels.
    """
    dt = T / steps
    Q_range = range(-Q, Q + 1)
    num_q = 2 * Q + 1
    
    # V[n, q] for n = 0..steps, q = -Q..Q
    # We'll store V as a 2D array: V[n, q_idx] where q_idx = q + Q
    V = np.zeros((steps + 1, num_q))
    
    # Optimal depths: opt_b[n, q_idx], opt_a[n, q_idx]
    opt_b = np.zeros((steps, num_q))
    opt_a = np.zeros((steps, num_q))
    
    # Grid of possible depths to search over
    # Depths should be in a reasonable range. Let's say 0 to 10/k or so.
    # Actually, the optimal depth is likely around 1/k. Let's use a fine grid.
    max_depth = 10.0 / k
    num_depths = 200
    depth_grid = np.linspace(0.0, max_depth, num_depths)
    
    for n in range(steps - 1, -1, -1):
        for qi, q in enumerate(Q_range):
            # Holding cost
            hold_cost = -0.5 * gamma * sigma**2 * q**2 * dt
            
            # Determine if bid/ask fills are possible
            bid_allowed = (q < Q)
            ask_allowed = (q > -Q)
            
            best_val = -np.inf
            best_b = 0.0
            best_a = 0.0
            
            # If at boundary, one side is fixed
            if bid_allowed and ask_allowed:
                # Search over both
                for bi, b in enumerate(depth_grid):
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    for ai, a in enumerate(depth_grid):
                        p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                        
                        ev = hold_cost
                        ev += p_b * (1 - p_a) * (b + V[n+1, qi-1])
                        ev += p_a * (1 - p_b) * (a + V[n+1, qi+1])
                        ev += p_b * p_a * (b + a + V[n+1, qi])
                        ev += (1 - p_b) * (1 - p_a) * V[n+1, qi]
                        
                        if ev > best_val:
                            best_val = ev
                            best_b = b
                            best_a = a
            elif bid_allowed:
                # Only bid can fill
                for bi, b in enumerate(depth_grid):
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    
                    ev = hold_cost
                    ev += p_b * (b + V[n+1, qi-1])
                    ev += (1 - p_b) * V[n+1, qi]
                    
                    if ev > best_val:
                        best_val = ev
                        best_b = b
                        best_a = 0.0
            elif ask_allowed:
                # Only ask can fill
                for ai, a in enumerate(depth_grid):
                    p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    
                    ev = hold_cost
                    ev += p_a * (a + V[n+1, qi+1])
                    ev += (1 - p_a) * V[n+1, qi]
                    
                    if ev > best_val:
                        best_val = ev
                        best_b = 0.0
                        best_a = a
            else:
                # Neither can fill (shouldn't happen since Q >= 1)
                best_val = hold_cost + V[n+1, qi]
                best_b = 0.0
                best_a = 0.0
            
            V[n, qi] = best_val
            opt_b[n, qi] = best_b
            opt_a[n, qi] = best_a
    
    return opt_b, opt_a


def _get_policy(sigma, A, k, gamma, Q, steps, T):
    """Get or pre-compute the policy for given parameters."""
    key = (sigma, A, k, gamma, Q, steps, T)
    if key not in _policy_cache:
        _policy_cache[key] = _precompute_policy(sigma, A, k, gamma, Q, steps, T)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """Quote both sides using pre-computed optimal policy based on inventory."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline
    if time.time() - _start_time > 280:
        # Return a safe default
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    try:
        opt_b, opt_a = _get_policy(sigma, A, k, gamma, Q, steps, T)
    except Exception:
        # Fallback to seed behavior
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    # Current step
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # inventory is an array of ints in [-Q, Q]
    # Map to index: q_idx = q + Q
    q_idx = inventory + Q
    
    # Clip to valid range
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Get optimal depths
    bid_depths = opt_b[n, q_idx].astype(np.float64)
    ask_depths = opt_a[n, q_idx].astype(np.float64)
    
    # Ensure no negative values
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # Check for invalid values
    min_valid = -3.0 / k
    bid_depths = np.where(bid_depths < min_valid, 0.0, bid_depths)
    ask_depths = np.where(ask_depths < min_valid, 0.0, ask_depths)
    bid_depths = np.where(np.isnan(bid_depths), 0.0, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), 0.0, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
