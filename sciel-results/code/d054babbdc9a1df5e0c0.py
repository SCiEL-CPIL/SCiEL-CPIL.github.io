"""
Optimal market making with inventory risk via analytic HJB approximation.

This approach derives closed-form optimal bid and ask depths using the
continuous-time HJB equation for the market-making problem. Rather than
discretizing the depth space and doing backward induction, we solve the
first-order conditions analytically.

Key derivation:
- The HJB equation for value function V(t, q) where t is time remaining:
  -dV/dt = max_{b,a} [ (1-exp(-A*exp(-k*b)*dt))*b + (1-exp(-A*exp(-k*a)*dt))*a
                        - (gamma/2)*sigma^2*q^2*dt
                        + expected future value ]

- For small dt, the fill probability approximates A*exp(-k*depth)*dt
- The optimal depth balances marginal revenue (fill probability) against
  marginal inventory risk cost

- First-order condition for bid depth b:
  d/dt [fill_prob(b) * b - lambda * fill_prob(b)] = 0
  where lambda is the shadow price of inventory

- This gives: b* = (1/k) * log(1 + k*A*dt / (lambda + 1))
  Similarly for ask: a* = (1/k) * log(1 + k*A*dt / (lambda + 1))

- The shadow price lambda depends on inventory: lambda = gamma * sigma^2 * q

We pre-compute lookup tables for efficiency at runtime.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for pre-computed policies
_policy_cache = {}
_DEADLINE = None


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 315  # 15s safety margin before 330s kill
    return _DEADLINE


def _solve_analytic_policy(sigma, A, k, gamma, T, steps, Q, deadline):
    """
    Solve the analytic HJB approximation for optimal bid and ask depths.
    
    Returns a tuple (bid_table, ask_table) where:
      bid_table[n, q_idx] = optimal bid depth at step n, inventory q
      ask_table[n, q_idx] = optimal ask depth at step n, inventory q
    q_idx = q + Q, so q_idx in [0, 2Q]
    """
    dt = T / steps
    n_levels = 2 * Q + 1  # number of inventory levels
    
    # Pre-compute for all time steps and inventory levels
    bid_table = np.zeros((steps, n_levels))
    ask_table = np.zeros((steps, n_levels))
    
    # Inventory values
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)  # shape (n_levels,)
    
    # Time-remaining values (for each step n, time remaining = (steps - n) * dt)
    # At step n, time remaining = (steps - n) * dt
    
    # For each time step, compute optimal depths for all inventory levels
    for n in range(steps):
        if time.time() > deadline:
            # Time's up, return what we have (will be partial, zeros for rest)
            break
        
        time_remaining = (steps - n) * dt
        
        # Shadow price of inventory: lambda = gamma * sigma^2 * q
        # This represents the cost of taking on additional inventory risk
        lambda_vals = gamma * sigma**2 * q_vals  # shape (n_levels,)
        
        # For bid depth:
        # Optimal bid balances fill revenue against inventory cost
        # When inventory is positive (long), we want to reduce it -> lower bid (wider spread)
        # When inventory is negative (short), we want to increase it -> higher bid (tighter spread)
        
        # The effective "risk premium" for bid:
        # If q > 0 (long), bid fills increase inventory -> penalize
        # If q < 0 (short), bid fills decrease inventory -> reward
        
        # For bid:
        # Marginal value of bid fill = depth - gamma*sigma^2*(q+1)*dt (new inventory cost)
        # Fill probability = A * exp(-k * bid_depth) * dt
        # 
        # Optimize: max_{b} A * exp(-k*b) * dt * (b - gamma*sigma^2*(q+1)*dt)
        # 
        # Let x = k*b, so b = x/k
        # max_{x} A * exp(-x) * dt * (x/k - gamma*sigma^2*(q+1)*dt)
        # 
        # d/dx: A * dt * [exp(-x) * 1/k + exp(-x) * (-1) * (x/k - c)] = 0
        # where c = gamma*sigma^2*(q+1)*dt
        # 
        # 1/k - (x/k - c) = 0
        # x/k = c + 1/k
        # x = c*k + 1
        # b = (c*k + 1) / k = c + 1/k
        # 
        # But this is the unconstrained optimum. We need b > 0.
        # Also, the fill probability decreases exponentially with b, so
        # the effective optimum accounts for the tradeoff.
        
        # More carefully, the expected value from quoting bid at depth b:
        # E[V] = (1 - exp(-A*exp(-k*b)*dt)) * (b - gamma*sigma^2*(q+1)*dt)
        #      + exp(-A*exp(-k*b)*dt) * (-gamma*sigma^2*q*dt)
        # 
        # For small dt, 1 - exp(-x) ≈ x, so:
        # E[V] ≈ A*exp(-k*b)*dt * b - A*exp(-k*b)*dt * gamma*sigma^2*(q+1)*dt
        #        - gamma*sigma^2*q*dt
        # 
        # Ignoring the constant -gamma*sigma^2*q*dt, maximize:
        # f(b) = A*exp(-k*b)*dt * (b - gamma*sigma^2*(q+1)*dt)
        # 
        # Let c = gamma*sigma^2*(q+1)*dt
        # f(b) = A*dt * exp(-k*b) * (b - c)
        # 
        # f'(b) = A*dt * [-k*exp(-k*b)*(b-c) + exp(-k*b)*1] = 0
        # exp(-k*b) * [1 - k*(b-c)] = 0
        # 1 - k*b + k*c = 0
        # b = (1 + k*c) / k = 1/k + c
        # b = 1/k + gamma*sigma^2*(q+1)*dt
        # 
        # This is positive when 1/k > -gamma*sigma^2*(q+1)*dt
        # For q+1 >= 0 (q >= -1), this is always positive.
        # For q < -1, we need to check.
        
        # Similarly for ask:
        # E[V] ≈ A*exp(-k*a)*dt * (a - gamma*sigma^2*(q-1)*dt)
        # a = 1/k + gamma*sigma^2*(q-1)*dt
        
        # Bid depth: b = 1/k + gamma*sigma^2*(q+1)*dt
        # Ask depth: a = 1/k + gamma*sigma^2*(q-1)*dt
        
        bid_depths = 1.0/k + gamma * sigma**2 * (q_vals + 1.0) * dt
        ask_depths = 1.0/k + gamma * sigma**2 * (q_vals - 1.0) * dt
        
        # Ensure depths are positive
        bid_depths = np.maximum(bid_depths, 1e-6)
        ask_depths = np.maximum(ask_depths, 1e-6)
        
        # Boundary conditions: no bid at q=Q, no ask at q=-Q
        # At q = Q (index 2Q), set bid to inf
        # At q = -Q (index 0), set ask to inf
        if Q > 0:
            bid_depths[2 * Q] = np.inf  # q = Q
            ask_depths[0] = np.inf  # q = -Q
        
        bid_table[n] = bid_depths
        ask_table[n] = ask_depths
    
    return bid_table, ask_table


def _build_policy(params, deadline):
    """Build or retrieve cached policy for given params."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _policy_cache:
        return _policy_cache[key]
    
    bid_table, ask_table = _solve_analytic_policy(sigma, A, k, gamma, T, steps, Q, deadline)
    
    _policy_cache[key] = (bid_table, ask_table)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) arrays of shape (4000,).
    Each value is the distance below/above mid to quote.
    np.inf means no quote.
    """
    deadline = _get_deadline()
    
    n_paths = len(inventory)
    
    # Get or build policy
    try:
        bid_table, ask_table = _build_policy(params, deadline)
    except Exception:
        # Fallback to seed policy
        depth = np.full(n_paths, 1.0 / params["k"])
        return depth, depth.copy()
    
    # Determine current time step
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = min(n, params['steps'] - 1)
    n = max(n, 0)
    
    Q = params['Q']
    
    # Map inventory to index
    q_idx = np.clip(inventory, -Q, Q) + Q
    q_idx = np.array(q_idx, dtype=int)
    
    # Look up optimal depths
    bid_depths = bid_table[n, q_idx].astype(np.float64)
    ask_depths = ask_table[n, q_idx].astype(np.float64)
    
    # Ensure validity: depths must be > -3/k, not -inf, not NaN
    min_valid = -3.0 / params['k']
    
    # Replace invalid values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_valid), ask_depths, np.inf)
    
    # Boundary conditions: no bid fill at q=Q, no ask fill at q=-Q
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
