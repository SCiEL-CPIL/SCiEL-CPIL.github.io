import numpy as np
import time

# Cache for precomputed policies
_policy_cache = {}
_start_time = time.time()
_TIMEOUT_LIMIT = 290.0  # Leave 10s margin for the 300s limit

def _precompute_policy(params):
    """
    Precomputes the optimal bid and ask depths for all time steps and inventory levels.
    Returns:
        bid_policy: array of shape (steps, 2*Q+1)
        ask_policy: array of shape (steps, 2*Q+1)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    
    # Grid of possible depths
    # Optimal depth is around 1/k. 1/k is ~0.66 to 1.33.
    # Let's use 0.0 to 2.0 with 200 points.
    depths = np.linspace(0.0, 2.0, 200)
    
    # Precompute probability of fill for each depth
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_term = np.exp(-k * depths)
    p_fills = 1.0 - np.exp(-A * exp_term * dt)
    
    # Initialize Value function V for n = steps (terminal)
    # V is of shape (2*Q+1,)
    V = np.zeros(2 * Q + 1)
    
    # Policies
    bid_policy = np.zeros((steps, 2 * Q + 1))
    ask_policy = np.zeros((steps, 2 * Q + 1))
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time
        if time.time() - _start_time > _TIMEOUT_LIMIT:
            # Fallback to simple heuristic if time is running out
            # This is unlikely to happen given the speed, but safe.
            break
            
        # Delta V for bid (q -> q+1) and ask (q -> q-1)
        # q index i corresponds to q = i - Q
        # q+1 is index i+1
        # q-1 is index i-1
        
        # For all q, we want to maximize:
        # J_b(b) = p(b) * (b + V[q+1] - V[q])
        # J_a(a) = p(a) * (a + V[q-1] - V[q])
        
        # Boundary conditions:
        # If q == Q (index 2Q), no bid fill. V[q+1] is out of bounds.
        # We handle this by setting the objective to -inf or just ignoring.
        # Actually, if q==Q, we must set bid depth to inf.
        # In the optimization, we can just mask it.
        
        # Vectorized optimization for all q at once
        
        # Delta V_bid[i] = V[i+1] - V[i] for i < 2Q
        # Delta V_bid[2Q] is irrelevant (will be masked)
        dV_bid = np.zeros(2 * Q + 1)
        dV_bid[:2*Q] = V[1:] - V[:2*Q]
        
        # Delta V_ask[i] = V[i-1] - V[i] for i > 0
        # Delta V_ask[0] is irrelevant
        dV_ask = np.zeros(2 * Q + 1)
        dV_ask[1:] = V[:-1] - V[1:]
        
        # Objective for bid: shape (2Q+1, N_d)
        # J_b[i, j] = p_fills[j] * (depths[j] + dV_bid[i])
        J_b = p_fills[None, :] * (depths[None, :] + dV_bid[:, None])
        
        # Mask out invalid states
        # If q == Q (index 2Q), bid is not allowed.
        # We can set J_b[2Q, :] = -np.inf
        J_b[2*Q, :] = -np.inf
        
        # Find optimal depth index for each q
        best_bid_idx = np.argmax(J_b, axis=1)
        opt_bid_depths = depths[best_bid_idx]
        
        # Handle the case where all are -inf (shouldn't happen for q<Q)
        # For q=Q, argmax will return 0, but we will override to inf later.
        
        # Objective for ask: shape (2Q+1, N_d)
        J_a = p_fills[None, :] * (depths[None, :] + dV_ask[:, None])
        
        # If q == -Q (index 0), ask is not allowed.
        J_a[0, :] = -np.inf
        
        best_ask_idx = np.argmax(J_a, axis=1)
        opt_ask_depths = depths[best_ask_idx]
        
        # Update Value Function
        # V_new[q] = -gamma/2 * sigma^2 * q^2 * dt + max_J_b + max_J_a
        # Note: The max_J_b and max_J_a are the maximum *additional* value from the fill.
        # The base value V[q] is included in the dV calculation?
        # Let's re-verify the recursion.
        # V_n(q) = -Risk + E[ Reward + V_{n+1}(q') ]
        # E[ ... ] = V_{n+1}(q) + p_b(1-p_a)(b + V_{n+1}(q+1) - V_{n+1}(q)) + ...
        # Wait, my previous derivation:
        # E[V_next] = V[q] + p_b(1-p_a)dV_bid + (1-p_b)p_a dV_ask - p_b p_a (dV_bid + dV_ask)
        # If we ignore coupling:
        # E[V_next] \approx V[q] + p_b dV_bid + p_a dV_ask
        # Profit = p_b b + p_a a
        # So V_n(q) = -Risk + V[q] + max_b(p_b(b)(b+dV_bid)) + max_a(p_a(a)(a+dV_ask))
        
        max_J_b = np.max(J_b, axis=1)
        max_J_a = np.max(J_a, axis=1)
        
        # Risk term
        q_vals = np.arange(-Q, Q + 1)
        risk = -0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        V_new = V + risk + max_J_b + max_J_a
        
        # Store policy
        bid_policy[n, :] = opt_bid_depths
        ask_policy[n, :] = opt_ask_depths
        
        # Update V
        V = V_new
        
    # Set boundaries to inf in the stored policies
    # q=Q is index 2Q
    bid_policy[:, 2*Q] = np.inf
    # q=-Q is index 0
    ask_policy[:, 0] = np.inf
    
    return bid_policy, ask_policy

def quote(t, inventory, mid, params):
    """
    Returns bid_depths and ask_depths for the given state.
    """
    # Identify regime key
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        _policy_cache[key] = _precompute_policy(params)
    
    bid_policy, ask_policy = _policy_cache[key]
    
    steps = params['steps']
    Q = params['Q']
    T = params['T']
    dt = T / steps
    
    # Determine time step index
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
        
    # Get optimal depths for this time step
    bid_depths_all = bid_policy[n, :]
    ask_depths_all = ask_policy[n, :]
    
    # Map inventory to indices
    # inventory is array of ints in [-Q, Q]
    idx = inventory + Q
    
    # Clip indices to be safe
    idx = np.clip(idx, 0, 2*Q)
    
    bid_depths = bid_depths_all[idx]
    ask_depths = ask_depths_all[idx]
    
    # Ensure valid shapes and types
    bid_depths = np.asarray(bid_depths, dtype=float)
    ask_depths = np.asarray(ask_depths, dtype=float)
    
    return bid_depths, ask_depths
