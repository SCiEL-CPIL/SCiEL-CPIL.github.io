"""Market making with inventory risk - optimized policy via value iteration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 325.0

# Cache for precomputed optimal depths per regime
_policy_cache = {}


def _compute_policy(sigma, A, k, gamma, T, steps, Q, time_budget):
    """
    Compute optimal bid and ask depths for each (time_step, inventory) pair
    using backward dynamic programming with analytical depth optimization.
    
    Returns:
        bid_depths: array of shape (steps, 2*Q+1)
        ask_depths: array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    n_states = 2 * Q + 1
    
    # Value function: V[time_step, inventory_index]
    # We'll store V for the next time step and compute current
    V_next = np.zeros(n_states)
    
    # Store optimal depths for each (time, inventory)
    bid_depths = np.full((steps, n_states), np.inf)
    ask_depths = np.full((steps, n_states), np.inf)
    
    # For each time step, going backward from T to 0
    for t in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _start_time > time_budget:
            break
        
        # For each inventory level
        for q_idx in range(n_states):
            q = q_idx - Q  # inventory level from -Q to Q
            
            # Boundary conditions: no bid at q=Q, no ask at q=-Q
            if q == Q:
                bid_depths[t, q_idx] = np.inf
                # Only optimize ask depth
                delta_V = V_next[q_idx - 1] - V_next[q_idx]  # V(q-1) - V(q)
                # Maximize p(d) * (d + delta_V) where p(d) = 1 - exp(-A*exp(-k*d)*dt)
                # For small dt: p(d) ≈ A*exp(-k*d)*dt
                # f(d) ≈ A*dt*exp(-k*d)*(d + delta_V)
                # f'(d) = A*dt*exp(-k*d)*(1 - k*(d + delta_V)) = 0
                # d* = 1/k - delta_V
                d_opt = 1.0/k - delta_V
                if d_opt < 0:
                    d_opt = 1e-8
                ask_depths[t, q_idx] = d_opt
            elif q == -Q:
                ask_depths[t, q_idx] = np.inf
                # Only optimize bid depth
                delta_V = V_next[q_idx + 1] - V_next[q_idx]  # V(q+1) - V(q)
                d_opt = 1.0/k - delta_V
                if d_opt < 0:
                    d_opt = 1e-8
                bid_depths[t, q_idx] = d_opt
            else:
                # Both sides can be quoted
                # delta_V_bid = V(q+1) - V(q)
                # delta_V_ask = V(q-1) - V(q)
                delta_V_bid = V_next[q_idx + 1] - V_next[q_idx]
                delta_V_ask = V_next[q_idx - 1] - V_next[q_idx]
                
                # Optimize bid depth: maximize p_b * (b + delta_V_bid)
                # p_b = 1 - exp(-A*exp(-k*b)*dt)
                # For small dt approximation: p_b ≈ A*dt*exp(-k*b)
                # f(b) ≈ A*dt*exp(-k*b)*(b + delta_V_bid)
                # f'(b) = A*dt*exp(-k*b)*(1 - k*(b + delta_V_bid)) = 0
                # b* = 1/k - delta_V_bid
                b_opt = 1.0/k - delta_V_bid
                if b_opt < 0:
                    b_opt = 1e-8
                bid_depths[t, q_idx] = b_opt
                
                # Optimize ask depth: maximize p_a * (a + delta_V_ask)
                a_opt = 1.0/k - delta_V_ask
                if a_opt < 0:
                    a_opt = 1e-8
                ask_depths[t, q_idx] = a_opt
            
            # Compute V[t, q] using the optimal depths
            b = bid_depths[t, q_idx]
            a = ask_depths[t, q_idx]
            
            # Fill probabilities
            if np.isinf(b):
                p_b = 0.0
            else:
                p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
            
            if np.isinf(a):
                p_a = 0.0
            else:
                p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
            
            # Expected value
            # Cases:
            # 1. No fills: prob (1-p_b)(1-p_a), earn 0, go to V(q)
            # 2. Bid only: prob p_b(1-p_a), earn b, go to V(q+1)
            # 3. Ask only: prob (1-p_b)p_a, earn a, go to V(q-1)
            # 4. Both: prob p_b*p_a, earn b+a, go to V(q) (net inventory change 0)
            
            V_q = V_next[q_idx]
            V_q_plus = V_next[min(q_idx + 1, n_states - 1)]
            V_q_minus = V_next[max(q_idx - 1, 0)]
            
            # If q=Q, q+1 is invalid; if q=-Q, q-1 is invalid
            if q == Q:
                V_q_plus = V_q  # Can't go above Q
                p_b = 0.0
            elif q == -Q:
                V_q_minus = V_q  # Can't go below -Q
                p_a = 0.0
            
            # Inventory cost
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            V_curr = (p_b * (1 - p_a) * (b + V_q_plus) +
                      (1 - p_b) * p_a * (a + V_q_minus) +
                      p_b * p_a * (b + a + V_q) +
                      (1 - p_b) * (1 - p_a) * V_q) - cost
            
            # Store for next iteration
            V_curr_arr = V_next  # We'll create a new array
            V_next[q_idx] = V_curr
        
        # After processing all states for this time step,
        # we need to be careful: we modified V_next in place, which is wrong.
        # Let me restructure this.
    
    # The above has a bug - we're modifying V_next while still using it.
    # Let me redo this properly.
    
    return bid_depths, ask_depths


def _compute_policy_correct(sigma, A, k, gamma, T, steps, Q, time_budget):
    """
    Correct backward DP with separate arrays for current and next value functions.
    """
    dt = T / steps
    n_states = 2 * Q + 1
    
    # V_next will hold V_{t+1}, V_curr will hold V_t
    V_next = np.zeros(n_states)
    
    # Store optimal depths for each (time, inventory)
    bid_depths = np.full((steps, n_states), np.inf)
    ask_depths = np.full((steps, n_states), np.inf)
    
    for t in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _start_time > time_budget:
            # Fill remaining with a simple heuristic
            for t_rem in range(t, -1, -1):
                for q_idx in range(n_states):
                    q = q_idx - Q
                    base = 1.0/k
                    if q == Q:
                        bid_depths[t_rem, q_idx] = np.inf
                        ask_depths[t_rem, q_idx] = base
                    elif q == -Q:
                        ask_depths[t_rem, q_idx] = np.inf
                        bid_depths[t_rem, q_idx] = base
                    else:
                        # Simple heuristic based on inventory
                        bid_depths[t_rem, q_idx] = max(1e-8, base * (1 - 0.3 * q / Q))
                        ask_depths[t_rem, q_idx] = max(1e-8, base * (1 + 0.3 * q / Q))
            return bid_depths, ask_depths
        
        V_curr = np.zeros(n_states)
        
        for q_idx in range(n_states):
            q = q_idx - Q  # inventory level from -Q to Q
            
            # Get value differences
            if q < Q:
                delta_V_bid = V_next[q_idx + 1] - V_next[q_idx]
            else:
                delta_V_bid = 0.0
            
            if q > -Q:
                delta_V_ask = V_next[q_idx - 1] - V_next[q_idx]
            else:
                delta_V_ask = 0.0
            
            # Optimize bid depth
            if q == Q:
                b_opt = np.inf
                p_b = 0.0
            else:
                b_opt = 1.0/k - delta_V_bid
                if b_opt < 1e-8:
                    b_opt = 1e-8
                p_b = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
            
            # Optimize ask depth
            if q == -Q:
                a_opt = np.inf
                p_a = 0.0
            else:
                a_opt = 1.0/k - delta_V_ask
                if a_opt < 1e-8:
                    a_opt = 1e-8
                p_a = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
            
            bid_depths[t, q_idx] = b_opt
            ask_depths[t, q_idx] = a_opt
            
            # Compute expected value
            V_q = V_next[q_idx]
            
            if q == Q:
                V_q_plus = V_q
                p_b = 0.0
                b_val = 0.0
            else:
                V_q_plus = V_next[q_idx + 1]
                b_val = b_opt if np.isfinite(b_opt) else 0.0
            
            if q == -Q:
                V_q_minus = V_q
                p_a = 0.0
                a_val = 0.0
            else:
                V_q_minus = V_next[q_idx - 1]
                a_val = a_opt if np.isfinite(a_opt) else 0.0
            
            # Inventory cost
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Expected value from fills
            # No fill: (1-p_b)(1-p_a) * V_q
            # Bid only: p_b(1-p_a) * (b + V_q_plus)
            # Ask only: (1-p_b)p_a * (a + V_q_minus)
            # Both: p_b*p_a * (b + a + V_q)
            
            V_curr[q_idx] = ((1 - p_b) * (1 - p_a) * V_q +
                             p_b * (1 - p_a) * (b_val + V_q_plus) +
                             (1 - p_b) * p_a * (a_val + V_q_minus) +
                             p_b * p_a * (b_val + a_val + V_q)) - cost
        
        V_next = V_curr
    
    return bid_depths, ask_depths


def _get_policy(params):
    """Get or compute the optimal policy for a given set of parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key in _policy_cache:
        return _policy_cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Time budget for computation
    time_budget = _TIME_LIMIT - (time.time() - _start_time) - 5.0  # 5s safety margin
    if time_budget < 1.0:
        time_budget = 1.0
    
    bid_depths, ask_depths = _compute_policy_correct(sigma, A, k, gamma, T, steps, Q, time_budget)
    
    policy = {'bid_depths': bid_depths, 'ask_depths': ask_depths, 'Q': Q, 'steps': steps}
    _policy_cache[key] = policy
    return policy


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    Q = params['Q']
    steps = params['steps']
    
    inv = np.asarray(inventory, dtype=np.int32)
    n_paths = len(inv)
    
    # Get policy
    policy = _get_policy(params)
    bid_depths_table = policy['bid_depths']
    ask_depths_table = policy['ask_depths']
    
    # Determine current time step
    dt = params['T'] / steps
    t_step = int(round(t / dt))
    t_step = np.clip(t_step, 0, steps - 1)
    
    # Convert inventory to index: q -> q + Q
    inv_idx = inv + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_depths_table[t_step, inv_idx]
    ask_depths = ask_depths_table[t_step, inv_idx]
    
    # Ensure valid values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
