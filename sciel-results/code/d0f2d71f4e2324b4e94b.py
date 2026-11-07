import numpy as np
import time

# Global cache for pre-computed policies
_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0 # Safety margin

def _solve_optimal_depth(C, alpha, k):
    """
    Finds optimal depth d >= 0 maximizing P(d)*(d + C)
    where P(d) = 1 - exp(-alpha * exp(-k*d))
    Uses Newton-Raphson on the derivative J'(d) = P'(d)(d+C) + P(d)
    """
    if alpha <= 0:
        return 0.0
    
    # If C is very large, we might want to maximize P, so d->0?
    # If C is very negative, we want P->0, so d->inf?
    # The function is unimodal.
    
    # Initial guess: 1/k is a reasonable scale
    d = 1.0 / k
    max_iter = 20
    tol = 1e-8
    
    for _ in range(max_iter):
        E = np.exp(-k * d)
        alpha_E = alpha * E
        P = 1.0 - np.exp(-alpha_E)
        # P' = k * alpha * E * exp(-alpha * E)
        P_prime = k * alpha_E * np.exp(-alpha_E)
        # P'' = k^2 * alpha * E * exp(-alpha * E) * (alpha * E - 1)
        P_double_prime = k * k * alpha_E * np.exp(-alpha_E) * (alpha_E - 1.0)
        
        J_prime = P_prime * (d + C) + P
        J_double_prime = P_double_prime * (d + C) + 2.0 * P_prime
        
        if abs(J_double_prime) < 1e-12:
            break
            
        step = J_prime / J_double_prime
        d_new = d - step
        
        # Constrain d to [0, 10/k]
        if d_new < 0:
            d_new = 0.0
        elif d_new > 10.0 / k:
            d_new = 10.0 / k
            
        if abs(d_new - d) < tol:
            d = d_new
            break
        d = d_new
        
    return d

def _compute_policy(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
        
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    alpha = A * dt
    
    n_states = 2 * Q + 1
    V = np.zeros((steps + 1, n_states))
    
    bid_depths = np.full((steps, n_states), np.inf)
    ask_depths = np.full((steps, n_states), np.inf)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 5.0:
            # Fallback: return current V and default depths
            for i in range(n, steps):
                bid_depths[i] = 1.0/k
                ask_depths[i] = 1.0/k
            _POLICY_CACHE[key] = (bid_depths, ask_depths)
            return _POLICY_CACHE[key]

        V_next = V[n+1]
        V_curr = V[n]
        
        # Vectorized computation for all q
        # q indices: 0 to n_states-1 correspond to q = -Q to Q
        q_vals = np.arange(-Q, Q + 1)
        
        # Delta V for bid: V(q+1) - V(q)
        # For q < Q, q+1 is valid. For q=Q, no bid.
        # idx q corresponds to q_val.
        # V_next[idx+1] - V_next[idx]
        
        # Bid side: valid for q < Q (indices 0 to n_states-2)
        dV_bid = V_next[1:] - V_next[:-1] # Shape (n_states-1,)
        # Ask side: valid for q > -Q (indices 1 to n_states-1)
        # V(q-1) - V(q) => V_next[idx-1] - V_next[idx]
        dV_ask = V_next[:-1] - V_next[1:] # Shape (n_states-1,)
        
        # Solve for optimal depths
        # This loop is small (60 iterations), so Python loop is fine.
        # We can also vectorize the Newton steps if we handle the array carefully, 
        # but given the small size, a simple loop is robust.
        
        # Initialize with 1/k
        opt_d_bid = np.full(n_states - 1, 1.0/k)
        opt_d_ask = np.full(n_states - 1, 1.0/k)
        
        # We can parallelize this or just do it sequentially.
        # Let's do a simple loop for clarity and safety, or vectorized Newton.
        # Vectorized Newton:
        d_bid = np.full(n_states - 1, 1.0/k)
        d_ask = np.full(n_states - 1, 1.0/k)
        
        for _ in range(20):
            # Bid
            E_b = np.exp(-k * d_bid)
            aE_b = alpha * E_b
            P_b = 1.0 - np.exp(-aE_b)
            Pp_b = k * aE_b * np.exp(-aE_b)
            Ppp_b = k * k * aE_b * np.exp(-aE_b) * (aE_b - 1.0)
            
            Jp_b = Pp_b * (d_bid + dV_bid) + P_b
            Jpp_b = Ppp_b * (d_bid + dV_bid) + 2.0 * Pp_b
            
            valid_b = np.abs(Jpp_b) > 1e-12
            step_b = np.zeros_like(d_bid)
            step_b[valid_b] = Jp_b[valid_b] / Jpp_b[valid_b]
            
            d_bid_new = d_bid - step_b
            d_bid_new = np.clip(d_bid_new, 0.0, 10.0/k)
            
            if np.max(np.abs(d_bid_new - d_bid)) < 1e-8:
                d_bid = d_bid_new
                break
            d_bid = d_bid_new
            
            # Ask
            E_a = np.exp(-k * d_ask)
            aE_a = alpha * E_a
            P_a = 1.0 - np.exp(-aE_a)
            Pp_a = k * aE_a * np.exp(-aE_a)
            Ppp_a = k * k * aE_a * np.exp(-aE_a) * (aE_a - 1.0)
            
            Jp_a = Pp_a * (d_ask + dV_ask) + P_a
            Jpp_a = Ppp_a * (d_ask + dV_ask) + 2.0 * Pp_a
            
            valid_a = np.abs(Jpp_a) > 1e-12
            step_a = np.zeros_like(d_ask)
            step_a[valid_a] = Jp_a[valid_a] / Jpp_a[valid_a]
            
            d_ask_new = d_ask - step_a
            d_ask_new = np.clip(d_ask_new, 0.0, 10.0/k)
            
            if np.max(np.abs(d_ask_new - d_ask)) < 1e-8:
                d_ask = d_ask_new
                break
            d_ask = d_ask_new
        
        # Fill in the tables
        # Bid: indices 0 to n_states-2
        bid_depths[n, :-1] = d_bid
        # Ask: indices 1 to n_states-1
        ask_depths[n, 1:] = d_ask
        
        # Calculate V_curr
        # V_curr[q] = -cost + V_next[q] + E[gains]
        # Gains:
        # Bid gain: P_b * (d_bid + V_next[q+1] - V_next[q]) ? 
        # No, the value equation is:
        # V_n(q) = -c + (1-p_b-p_a)V_{n+1}(q) + p_b(d_b + V_{n+1}(q+1)) + p_a(d_a + V_{n+1}(q-1))
        # = V_{n+1}(q) - c + p_b(d_b + V_{n+1}(q+1) - V_{n+1}(q)) + p_a(d_a + V_{n+1}(q-1) - V_{n+1}(q))
        # = V_{n+1}(q) - c + p_b(d_b + dV_bid) + p_a(d_a + dV_ask)
        
        cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        # Initialize gains to 0
        gains_b = np.zeros(n_states)
        gains_a = np.zeros(n_states)
        
        # Bid gains for q < Q
        if n_states > 1:
            P_b_vals = 1.0 - np.exp(-alpha * np.exp(-k * d_bid))
            gains_b[:-1] = P_b_vals * (d_bid + dV_bid)
            
            # Ask gains for q > -Q
            P_a_vals = 1.0 - np.exp(-alpha * np.exp(-k * d_ask))
            gains_a[1:] = P_a_vals * (d_ask + dV_ask)
        
        V_curr = V_next - cost + gains_b + gains_a
        V[n] = V_curr

    _POLICY_CACHE[key] = (bid_depths, ask_depths)
    return _POLICY_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) arrays of shape (4000,)
    """
    # Get or compute policy
    bid_table, ask_table = _compute_policy(params)
    
    # Determine step index
    steps = params["steps"]
    T = params["T"]
    dt = T / steps
    
    n = int(t / dt)
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    Q = params["Q"]
    
    # Map inventory to index
    # inventory is array of ints
    # idx = inventory + Q
    # Clip to valid range just in case
    inv_idx = np.clip(inventory + Q, 0, 2 * Q)
    
    # Lookup
    bid_depths = bid_table[n, inv_idx]
    ask_depths = ask_table[n, inv_idx]
    
    # Ensure floats
    return bid_depths.astype(float), ask_depths.astype(float)
