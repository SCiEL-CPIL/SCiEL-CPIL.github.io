import numpy as np
import time

_dp_cache = {}

def _compute_dp_tables(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    num_q = 2 * Q + 1
    q_min = -Q
    q_max = Q
    
    d_max = 5.0 / k
    M = 50
    depths = np.linspace(0, d_max, M)
    
    exponents = -k * depths
    probs_base = 1.0 - np.exp(-A * np.exp(exponents) * dt)
    rewards_base = probs_base * depths
    
    V_next = np.zeros(num_q)
    policy_bid = np.zeros((steps, num_q))
    policy_ask = np.zeros((steps, num_q))
    
    q_vals = np.arange(q_min, q_max + 1)
    penalty = 0.5 * gamma * (sigma ** 2) * (q_vals ** 2) * dt
    
    for t in range(steps - 1, -1, -1):
        V_curr = np.zeros(num_q)
        
        for i in range(num_q):
            q = q_vals[i]
            
            p_b = probs_base.copy()
            p_a = probs_base.copy()
            r_b = rewards_base.copy()
            r_a = rewards_base.copy()
            
            if q == Q:
                p_b[:] = 0.0
                r_b[:] = 0.0
            if q == -Q:
                p_a[:] = 0.0
                r_a[:] = 0.0
            
            V_q = V_next[i]
            V_qm1 = V_next[i-1] if i > 0 else 0.0
            V_qp1 = V_next[i+1] if i < num_q - 1 else 0.0
            
            Pb = p_b[:, np.newaxis]
            Pa = p_a[np.newaxis, :]
            
            # Bid fill -> q+1. Ask fill -> q-1.
            # E[V] = V_q * ( (1-Pb)(1-Pa) + Pb Pa ) 
            #       + V_qp1 * Pb (1-Pa) 
            #       + V_qm1 * Pa (1-Pb)
            
            term_q = V_q * ( (1.0 - Pb) * (1.0 - Pa) + Pb * Pa )
            term_qp1 = V_qp1 * Pb * (1.0 - Pa)
            term_qm1 = V_qm1 * Pa * (1.0 - Pb)
            
            E_val = term_q + term_qp1 + term_qm1
            
            Rb = r_b[:, np.newaxis]
            Ra = r_a[np.newaxis, :]
            R_val = Rb + Ra
            
            Total = -penalty[i] + R_val + E_val
            
            max_idx = np.argmax(Total)
            idx_b, idx_a = np.unravel_index(max_idx, (M, M))
            
            V_curr[i] = Total[max_idx]
            policy_bid[t, i] = depths[idx_b]
            policy_ask[t, i] = depths[idx_a]
            
        V_next = V_curr
        
    return policy_bid, policy_ask

def quote(t, inventory, mid, params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _dp_cache:
        start_time = time.time()
        policy_bid, policy_ask = _compute_dp_tables(params)
        _dp_cache[key] = (policy_bid, policy_ask)
        elapsed = time.time() - start_time
        # If this took too long, we might have issues, but 300s is large.
        
    else:
        policy_bid, policy_ask = _dp_cache[key]
        
    steps = params['steps']
    T = params['T']
    dt = T / steps
    
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    Q = params['Q']
    q_indices = inventory + Q
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    bid_depths = policy_bid[n, q_indices]
    ask_depths = policy_ask[n, q_indices]
    
    return bid_depths, ask_depths
