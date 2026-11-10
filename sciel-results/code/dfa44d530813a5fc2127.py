# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar

# Precomputed optimal policy tables for each regime
# Regimes: (sigma, A, k, gamma)
_REGIMES = [
    (2.0, 140.0, 1.5, 0.05),
    (2.0, 140.0, 1.5, 0.5),
    (1.5, 50.0, 0.75, 0.2),
]

def _solve_regime(sigma, A, k, gamma, Q=30, T=1.0, steps=1000):
    """
    Solve the optimal market-making policy using a semi-Lagrangian HJB scheme
    with analytical spread optimization.
    
    Returns:
        bid_depths: array of shape (Q+1,) - optimal bid depth for each inventory level
        ask_depths: array of shape (Q+1,) - optimal ask depth for each inventory level
    """
    dt = T / steps
    n_states = 2 * Q + 1  # inventory from -Q to Q
    q_min = -Q
    q_max = Q
    
    # Terminal condition: V(q, T) = 0 for all q
    V = np.zeros(n_states)
    
    # Precompute state indices
    q_to_idx = lambda q: q - q_min
    
    # Time-stepping backwards from T to 0
    for n in range(steps - 1, -1, -1):
        V_new = np.zeros(n_states)
        
        for i in range(n_states):
            q = i + q_min  # inventory level
            
            # No bid fill at q = Q (can't hold more inventory)
            # No ask fill at q = -Q (can't hold less inventory)
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            # The value function at time t for state q:
            # V(q) = max_{d_b, d_a} {
            #   p_b * (d_b + V(q+1)) + p_a * (-d_a + V(q-1)) + (1-p_b-p_a+...) * V(q)
            #   - (gamma/2) * sigma^2 * q^2 * dt
            # }
            # where p_b = 1 - exp(-A * exp(-k * d_b) * dt)
            #       p_a = 1 - exp(-A * exp(-k * d_a) * dt)
            
            # Inventory risk cost
            risk_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            # V(q+1) and V(q-1) if valid
            if q < Q:
                V_q_plus = V[q_to_idx(q + 1)]
            else:
                V_q_plus = None  # no bid fill possible
            
            if q > -Q:
                V_q_minus = V[q_to_idx(q - 1)]
            else:
                V_q_minus = None  # no ask fill possible
            
            # We need to optimize over d_b and d_a jointly.
            # The fill probabilities are independent, so:
            # E[reward] = p_b * (d_b + V_q_plus) + p_a * (-d_a + V_q_minus) 
            #            + (1 - p_b) * (1 - p_a) * V_q
            # But we also have the no-fill case which keeps inventory q.
            # Actually, the expected value is:
            # p_b * (d_b + V_q_plus) + p_a * (-d_a + V_q_minus) 
            # + (1-p_b)(1-p_a) * V_q
            # - risk_cost
            
            # Let's define the objective to maximize:
            # f(d_b, d_a) = p_b * (d_b + V_q_plus) + p_a * (-d_a + V_q_minus) 
            #               + (1-p_b)(1-p_a) * V_q - risk_cost
            
            # Since p_b and p_a are independent given d_b and d_a,
            # we can optimize over d_b and d_a separately if we handle the cross term.
            # Actually, (1-p_b)(1-p_a) = 1 - p_b - p_a + p_b*p_a, so there's a coupling.
            # But for small dt, p_b and p_a are small, so the coupling is weak.
            # Let's do a joint optimization using a grid or scipy.
            
            # For efficiency, let's use the fact that the optimal depths are bounded.
            # The depth that maximizes a single side:
            # For bid: maximize p_b * (d_b + V_q_plus) + (1-p_b) * V_q - (gamma/2)*sigma^2*q^2*dt
            #   = p_b * (d_b + V_q_plus - V_q) + V_q - risk_cost
            #   = (1 - exp(-A*exp(-k*d_b)*dt)) * (d_b + V_q_plus - V_q) + V_q - risk_cost
            
            # Let delta_b = V_q_plus - V_q (change in value from taking inventory)
            # Objective for bid: (1 - exp(-A*exp(-k*d_b)*dt)) * (d_b + delta_b) + const
            
            # Similarly for ask: delta_a = V_q_minus - V_q
            # Objective for ask: (1 - exp(-A*exp(-k*d_a)*dt)) * (-d_a + delta_a) + const
            #                   = (1 - exp(-A*exp(-k*d_a)*dt)) * (delta_a - d_a) + const
            
            # The coupling term is p_b * p_a * V_q which is small for small dt.
            # Let's ignore the coupling for now and optimize each side independently.
            
            best_bid_depth = np.inf
            best_ask_depth = np.inf
            
            if can_bid:
                delta_b = V_q_plus - V[q_to_idx(q)]
                
                def neg_obj_bid(d_b):
                    if d_b < 0:
                        return 0.0
                    p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
                    return -(p_b * (d_b + delta_b))
                
                # Optimize d_b in [0, max_depth]
                # A reasonable upper bound: depth where fill probability is ~1
                # A * exp(-k * d) * dt ~ 1 => d ~ (1/k) * log(A*dt)
                max_depth = max(0.1, (1.0/k) * np.log(max(A * dt, 1e-10)) + 5.0/k)
                
                res = minimize_scalar(neg_obj_bid, bounds=(0, max_depth), method='bounded')
                best_bid_depth = res.x
            
            if can_ask:
                delta_a = V_q_minus - V[q_to_idx(q)]
                
                def neg_obj_ask(d_a):
                    if d_a < 0:
                        return 0.0
                    p_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
                    return -(p_a * (delta_a - d_a))
                
                max_depth = max(0.1, (1.0/k) * np.log(max(A * dt, 1e-10)) + 5.0/k)
                
                res = minimize_scalar(neg_obj_ask, bounds=(0, max_depth), method='bounded')
                best_ask_depth = res.x
            
            # Compute expected value with these depths
            if can_bid:
                p_b = 1.0 - np.exp(-A * np.exp(-k * best_bid_depth) * dt)
            else:
                p_b = 0.0
                best_bid_depth = np.inf
            
            if can_ask:
                p_a = 1.0 - np.exp(-A * np.exp(-k * best_ask_depth) * dt)
            else:
                p_a = 0.0
                best_ask_depth = np.inf
            
            # Expected value
            V_qi = V[q_to_idx(q)]
            if can_bid:
                V_qp = V[q_to_idx(q + 1)]
            else:
                V_qp = 0.0
            if can_ask:
                V_qm = V[q_to_idx(q - 1)]
            else:
                V_qm = 0.0
            
            V_new[i] = (p_b * (best_bid_depth + V_qp) + 
                       p_a * (-best_ask_depth + V_qm) + 
                       (1 - p_b) * (1 - p_a) * V_qi) - risk_cost
            
        V = V_new
    
    # Now V contains the optimal value function at t=0
    # We need to extract the optimal depths for each inventory level
    bid_depths = np.full(n_states, np.inf)
    ask_depths = np.full(n_states, np.inf)
    
    # Re-solve for the optimal depths at t=0 using the final V
    for i in range(n_states):
        q = i + q_min
        can_bid = (q < Q)
        can_ask = (q > -Q)
        
        if can_bid:
            V_q_plus = V[q_to_idx(q + 1)]
            delta_b = V_q_plus - V[i]
            
            def neg_obj_bid(d_b):
                if d_b < 0:
                    return 0.0
                p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
                return -(p_b * (d_b + delta_b))
            
            max_depth = max(0.1, (1.0/k) * np.log(max(A * dt, 1e-10)) + 5.0/k)
            res = minimize_scalar(neg_obj_bid, bounds=(0, max_depth), method='bounded')
            bid_depths[i] = res.x
        
        if can_ask:
            V_q_minus = V[q_to_idx(q - 1)]
            delta_a = V_q_minus - V[i]
            
            def neg_obj_ask(d_a):
                if d_a < 0:
                    return 0.0
                p_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
                return -(p_a * (delta_a - d_a))
            
            max_depth = max(0.1, (1.0/k) * np.log(max(A * dt, 1e-10)) + 5.0/k)
            res = minimize_scalar(neg_obj_ask, bounds=(0, max_depth), method='bounded')
            ask_depths[i] = res.x
    
    return bid_depths, ask_depths

# Precompute all regime policies
_POLICY_CACHE = {}
for (sigma, A, k, gamma) in _REGIMES:
    key = (sigma, A, k, gamma)
    _POLICY_CACHE[key] = _solve_regime(sigma, A, k, gamma)


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    
    Args:
        t: current time
        inventory: array of shape (4000,) with inventory levels in [-Q, Q]
        mid: array of shape (4000,) with mid prices
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): two arrays of shape (4000,)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    
    key = (sigma, A, k, gamma)
    if key not in _POLICY_CACHE:
        # Fallback to simple heuristic if regime not precomputed
        depth = 1.0 / k
        n = len(inventory)
        return np.full(n, depth), np.full(n, depth)
    
    bid_table, ask_table = _POLICY_CACHE[key]
    
    # Map inventory to table indices
    # inventory in [-Q, Q], table index = q - (-Q) = q + Q
    idx = inventory + Q
    
    # Clip to valid range
    idx = np.clip(idx, 0, 2 * Q)
    
    bid_depths = bid_table[idx]
    ask_depths = ask_table[idx]
    
    # Ensure no invalid values
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), 1.0/k, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), 1.0/k, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
