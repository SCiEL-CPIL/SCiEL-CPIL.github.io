import numpy as np
import time

# Global cache for the optimal policy
_POLICY_CACHE = {}

def _solve_optimal_depth(delta, A, k, dt):
    """
    Solve for optimal depth d that maximizes P(d) * (d + delta)
    where P(d) = 1 - exp(-A * exp(-k*d) * dt)
    """
    C = A * dt
    # If C is very small, probability is very small, optimal depth is 0?
    # Actually if C->0, P(d) ~ C exp(-kd).
    # f(d) ~ C exp(-kd) (d+delta).
    # f'(d) = C [ -k exp(-kd)(d+delta) + exp(-kd) ] = C exp(-kd) [ 1 - k(d+delta) ] = 0
    # d = 1/k - delta.
    # If 1/k - delta < 0, then d=0.
    
    # General case:
    # Equation: 1 - exp(-u) = k * C * u * (d + delta)
    # where u = C * exp(-k*d) => d = -1/k * ln(u/C)
    # Let G(u) = 1 - exp(-u) - k * C * u * (delta - 1/k * ln(u/C))
    # We want G(u) = 0.
    
    # Initial guess
    # If delta is large positive, we want small d (large u).
    # If delta is large negative, we want large d (small u).
    
    # Let's use Newton's method on u.
    # G(u) = 1 - e^{-u} - k C u delta + C u ln(u/C)
    # G'(u) = e^{-u} - k C delta + C ln(u/C) + C
    
    # Handle u -> 0. If u is very small, ln(u/C) is large negative.
    # If delta is very negative, d should be large, u small.
    
    # Let's try a few initial guesses or use a robust method.
    # Since k, C are fixed for the regime, and delta varies,
    # we can bound u.
    
    # If delta <= 0, optimal d is likely >= 1/k.
    # If delta > 0, optimal d is likely < 1/k.
    
    # Let's just solve for d directly?
    # H(d) = 1 - exp(-C exp(-kd)) - k C exp(-kd) (d + delta) = 0
    # H'(d) = -k C exp(-kd) exp(-C exp(-kd)) - k C [ -k exp(-kd)(d+delta) + exp(-kd) ]
    #       = -k C e^{-kd} [ e^{-C e^{-kd}} - k(d+delta) + 1 ]
    # Setting H'(d) = 0 => e^{-C e^{-kd}} = 1 + k(d+delta)
    # This is much simpler!
    # Let y = C e^{-kd}. Then e^{-y} = 1 + k(d+delta).
    # Also y = C e^{-kd} => d = -1/k ln(y/C).
    # So e^{-y} = 1 + k(-1/k ln(y/C) + delta) = 1 - ln(y/C) + k delta.
    # Let L(y) = e^{-y} + ln(y/C) - 1 - k delta = 0.
    
    # Solve L(y) = 0 for y.
    # L'(y) = -e^{-y} + 1/y.
    
    # If 1 + k(d+delta) < 0, no solution (derivative doesn't cross 0 in valid range).
    # This happens if k(d+delta) < -1.
    # Since d >= 0, if delta < -1/k, then k(d+delta) can be < -1.
    # In that case, the function f(d) is decreasing for all d >= 0?
    # f'(d) sign is opposite to H'(d) factor?
    # Recall f'(d) = P'(d)(d+delta) + P(d).
    # We set f'(d)=0.
    # If 1 + k(d+delta) < 0, then e^{-y} < 0 which is impossible.
    # This implies f'(d) never becomes 0?
    # Let's check f'(0).
    # If f'(0) < 0, then max is at d=0.
    
    # So:
    # 1. Check if solution y exists for L(y)=0.
    # 2. If not, or if resulting d < 0, set d=0.
    
    # Let's solve L(y) = 0.
    # y must be > 0.
    # As y -> 0, ln(y/C) -> -inf, L(y) -> -inf.
    # As y -> inf, e^{-y} -> 0, ln(y/C) -> inf, L(y) -> inf.
    # So there is always a root y > 0.
    # The corresponding d = -1/k ln(y/C).
    # If d < 0, we clamp to 0.
    
    # Newton method for L(y) = 0
    # Initial guess: y0 = C (implies d=0).
    # If delta is large positive, d is small, y is large.
    # If delta is large negative, d is large, y is small.
    
    # Let's just use a simple iteration or Newton.
    y = C
    for _ in range(10):
        # L(y) = exp(-y) + log(y/C) - 1 - k*delta
        # L'(y) = -exp(-y) + 1/y
        exp_neg_y = np.exp(-y)
        L_val = exp_neg_y + np.log(y / C) - 1.0 - k * delta
        L_deriv = -exp_neg_y + 1.0 / y
        
        if abs(L_deriv) < 1e-15:
            break
            
        y_new = y - L_val / L_deriv
        if y_new <= 0:
            y_new = y / 2.0
        if abs(y_new - y) < 1e-12:
            y = y_new
            break
        y = y_new
        
    d = -1.0 / k * np.log(y / C)
    if d < 0:
        d = 0.0
    return d

def quote(t, inventory, mid, params):
    global _POLICY_CACHE
    
    # Check if params changed
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _POLICY_CACHE:
        # Compute optimal policy
        sigma = params['sigma']
        A = params['A']
        k = params['k']
        gamma = params['gamma']
        T = params['T']
        steps = params['steps']
        Q = params['Q']
        
        dt = T / steps
        n_q = 2 * Q + 1
        
        # V[t, q_index]
        # q_index 0 corresponds to q = -Q
        # q_index Q corresponds to q = 0
        # q_index 2Q corresponds to q = Q
        
        # We store bid and ask depths for each t and q
        # bid_depths[t, q_index]
        # ask_depths[t, q_index]
        
        bid_depths = np.zeros((steps, n_q))
        ask_depths = np.zeros((steps, n_q))
        
        # V_next
        V_next = np.zeros(n_q)
        
        # Backward induction
        for t in range(steps - 1, -1, -1):
            V_curr = np.zeros(n_q)
            for q_idx in range(n_q):
                q = q_idx - Q
                
                # Inventory holding cost
                hold_cost = 0.5 * gamma * sigma**2 * q**2 * dt
                
                # Calculate deltas
                # Delta V_minus = V(q-1) - V(q)
                # Delta V_plus = V(q+1) - V(q)
                
                if q > -Q:
                    delta_minus = V_next[q_idx - 1] - V_next[q_idx]
                else:
                    delta_minus = -np.inf # No bid fill possible
            
                if q < Q:
                    delta_plus = V_next[q_idx + 1] - V_next[q_idx]
                else:
                    delta_plus = -np.inf # No ask fill possible
            
                # Optimize b
                if q == Q:
                    b_opt = np.inf # No bid
                else:
                    b_opt = _solve_optimal_depth(delta_minus, A, k, dt)
                
                # Optimize a
                if q == -Q:
                    a_opt = np.inf # No ask
                else:
                    a_opt = _solve_optimal_depth(delta_plus, A, k, dt)
                
                bid_depths[t, q_idx] = b_opt
                ask_depths[t, q_idx] = a_opt
                
                # Calculate expected value
                # P_b = 1 - exp(-A exp(-k b) dt)
                # P_a = 1 - exp(-A exp(-k a) dt)
                
                if q == Q:
                    P_b = 0.0
                else:
                    P_b = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
                
                if q == -Q:
                    P_a = 0.0
                else:
                    P_a = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
                
                # Value
                # E = V_next[q] + P_b * (b_opt + delta_minus) + P_a * (a_opt + delta_plus) - hold_cost
                # Note: if q=Q, delta_minus is -inf, but P_b=0, so term is 0.
                # We need to handle the 0 * inf case.
                
                term_b = 0.0
                if q != Q:
                    term_b = P_b * (b_opt + delta_minus)
                    
                term_a = 0.0
                if q != -Q:
                    term_a = P_a * (a_opt + delta_plus)
                    
                V_curr[q_idx] = V_next[q_idx] + term_b + term_a - hold_cost
            
            V_next = V_curr
        
        _POLICY_CACHE[key] = (bid_depths, ask_depths)
    
    bid_table, ask_table = _POLICY_CACHE[key]
    
    # Map inventory to index
    Q = params['Q']
    # inventory is (4000,)
    # q_index = inventory + Q
    idx = inventory + Q
    
    # Get depths
    # t is the current step index (0..999)
    # The DP was computed for t=0..steps-1
    # V_t is the value *before* the step t's fills?
    # "q the inventory before that step's fills"
    # "called at every step n = 0..999"
    # So at step n, we use the policy calculated for time index n.
    
    b_depths = bid_table[t, idx]
    a_depths = ask_table[t, idx]
    
    # Handle inf (no quote)
    # np.inf is allowed? "np.inf means no quote"
    # But "Values below -3/k ... invalid". Inf is not below.
    # However, returning inf might be tricky for the evaluator if it expects floats.
    # The prompt says "np.inf means no quote". So we can return inf.
    
    return b_depths, a_depths
