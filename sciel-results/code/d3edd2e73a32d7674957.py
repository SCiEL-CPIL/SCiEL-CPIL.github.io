import numpy as np
import scipy.optimize as opt

_policy_cache = {}

def _optimize_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    lambda_base = A * dt
    
    # Objective function for parametric policy (A_poly, B_poly)
    def objective(params_ab):
        A_poly = params_ab[0]
        B_poly = params_ab[1]
        
        # Ensure non-negative depths
        # We evaluate the expected reward starting from q=0
        # P_t is array of size 2Q+1
        P_t = np.zeros(2*Q + 1)
        P_t[Q] = 1.0 # Start at 0 inventory
        
        total_reward = 0.0
        
        for t in range(steps):
            q_vals = np.arange(-Q, Q+1)
            
            # Compute depths for each q
            d_b = np.maximum(0, A_poly - B_poly * q_vals)
            d_a = np.maximum(0, A_poly + B_poly * q_vals)
            
            # Compute fill probabilities
            # P(d) = 1 - exp(-lambda * exp(-k*d))
            # Handle d=0 -> P=0? exp(-k*0)=1. P = 1-exp(-lambda).
            # If d is large, P is small.
            
            # Vectorized probability calculation
            exp_term_b = np.exp(-k * d_b)
            P_b = 1.0 - np.exp(-lambda_base * exp_term_b)
            
            exp_term_a = np.exp(-k * d_a)
            P_a = 1.0 - np.exp(-lambda_base * exp_term_a)
            
            # Risk cost
            risk_cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
            exp_risk = np.sum(P_t * risk_cost)
            
            # Immediate reward: d_b * P_b + d_a * P_a
            # Note: This is the expected value of the depth *earned* if filled?
            # "A fill earns its depth".
            # So if bid fills, we earn d_b. Prob is P_b.
            # Expected gain = d_b * P_b.
            exp_gain_b = np.sum(P_t * d_b * P_b)
            exp_gain_a = np.sum(P_t * d_a * P_a)
            
            total_reward += exp_gain_b + exp_gain_a - exp_risk
            
            # Propagate probability distribution
            P_next = np.zeros(2*Q + 1)
            
            # Transitions:
            # q -> q+1 with prob P_b * (1 - P_a)
            # q -> q-1 with prob P_a * (1 - P_b)
            # q -> q with prob (1 - P_b)(1 - P_a) + P_b P_a
            
            # Indices: q is at index i = q + Q
            # q+1 is at i+1
            # q-1 is at i-1
            
            # We need to be careful with boundaries q=Q and q=-Q
            # "no bid fill at q = Q" -> P_b[Q] should be 0?
            # "no ask fill at q = -Q" -> P_a[-Q] should be 0?
            # The task says: "no bid fill at q = Q, no ask fill at q = -Q".
            # So we set P_b[Q] = 0 and P_a[-Q] = 0.
            
            P_b[Q+Q] = 0.0 # Index for q=Q is 2Q
            P_a[0] = 0.0   # Index for q=-Q is 0
            
            # Recalculate gains with modified probs?
            # The reward calculation above used P_b, P_a.
            # If we force P_b=0 at Q, the gain is 0 there, which is correct.
            
            # Update P_next
            # P_next[i+1] += P_t[i] * P_b[i] * (1 - P_a[i])
            # P_next[i-1] += P_t[i] * P_a[i] * (1 - P_b[i])
            # P_next[i]   += P_t[i] * (1 - P_b[i] * (1 - P_a[i]) - P_a[i] * (1 - P_b[i]))
            #             = P_t[i] * (1 - P_b - P_a + 2 P_b P_a) ? No.
            # Prob(no fill) = (1-P_b)(1-P_a) = 1 - P_b - P_a + P_b P_a
            # Prob(both fill) = P_b P_a
            # Total stay = 1 - P_b - P_a + 2 P_b P_a
            
            # Vectorized update
            # Shifts
            # i from 0 to 2Q-1: P_t[i] contributes to i+1
            P_next[1:] += P_t[:-1] * P_b[:-1] * (1 - P_a[:-1])
            # i from 1 to 2Q: P_t[i] contributes to i-1
            P_next[:-1] += P_t[1:] * P_a[1:] * (1 - P_b[1:])
            # i from 0 to 2Q: P_t[i] contributes to i
            P_next += P_t * (1 - P_b - P_a + 2 * P_b * P_a)
            
            # Normalize just in case
            P_t = P_next / np.sum(P_next)
            
        return -total_reward # Minimize negative reward

    # Optimization
    bounds = [(0, 10.0/k), (0, 10.0/k)]
    result = opt.differential_evolution(objective, bounds, seed=42, maxiter=50, polish=True)
    
    A_opt = result.x[0]
    B_opt = result.x[1]
    
    return A_opt, B_opt

def quote(t, inventory, mid, params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        _policy_cache[key] = _optimize_policy(params)
        
    A_opt, B_opt = _policy_cache[key]
    
    q = inventory
    d_b = np.maximum(0, A_opt - B_opt * q)
    d_a = np.maximum(0, A_opt + B_opt * q)
    
    return d_b, d_a
