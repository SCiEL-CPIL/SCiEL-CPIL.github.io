"""
Strategy: Continuous-depth optimal market making via scalar root-finding.
Instead of discretizing the depth space into a grid, we treat the depth as a continuous variable
and solve the Bellman optimality condition for the optimal depth using Brent's method.
The objective function for depth is unimodal (concave) for reasonable parameters, allowing
fast and exact convergence to the optimal depth. This eliminates grid discretization error
and allows for much higher accuracy within the time budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

_policy_cache = {}
_start_time = time.time()
TIME_LIMIT = 325.0  # Safety margin (limit is 330s)

def _compute_optimal_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Precompute cost
    cost_per_state = 0.5 * gamma * sigma**2 * q_values**2 * dt
    
    # Bounds for depth search
    # Minimum depth: small positive number
    d_min = 1e-4
    # Maximum depth: where probability becomes negligible
    # p = 1 - exp(-A exp(-k d) dt)
    # For d > 10/k, exp(-k d) < exp(-10) ~ 4.5e-5
    # A * exp(-k d) * dt is very small
    d_max = 10.0 / k
    
    # We will iterate to find the stationary policy.
    # V[i] is the value for inventory i (where i=0 corresponds to q=-Q, etc? No, let's map carefully)
    # Let index i correspond to q = i - Q. So i=0 -> q=-Q, i=2Q -> q=Q.
    # V[i] = V(q=i-Q)
    
    # Initialize V to 0
    V = np.zeros(n_states)
    
    opt_bid_q = np.zeros(n_states)
    opt_ask_q = np.zeros(n_states)
    
    max_iter = 100
    tol = 1e-8
    
    for it in range(max_iter):
        if time.time() - _start_time > TIME_LIMIT:
            break
            
        new_V = np.zeros(n_states)
        new_opt_bid = np.zeros(n_states)
        new_opt_ask = np.zeros(n_states)
        
        for i in range(n_states):
            q = i - Q
            
            # Get neighbor values
            # Bid fill: q -> q-1 (index i-1). If q == -Q (i==0), no bid fill allowed.
            # Ask fill: q -> q+1 (index i+1). If q == Q (i==n_states-1), no ask fill allowed.
            
            v_q = V[i]
            
            if i > 0:
                v_qp = V[i-1]  # Value after bid fill (q decreases)
            else:
                v_qp = 0.0     # No bid fill at boundary, or treat as 0 contribution?
                # Actually, if no bid fill is allowed, we must set bid_depth = inf.
                # But in the optimization, we can handle this by setting the bid term to -inf or simply skipping.
                # Let's handle boundaries explicitly.
            
            if i < n_states - 1:
                v_qm = V[i+1]  # Value after ask fill (q increases)
            else:
                v_qm = 0.0
            
            # Define objective function for bid depth d_b and ask depth d_a
            # We want to maximize:
            # (1-pb)(1-pa) v_q + pb(1-pa) (v_qp + d_b) + (1-pb)pa (v_qm + d_a) + pb pa (v_q + d_b + d_a) - cost
            # Let's separate the maximization. It's a coupled problem.
            # However, for a given d_a, the optimal d_b can be found, and vice versa.
            # Or we can maximize jointly. Since the function is separable in terms of the "fill" terms?
            # Let's expand:
            # E[reward] = v_q (1-pb)(1-pa) + pb(1-pa)(v_qp + d_b) + (1-pb)pa(v_qm + d_a) + pb pa (v_q + d_b + d_a)
            # = v_q [ (1-pb)(1-pa) + pb pa ] + pb(1-pa)(v_qp + d_b) + (1-pb)pa(v_qm + d_a) + pb pa (d_b + d_a)
            # Note: (1-pb)(1-pa) + pb pa = 1 - pb - pa + 2 pb pa. Not constant.
            
            # Let's define the function to maximize for a pair (d_b, d_a).
            # Since the dimension is 2, we can use scipy.optimize.minimize (negate for max) or brentq if separable.
            # Is it separable?
            # Term with d_b: pb(1-pa) d_b + pb pa d_b = pb d_b (1-pa + pa) = pb d_b
            # Term with d_a: (1-pb)pa d_a + pb pa d_a = pa d_a (1-pb + pb) = pa d_a
            # Wait, let's re-verify the expansion.
            # F(d_b, d_a) = (1-pb)(1-pa)v_q + pb(1-pa)(v_qp+d_b) + (1-pb)pa(v_qm+d_a) + pb pa(v_q+d_b+d_a)
            # Coeff of d_b: pb(1-pa) + pb pa = pb
            # Coeff of d_a: (1-pb)pa + pb pa = pa
            # So F = [ (1-pb)(1-pa)v_q + pb(1-pa)v_qp + (1-pb)pa v_qm + pb pa v_q ] + pb d_b + pa d_a
            # The value terms do not depend on d_b or d_a except through p_b and p_a.
            # Let G(d_b, d_a) = (1-pb)(1-pa)v_q + pb(1-pa)v_qp + (1-pb)pa v_qm + pb pa v_q + pb d_b + pa d_a
            # This is still coupled because p_b depends on d_b and p_a depends on d_a.
            # However, we can solve for d_b given d_a, and d_a given d_b?
            # Let's fix d_a and maximize over d_b.
            # p_a is constant.
            # H(d_b) = (1-pb)(1-pa)v_q + pb(1-pa)v_qp + (1-pb)pa v_qm + pb pa v_q + pb d_b + pa d_a
            # H(d_b) = (1-pb)[(1-pa)v_q + pa v_qm] + pb[(1-pa)v_qp + pa v_q + d_b] + pa d_a
            # Let C1 = (1-pa)v_q + pa v_qm
            # Let C2 = (1-pa)v_qp + pa v_q
            # H(d_b) = (1-pb)C1 + pb(C2 + d_b) + pa d_a
            # H(d_b) = C1 - pb C1 + pb C2 + pb d_b + pa d_a
            # H(d_b) = C1 + pb (C2 - C1 + d_b) + pa d_a
            # We want to maximize this over d_b >= d_min.
            # Let f(d_b) = pb(d_b) * (C2 - C1 + d_b).
            # pb(d_b) = 1 - exp(-A exp(-k d_b) dt)
            
            # This function f(d_b) is typically unimodal.
            # We can use brentq on the derivative, or just minimize -H(d_b) using a bounded scalar minimizer.
            # scipy.optimize.minimize_scalar is good for 1D.
            
            # To be robust and fast, let's use a simple grid + refinement or just minimize_scalar.
            # Given the time limit, minimize_scalar might be slow if called 61*100 times.
            # 61 states * 100 iterations = 6100 calls.
            # Each call to minimize_scalar takes ~10-20 evaluations.
            # 6100 * 20 = 122,000 function evaluations.
            # Each evaluation is cheap (exp, log). Should be fine in Python?
            # 122,000 Python calls might take ~1-2 seconds. Acceptable.
            
            # However, we have a coupled system.
            # Strategy: Alternating minimization.
            # 1. Fix d_a, find optimal d_b.
            # 2. Fix d_b, find optimal d_a.
            # Repeat a few times? Or just one pass?
            # Given the strong coupling, one pass might not be enough.
            # But since we are doing value iteration, the V values change slowly.
            # Let's try a few iterations of alternating optimization for each state.
            
            # Initialize d_b, d_a from previous iteration or heuristic
            d_b = opt_bid_q[i] if it > 0 else 1.0/k
            d_a = opt_ask_q[i] if it > 0 else 1.0/k
            
            # Handle boundaries
            can_bid = (i > 0)
            can_ask = (i < n_states - 1)
            
            if not can_bid and not can_ask:
                # No fills possible, value is just V[i] - cost
                new_V[i] = v_q - cost_per_state[i]
                new_opt_bid[i] = np.inf
                new_opt_ask[i] = np.inf
                continue
            
            # Define helper to evaluate H for bid given ask
            def eval_bid(d_b, d_a_val):
                if d_b < d_min: d_b = d_min
                if d_b > d_max: d_b = d_max
                p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
                p_a = 1.0 - np.exp(-A * np.exp(-k * d_a_val) * dt)
                
                C1 = (1-pa_val) * v_q + pa_val * v_qm if can_ask else (1-pa_val) * v_q # v_qm is 0 if not can_ask? 
                # Wait, if not can_ask, pa should be 0? No, we just don't quote.
                # If we don't quote ask, p_a = 0.
                # Let's enforce: if not can_ask, p_a = 0.
                # If not can_bid, p_b = 0.
                
                if not can_ask:
                    p_a = 0.0
                else:
                    p_a = 1.0 - np.exp(-A * np.exp(-k * d_a_val) * dt)
                
                if not can_bid:
                    p_b = 0.0
                else:
                    p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
                
                # H = (1-pb)(1-pa)v_q + pb(1-pa)(v_qp+d_b) + (1-pb)pa(v_qm+d_a) + pb pa(v_q+d_b+d_a)
                # If not can_bid, v_qp is irrelevant (pb=0).
                # If not can_ask, v_qm is irrelevant (pa=0).
                
                val = (1-pb)*(1-pa)*v_q
                if can_bid:
                    val += pb*(1-pa)*(v_qp + d_b)
                if can_ask:
                    val += (1-pb)*pa*(v_qm + d_a_val)
                if can_bid and can_ask:
                    val += pb*pa*(v_q + d_b + d_a_val)
                else:
                    # If one side is missing, the "both" term is 0
                    pass
                
                return val - cost_per_state[i]
            
            # Alternating optimization
            for _ in range(5): # 5 iterations of alternating min
                # Optimize d_b
                if can_bid:
                    res_b = minimize_scalar(lambda x: -eval_bid(x, d_a), bounds=(d_min, d_max), method='bounded')
                    d_b = res_b.x
                # Optimize d_a
                if can_ask:
                    res_a = minimize_scalar(lambda x: -eval_bid(d_b, x), bounds=(d_min, d_max), method='bounded')
                    d_a = res_a.x
            
            new_V[i] = eval_bid(d_b, d_a)
            new_opt_bid[i] = d_b if can_bid else np.inf
            new_opt_ask[i] = d_a if can_ask else np.inf
        
        V = new_V
        opt_bid_q = new_opt_bid
        opt_ask_q = new_opt_ask
        
        # Check convergence
        # We can check max change in V
        # diff = np.max(np.abs(V - old_V)) # Need to store old V
        # For simplicity, run fixed number of iterations or rely on time limit.
        # Let's add a simple convergence check
        if it > 0:
             diff = np.max(np.abs(V - V_prev))
             if diff < 1e-6:
                 break
        V_prev = V.copy()

    # Construct the full policy arrays of shape (steps, n_states)
    # Stationary policy
    opt_bid = np.tile(opt_bid_q[None, :], (steps, 1))
    opt_ask = np.tile(opt_ask_q[None, :], (steps, 1))
    
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """
    Look up optimal depths from the pre-computed policy.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
        
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    return bids, asks

# EVOLVE-BLOCK-END
