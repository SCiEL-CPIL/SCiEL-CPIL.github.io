"""Optimal market making with inventory risk using pre-computed backward-induction policy."""
import numpy as np
import time
from scipy.optimize import minimize_scalar

# Module-level cache for policy tables
_POLICY_CACHE = {}
_START_TIME = time.time()
_TIME_BUDGET = 320  # seconds, safety margin before 330s kill

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth for each (step, inventory) via backward induction.
    Uses grid search followed by local Newton refinement for speed and precision.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # Value function
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    
    # Policy tables: store actual float depths
    bid_policy = np.zeros((steps, n_inv), dtype=np.float64)
    ask_policy = np.zeros((steps, n_inv), dtype=np.float64)
    
    # Depth bounds for optimization
    # Probabilistic mass becomes negligible for large depths.
    # 1 - exp(-A exp(-k D) dt) ~ 0 when A exp(-k D) dt << 1
    # exp(-k D) << 1/(A dt) => -k D < ln(1/(A dt)) => D > ln(A dt) / k
    # For A=140, dt=0.001, ln(0.14) ~ -2, so D > 0.
    # But we want the range where the objective is maximized.
    # Typically depths are small. 10/k is a safe upper bound.
    max_depth = 10.0 / k
    
    # Precompute inventory indices and holding costs
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    hold_costs = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
    
    # Grid definitions for initial guess
    n_coarse = 40
    b_coarse = np.linspace(0, max_depth, n_coarse)
    a_coarse = np.linspace(0, max_depth, n_coarse)
    
    # Precompute fill probs for coarse grid to speed up evaluation
    # lambda_b = 1 - exp(-A exp(-k b) dt)
    PB_coarse = 1.0 - np.exp(-A * np.exp(-k * b_coarse) * dt)
    PA_coarse = 1.0 - np.exp(-A * np.exp(-k * a_coarse) * dt)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > _TIME_BUDGET:
            # If we run out of time, break. Remaining steps use default 0.0 depths.
            break
            
        V_next = V[n+1, :]
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            hold_cost = hold_costs[q_idx]
            
            V_next_q = V_next[q_idx]
            
            if q < Q:
                V_next_q_plus = V_next[q_idx + 1]
            else:
                V_next_q_plus = -np.inf # Won't be used if can_bid is false
            
            if q > -Q:
                V_next_q_minus = V_next[q_idx - 1]
            else:
                V_next_q_minus = -np.inf

            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            best_bid_depth = 0.0
            best_ask_depth = 0.0
            best_val = -np.inf
            
            if can_bid and can_ask:
                # Both sides active
                # 1. Coarse Grid Search to find approximate (b0, a0)
                PB_2d = PB_coarse[:, None]
                PA_2d = PA_coarse[None, :]
                B_2d = b_coarse[:, None]
                A_2d = a_coarse[None, :]
                
                # Objective function components
                # J = (1-PB)(1-PA)V_q + PB(1-PA)(B + V_qp) + (1-PB)PA(A + V_qm) + PB PA (B + A + V_q) - HC
                # Simplify:
                # Terms with V_q: (1-PB)(1-PA)V_q + PB PA V_q = V_q [ (1-PB)(1-PA) + PB PA ]
                # Terms with V_qp: PB(1-PA)V_qp
                # Terms with V_qm: (1-PB)PA V_qm
                # Terms with B: PB(1-PA)B + PB PA B = PB B
                # Terms with A: (1-PB)PA A + PB PA A = PA A
                # So J = PB B + PA A + V_q [ (1-PB)(1-PA) + PB PA ] + V_qp PB(1-PA) + V_qm (1-PB)PA - HC
                
                obj_grid = (PB_2d * B_2d + PA_2d * A_2d 
                           + V_next_q * ((1-PB_2d)*(1-PA_2d) + PB_2d*PA_2d)
                           + V_next_q_plus * PB_2d*(1-PA_2d)
                           + V_next_q_minus * (1-PB_2d)*PA_2d
                           - hold_cost)
                
                flat_idx = np.argmax(obj_grid)
                b0 = b_coarse[flat_idx // n_coarse]
                a0 = a_coarse[flat_idx % n_coarse]
                
                # 2. Local Refinement using Newton's method / Root finding
                # We optimize b and a alternately.
                # Define helper to compute optimal b given a
                def neg_obj_b(b, a_fixed):
                    # Calculate lambda_b, lambda_a
                    lb = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    la = 1.0 - np.exp(-A * np.exp(-k * a_fixed) * dt)
                    # Objective
                    val = (lb * b + la * a_fixed 
                           + V_next_q * ((1-lb)*(1-la) + lb*la)
                           + V_next_q_plus * lb*(1-la)
                           + V_next_q_minus * (1-lb)*la
                           - hold_cost)
                    return -val # Negate for minimization

                def neg_obj_a(a, b_fixed):
                    lb = 1.0 - np.exp(-A * np.exp(-k * b_fixed) * dt)
                    la = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    val = (lb * b_fixed + la * a 
                           + V_next_q * ((1-lb)*(1-la) + lb*la)
                           + V_next_q_plus * lb*(1-la)
                           + V_next_q_minus * (1-lb)*la
                           - hold_cost)
                    return -val

                # Iterate refinement
                b_cur = b0
                a_cur = a0
                for _ in range(5): # 5 iterations of alternating optimization
                    # Optimize b
                    res_b = minimize_scalar(neg_obj_b, bounds=(0, max_depth), args=(a_cur,), method='bounded', options={'xatol': 1e-5})
                    b_cur = res_b.x
                    # Optimize a
                    res_a = minimize_scalar(neg_obj_a, bounds=(0, max_depth), args=(b_cur,), method='bounded', options={'xatol': 1e-5})
                    a_cur = res_a.x

                best_bid_depth = b_cur
                best_ask_depth = a_cur
                best_val = -neg_obj_b(b_cur, a_cur) # Recover objective value

            elif can_bid:
                # Only bid side active
                # J(b) = PB b + V_next_q (1-PB) + V_next_q_plus PB - hold_cost
                # J(b) = PB (b + V_next_q_plus) + V_next_q (1-PB) - hold_cost
                def neg_obj_b_only(b):
                    lb = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    val = lb * (b + V_next_q_plus) + V_next_q * (1-lb) - hold_cost
                    return -val
                res_b = minimize_scalar(neg_obj_b_only, bounds=(0, max_depth), method='bounded', options={'xatol': 1e-5})
                best_bid_depth = res_b.x
                best_ask_depth = 0.0
                best_val = -res_b.fun

            elif can_ask:
                # Only ask side active
                def neg_obj_a_only(a):
                    la = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    val = la * (a + V_next_q_minus) + V_next_q * (1-la) - hold_cost
                    return -val
                res_a = minimize_scalar(neg_obj_a_only, bounds=(0, max_depth), method='bounded', options={'xatol': 1e-5})
                best_ask_depth = res_a.x
                best_bid_depth = 0.0
                best_val = -res_a.fun

            else:
                best_bid_depth = 0.0
                best_ask_depth = 0.0
                best_val = -hold_cost + V_next_q
            
            V[n, q_idx] = best_val
            bid_policy[n, q_idx] = best_bid_depth
            ask_policy[n, q_idx] = best_ask_depth
            
    return bid_policy, ask_policy

def _get_policy_table(params):
    """Get or compute the policy table for the given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _POLICY_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) arrays of shape (4000,).
    Each element is the distance below/above mid to quote.
    np.inf means no quote.
    """
    # Determine step index from t
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = min(max(n, 0), params['steps'] - 1)
    
    Q = params['Q']
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Map inventory to index
    inv_idx = inventory + Q  # inventory in [-Q, Q], so inv_idx in [0, 2Q]
    
    # Clip to valid range
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[n, inv_idx].copy()
    ask_depths = ask_table[n, inv_idx].copy()
    
    # At q=Q, no bid fill possible -> set bid to inf
    # At q=-Q, no ask fill possible -> set ask to inf
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    # Ensure values are valid: >= 0, not NaN, not -inf
    # The problem states values below -3/k are invalid, but we output >= 0.
    # np.inf is allowed for "no quote".
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), np.inf, ask_depths)
    
    # Ensure shape is (4000,)
    if len(bid_depths) != 4000:
        if len(bid_depths) < 4000:
            pad = np.full(4000 - len(bid_depths), np.inf)
            bid_depths = np.concatenate([bid_depths, pad])
            ask_depths = np.concatenate([ask_depths, pad])
        else:
            bid_depths = bid_depths[:4000]
            ask_depths = ask_depths[:4000]
    
    return bid_depths, ask_depths
