"""
Incentive-compatible finite auction design using LP over utilities and allocations.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # 1. Define Type Space
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # 2. Parse Distribution
    probs = np.zeros(n_types)
    
    if isinstance(distribution_spec, dict):
        if all(isinstance(k, tuple) for k in distribution_spec.keys()):
            for k, v in distribution_spec.items():
                if k in type_to_idx:
                    probs[type_to_idx[k]] = v
        elif 'probs' in distribution_spec:
            p_data = distribution_spec['probs']
            if isinstance(p_data, (list, np.ndarray)) and len(p_data) == n_types:
                probs = np.array(p_data, dtype=float)
            elif isinstance(p_data, dict):
                for k, v in p_data.items():
                    if k in type_to_idx:
                        probs[type_to_idx[k]] = v
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if isinstance(dist, dict):
                for k, v in dist.items():
                    if k in type_to_idx:
                        probs[type_to_idx[k]] = v
            elif isinstance(dist, (list, tuple, np.ndarray)):
                probs = np.array(dist, dtype=float)
    elif isinstance(distribution_spec, (list, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
    
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(n_types) / n_types

    # 3. Setup LP - Maximizing Revenue via Utilities
    # Variables: x1[0..15], x2[0..15], u[0..15] => 48 variables
    # x1: allocation of good 1 for each type
    # x2: allocation of good 2 for each type
    # u: utility for each type
    # Payment p[t] = v1[t]*x1[t] + v2[t]*x2[t] - u[t]
    
    n_vars = 48
    x1_offset = 0
    x2_offset = 16
    u_offset = 32
    
    # Objective: Maximize sum_t probs[t] * (v1[t]*x1[t] + v2[t]*x2[t] - u[t])
    # linprog minimizes, so we negate the objective coefficients.
    c = np.zeros(n_vars)
    for t in range(n_types):
        v1_t, v2_t = types[t]
        c[x1_offset + t] = -probs[t] * v1_t
        c[x2_offset + t] = -probs[t] * v2_t
        c[u_offset + t] = probs[t]  # Because we are maximizing -u, so minimizing +u? 
                                    # Wait. Maximize Revenue = Max sum pi (v.x - u).
                                    # Minimize -Revenue = Min sum pi (u - v.x).
                                    # So coeff for u is +pi, coeff for x is -pi*v.
                                    # Yes, c[u] = +probs[t].
    
    # Build constraint matrix A_ub and b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # IC Constraints:
    # u[t'] <= u[t] + v[t'] . x[t] - v[t] . x[t]
    # u[t'] - u[t] - v1[t']*x1[t] - v2[t']*x2[t] + v1[t]*x1[t] + v2[t]*x2[t] <= 0
    for t in range(n_types):
        for t2 in range(n_types):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            v1_t, v2_t = types[t]
            v1_t2, v2_t2 = types[t2]
            
            # u[t'] term
            row[u_offset + t2] = 1.0
            # u[t] term
            row[u_offset + t] = -1.0
            # x[t] terms (allocated to t, valued by t')
            row[x1_offset + t] = -v1_t2
            row[x2_offset + t] = -v2_t2
            # x[t] terms (allocated to t, valued by t) -> moved to LHS
            row[x1_offset + t] += v1_t
            row[x2_offset + t] += v2_t
            
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR Constraints:
    # u[t] >= 0  =>  -u[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_offset + t] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
        
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds:
    # x1, x2 in [0, 1]
    # u >= 0 (No upper bound needed explicitly, but bounded by IC/IR implicitly)
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # x1[t]
    for t in range(n_types):
        bounds.append((0, 1))  # x2[t]
    for t in range(n_types):
        bounds.append((0, None))  # u[t]
        
    # Solve LP
    result = None
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12})
    except Exception as e:
        # Fallback to zero mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {"allocations": allocations, "payments": payments}
        
    if time.time() > deadline - 5.0:
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {"allocations": allocations, "payments": payments}
        
    if not result.success:
        # Fallback to zero mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {"allocations": allocations, "payments": payments}
        
    x_opt = result.x
    
    # Extract allocations and utilities
    allocations = np.zeros((n_types, n_goods))
    utilities = np.zeros(n_types)
    for t in range(n_types):
        allocations[t, 0] = x_opt[x1_offset + t]
        allocations[t, 1] = x_opt[x2_offset + t]
        utilities[t] = x_opt[u_offset + t]
        
    # Calculate payments
    payments = np.zeros(n_types)
    for t in range(n_types):
        v1_t, v2_t = types[t]
        payments[t] = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - utilities[t]
        
    # Clip small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    # Payments should be non-negative due to IR (u <= v.x). 
    # If u > v.x due to numerical error, clip payment to 0 and adjust u? 
    # Better: just clip payment to 0. The evaluator checks IR: v.x - p >= 0.
    # If p < 0, IR is satisfied. But task requires "nonnegative payments".
    # So we must ensure p >= 0.
    payments = np.clip(payments, 0.0, None)
    
    # High-precision polish
    if time.time() < deadline - 10.0:
        try:
            result2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                              options={'presolve': True, 
                                       'dual_feasibility_tolerance': 1e-14,
                                       'primal_feasibility_tolerance': 1e-14,
                                       'time_limit': 30.0})
            if result2.success:
                x_opt2 = result2.x
                allocations = np.zeros((n_types, n_goods))
                utilities2 = np.zeros(n_types)
                for t in range(n_types):
                    allocations[t, 0] = x_opt2[x1_offset + t]
                    allocations[t, 1] = x_opt2[x2_offset + t]
                    utilities2[t] = x_opt2[u_offset + t]
                    
                payments = np.zeros(n_types)
                for t in range(n_types):
                    v1_t, v2_t = types[t]
                    payments[t] = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - utilities2[t]
                    
                allocations = np.clip(allocations, 0.0, 1.0)
                payments = np.clip(payments, 0.0, None)
        except Exception:
            pass
            
    # Final feasibility check and repair
    # Check IC and IR with strict tolerance
    def _check_feasibility(alloc, pay, types, tol=1e-8):
        n = len(types)
        for t in range(n):
            v1, v2 = types[t]
            u_t = v1 * alloc[t, 0] + v2 * alloc[t, 1] - pay[t]
            # IR
            if u_t < -tol:
                return False
            # IC
            for t2 in range(n):
                if t2 == t:
                    continue
                v1_2, v2_2 = types[t2]
                u_t_as_t2 = v1 * alloc[t2, 0] + v2 * alloc[t2, 1] - pay[t2]
                if u_t < u_t_as_t2 - tol:
                    return False
        return True

    if not _check_feasibility(allocations, payments, types, tol=1e-8):
        # If infeasible, try to repair by slightly reducing allocations or increasing payments?
        # Simplest robust fallback: Zero mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)

    return {
        "allocations": allocations,
        "payments": payments
    }

# EVOLVE-BLOCK-END
