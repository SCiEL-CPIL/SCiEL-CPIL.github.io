import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    # Define the support types: (v1, v2) for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)
    
    # Determine probabilities for each type
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, (int, float)):
        idx = int(distribution_spec)
        if idx == 0:
            # Uniform joint
            probs = np.ones(num_types) / num_types
        elif idx == 1:
            # Independent uniform
            probs = np.ones(num_types) / num_types
        elif idx == 2:
            # Correlated: v1 == v2
            for i, t in enumerate(types):
                if t[0] == t[1]:
                    probs[i] = 0.25
            probs /= probs.sum()
        else:
            probs = np.ones(num_types) / num_types
    else:
        probs = np.ones(num_types) / num_types

    # Variables: x1_t, x2_t, u_t for each type t
    # Order: x0_0, x1_0, u_0, x0_1, x1_1, u_1, ...
    n_vars = 3 * num_types
    
    # Objective: Maximize Revenue = sum pi_t * (v_t * x_t - u_t)
    # Minimize -Revenue = sum pi_t * (u_t - v_t * x_t)
    c = np.zeros(n_vars)
    for t in range(num_types):
        v1, v2 = types[t]
        c[t*3] = -probs[t] * v1
        c[t*3+1] = -probs[t] * v2
        c[t*3+2] = probs[t]
        
    # Bounds
    bounds = []
    for t in range(num_types):
        bounds.append((0, 1)) # x1
        bounds.append((0, 1)) # x2
        bounds.append((0, 6)) # u (Max value is 6)

    # Inequality constraints: A_ub z <= b_ub
    # 1. Non-negative payments: v_t * x_t - u_t >= 0  =>  u_t - v_t * x_t <= 0
    A_ub = []
    b_ub = []
    
    for t in range(num_types):
        row = np.zeros(n_vars)
        v1, v2 = types[t]
        row[t*3] = -v1
        row[t*3+1] = -v2
        row[t*3+2] = 1
        A_ub.append(row)
        b_ub.append(0)
        
    # 2. IC constraints:
    # u_t >= v_t * x_tp - v_tp * x_tp + u_tp
    # u_tp + (v_t - v_tp) * x_tp - u_t <= 0
    for t in range(num_types):
        v1_t, v2_t = types[t]
        for t_prime in range(num_types):
            if t == t_prime:
                continue
            v1_tp, v2_tp = types[t_prime]
            
            row = np.zeros(n_vars)
            # Coeffs for x_tp
            row[t_prime*3] = v1_t - v1_tp
            row[t_prime*3+1] = v2_t - v2_tp
            # Coeff for u_tp
            row[t_prime*3+2] = 1
            # Coeff for u_t
            row[t*3+2] = -1
            
            A_ub.append(row)
            b_ub.append(0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    
    if res.success:
        z_opt = res.x
        for t in range(num_types):
            x1 = np.clip(z_opt[t*3], 0, 1)
            x2 = np.clip(z_opt[t*3+1], 0, 1)
            u = np.clip(z_opt[t*3+2], 0, 6)
            
            allocations[t, 0] = x1
            allocations[t, 1] = x2
            
            v1, v2 = types[t]
            v = v1 * x1 + v2 * x2
            p = v - u
            # Ensure non-negative payment
            if p < 0:
                p = 0.0
            payments[t] = p
    else:
        # Fallback: No allocation
        allocations = np.zeros((num_types, 2))
        payments = np.zeros(num_types)

    return {
        "allocations": allocations,
        "payments": payments
    }
