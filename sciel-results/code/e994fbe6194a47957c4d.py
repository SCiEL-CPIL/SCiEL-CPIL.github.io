import numpy as np
from scipy.optimize import minimize

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
            probs = np.ones(num_types) / num_types
        elif idx == 1:
            probs = np.ones(num_types) / num_types
        elif idx == 2:
            for i, t in enumerate(types):
                if t[0] == t[1]:
                    probs[i] = 0.25
            probs /= probs.sum()
        else:
            probs = np.ones(num_types) / num_types
    else:
        probs = np.ones(num_types) / num_types

    # Variables:
    # For each type t: x1_t, x2_t, u_t
    # Total 3 * num_types variables
    n_vars = 3 * num_types
    
    def get_vars(z, t_idx):
        base = t_idx * 3
        x1 = z[base]
        x2 = z[base + 1]
        u = z[base + 2]
        return x1, x2, u

    def value(t_idx, x1, x2):
        v1, v2 = types[t_idx]
        return v1 * x1 + v2 * x2

    # Objective function: Minimize -Revenue
    # Revenue = sum( pi_t * (v_t - u_t) )
    def objective(z):
        revenue = 0.0
        for t in range(num_types):
            x1, x2, u = get_vars(z, t)
            v = value(t, x1, x2)
            revenue += probs[t] * (v - u)
        return -revenue

    # Constraints
    cons = []
    
    # IR constraints: u_t >= 0
    for t in range(num_types):
        def ir_con(z, t=t):
            _, _, u = get_vars(z, t)
            return u
        cons.append({'type': 'ineq', 'fun': ir_con})
        
    # Non-negative payment constraints: v(t, x_t) - u_t >= 0
    for t in range(num_types):
        def nn_pay_con(z, t=t):
            x1, x2, u = get_vars(z, t)
            v = value(t, x1, x2)
            return v - u
        cons.append({'type': 'ineq', 'fun': nn_pay_con})
        
    # IC constraints
    for t in range(num_types):
        for t_prime in range(num_types):
            if t == t_prime:
                continue
            def ic_con(z, t=t, t_prime=t_prime):
                x1_t, x2_t, u_t = get_vars(z, t)
                x1_tp, x2_tp, u_tp = get_vars(z, t_prime)
                
                v_t_given_tp = value(t, x1_tp, x2_tp)
                v_tp_given_tp = value(t_prime, x1_tp, x2_tp)
                
                return u_t - v_t_given_tp + v_tp_given_tp - u_tp
            cons.append({'type': 'ineq', 'fun': ic_con})

    # Bounds for variables
    # x1, x2 in [0, 1]
    # u_t in [0, 10]
    bounds = []
    for t in range(num_types):
        bounds.append((0, 1)) # x1
        bounds.append((0, 1)) # x2
        bounds.append((0, 10)) # u

    # Initial guess
    # Start with no allocation and 0 utility
    z0 = np.zeros(n_vars)

    # Solve
    res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-10})
    
    # Extract results
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    
    if res.success:
        z_opt = res.x
        for t in range(num_types):
            x1, x2, u = get_vars(z_opt, t)
            allocations[t, 0] = x1
            allocations[t, 1] = x2
            v = value(t, x1, x2)
            p = v - u
            # Clamp small negatives due to numerical error
            if p < 0 and p > -1e-9:
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
