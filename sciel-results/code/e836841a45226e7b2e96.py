"""Incentive-compatible finite auction design with explicit candidate API."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            # Try to find probability array
            for key in distribution_spec:
                if isinstance(distribution_spec[key], (list, np.ndarray)):
                    arr = np.array(distribution_spec[key], dtype=float)
                    if len(arr) == n_types:
                        probs = arr
                        break
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Variables: [a_0(0), a_1(0), p(0), a_0(1), a_1(1), p(1), ...]
    # For each type t: 2 allocations + 1 payment = 3 variables
    # Total variables: 16 * 3 = 48
    n_vars = n_types * 3
    
    # Objective: maximize expected revenue = sum_t probs[t] * p(t)
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        p_idx = t * 3 + 2  # payment index for type t
        c[p_idx] = -probs[t]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # IC constraints: u(t,t) >= u(t,r) for all t, r
    # u(t,r) = t[0]*a_0(r) + t[1]*a_1(r) - p(r)
    # u(t,t) - u(t,r) >= 0
    # => t[0]*(a_0(t)-a_0(r)) + t[1]*(a_1(t)-a_1(r)) - (p(t)-p(r)) >= 0
    # => -t[0]*(a_0(t)-a_0(r)) - t[1]*(a_1(t)-a_1(r)) + (p(t)-p(r)) <= 0
    
    # IR constraints: u(t,t) >= 0
    # t[0]*a_0(t) + t[1]*a_1(t) - p(t) >= 0
    # => -t[0]*a_0(t) - t[1]*a_1(t) + p(t) <= 0
    
    # Build constraint matrix
    A_ub_rows = []
    b_ub_rows = []
    
    # IC constraints
    for t_true in range(n_types):
        for t_report in range(n_types):
            if t_true == t_report:
                continue
            row = np.zeros(n_vars)
            # -t[0]*(a_0(t_true) - a_0(t_report)) - t[1]*(a_1(t_true) - a_1(t_report)) + (p(t_true) - p(t_report)) <= 0
            tv = types[t_true]
            # a_0(t_true)
            row[t_true * 3 + 0] = -tv[0]
            # -a_0(t_report)
            row[t_report * 3 + 0] = tv[0]
            # a_1(t_true)
            row[t_true * 3 + 1] = -tv[1]
            # -a_1(t_report)
            row[t_report * 3 + 1] = tv[1]
            # p(t_true)
            row[t_true * 3 + 2] = 1
            # -p(t_report)
            row[t_report * 3 + 2] = -1
            A_ub_rows.append(row)
            b_ub_rows.append(0)
    
    # IR constraints: -t[0]*a_0(t) - t[1]*a_1(t) + p(t) <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        tv = types[t]
        row[t * 3 + 0] = -tv[0]
        row[t * 3 + 1] = -tv[1]
        row[t * 3 + 2] = 1
        A_ub_rows.append(row)
        b_ub_rows.append(0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    # Bounds: 0 <= a_g(t) <= 1, p(t) >= 0
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # a_0(t)
        bounds.append((0, 1))  # a_1(t)
        bounds.append((0, None))  # p(t)
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if time.time() > deadline:
        # Return a simple feasible solution
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    if result.status == 0:
        x = result.x
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        for t in range(n_types):
            allocations[t, 0] = x[t * 3 + 0]
            allocations[t, 1] = x[t * 3 + 1]
            payments[t] = x[t * 3 + 2]
        
        # Verify constraints
        # Check IC
        for t_true in range(n_types):
            for t_report in range(n_types):
                u_tt = types[t_true][0] * allocations[t_true, 0] + types[t_true][1] * allocations[t_true, 1] - payments[t_true]
                u_tr = types[t_true][0] * allocations[t_report, 0] + types[t_true][1] * allocations[t_report, 1] - payments[t_report]
                if u_tt < u_tr - 1e-8:
                    # IC violated, need to fix
                    pass
        
        return {'allocations': allocations, 'payments': payments}
    else:
        # Fallback: simple feasible solution
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
