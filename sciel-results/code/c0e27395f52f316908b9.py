def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    from scipy.optimize import linprog
    import time
    
    deadline = time.time() + 80  # 80s safety margin for 90s limit
    
    # Values for each good: {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    # All possible types (v1, v2)
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Distribution: distribution_spec gives probabilities for each type
    # Assume distribution_spec is a list of 16 probabilities or similar
    # Let's handle various formats
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if probs.size == 16:
            probs = probs.reshape(16)
        else:
            # Try to normalize
            probs = probs / probs.sum()
    else:
        # Default uniform
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Support: which reports are allowed
    # If support is None, all 16 types are allowed reports
    if support is None:
        report_indices = list(range(n_types))
    elif isinstance(support, (list, tuple, np.ndarray)):
        report_indices = list(support)
    else:
        report_indices = list(range(n_types))
    
    n_reports = len(report_indices)
    
    # Variables in LP:
    # For each report r (n_reports) and each good g (2): allocation x_{r,g} ∈ [0,1]
    # For each report r: payment p_r
    # Total variables: 2*n_reports + n_reports = 3*n_reports
    
    # Objective: maximize expected revenue = sum_r probs[r] * p_r
    # In linprog form (minimize): minimize -sum_r probs[r] * p_r
    
    # Constraints:
    # 1. Individual rationality: for each type t, u(t,t) >= 0
    #    u(t,t) = sum_g x_{t,g} * t_g - p_t >= 0
    #    => p_t <= sum_g x_{t,g} * t_g
    #    => -sum_g x_{t,g} * t_g + p_t <= 0
    #
    # 2. Incentive compatibility: for each true type t and report r:
    #    u(t, r) >= u(t, t)
    #    sum_g x_{r,g} * t_g - p_r >= sum_g x_{t,g} * t_g - p_t
    #    => -sum_g x_{r,g} * t_g + p_r + sum_g x_{t,g} * t_g - p_t <= 0
    #    => sum_g (x_{t,g} - x_{r,g}) * t_g + p_r - p_t <= 0
    #
    # 3. Allocation bounds: 0 <= x_{r,g} <= 1
    # 4. Payment nonnegativity: p_r >= 0
    
    # Variable ordering: [x_{r0,g0}, x_{r0,g1}, p_{r0}, x_{r1,g0}, x_{r1,g1}, p_{r1}, ...]
    # Index: for report i, good g: 3*i + g
    # For report i, payment: 3*i + 2
    
    def var_x(report_idx, good_idx):
        return 3 * report_idx + good_idx
    
    def var_p(report_idx):
        return 3 * report_idx + 2
    
    n_vars = 3 * n_reports
    
    # Objective: minimize -sum_i probs[report_indices[i]] * p_i
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[var_p(i)] = -probs[report_indices[i]]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints: for each type t that is in report_indices
    for i in range(n_reports):
        t = types[report_indices[i]]
        row = np.zeros(n_vars)
        # p_t <= x_{t,0}*t_0 + x_{t,1}*t_1
        # -x_{t,0}*t_0 - x_{t,1}*t_1 + p_t <= 0
        row[var_x(i, 0)] = -t[0]
        row[var_x(i, 1)] = -t[1]
        row[var_p(i)] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC constraints: for each true type t (in report_indices) and report r (in report_indices)
    for i in range(n_reports):
        t = types[report_indices[i]]
        for j in range(n_reports):
            r = types[report_indices[j]]
            row = np.zeros(n_vars)
            # sum_g (x_{t,g} - x_{r,g}) * t_g + p_r - p_t <= 0
            row[var_x(i, 0)] = t[0]
            row[var_x(i, 1)] = t[1]
            row[var_x(j, 0)] = -t[0]
            row[var_x(j, 1)] = -t[1]
            row[var_p(j)] = 1
            row[var_p(i)] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows) if A_ub_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_ub_vals) if b_ub_vals else np.zeros(0)
    
    # Bounds: 0 <= x <= 1, p >= 0
    bounds = []
    for i in range(n_reports):
        bounds.append((0, 1))  # x_{i,0}
        bounds.append((0, 1))  # x_{i,1}
        bounds.append((0, None))  # p_i
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.status != 0:
        # Fallback: uniform allocation, zero payments
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(n_reports):
            t_idx = report_indices[i]
            allocations[t_idx, 0] = 1.0
            allocations[t_idx, 1] = 1.0
        return {"allocations": allocations, "payments": payments}
    
    x = result.x
    
    # Extract allocations and payments for all 16 types
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i in range(n_reports):
        t_idx = report_indices[i]
        allocations[t_idx, 0] = x[var_x(i, 0)]
        allocations[t_idx, 1] = x[var_x(i, 1)]
        payments[t_idx] = x[var_p(i)]
    
    # Clip small negative values
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    return {"allocations": allocations, "payments": payments}
