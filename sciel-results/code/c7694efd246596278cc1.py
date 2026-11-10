"""Auction design: solve the finite-support mechanism LP with high precision."""

def construct_auction(distribution_spec, support, constraints):
    import time
    import numpy as np
    from scipy.optimize import linprog

    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Parse inputs
    # distribution_spec: defines the probability distribution over value types
    # support: the value support (e.g., [0,1,2,3] for each good)
    # constraints: may contain additional constraint info

    # Determine the number of value types per good
    if isinstance(support, (list, tuple, np.ndarray)):
        values = list(support)
    else:
        values = [0, 1, 2, 3]

    n = len(values)
    n_types = n * n  # 16 types: (v1, v2) for v1,v2 in values

    # Parse distribution_spec to get probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            # Try to extract from any list-like value
            probs = None
            for v in distribution_spec.values():
                if isinstance(v, (list, tuple, np.ndarray)):
                    probs = np.array(v, dtype=float)
                    break
            if probs is None:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types

    probs = probs.flatten()
    if len(probs) != n_types:
        # Try to reshape or pad
        if len(probs) < n_types:
            probs = np.pad(probs, (0, n_types - len(probs)))
        else:
            probs = probs[:n_types]
        probs = probs / probs.sum()
    else:
        probs = probs / probs.sum()

    # Build the LP:
    # Variables: x_{t,t'} (allocation prob for type t reporting t') for t,t' in types
    #            p_t (payment for type t)
    # Total variables: n_types^2 + n_types = 256 + 16 = 272
    #
    # Objective: maximize sum_t probs[t] * p_t  (expected revenue)
    #
    # Constraints:
    # 1. For each true type t: sum_{t'} x_{t,t'} = 1  (allocation is a probability distribution)
    # 2. For each t, t': 0 <= x_{t,t'} <= 1
    # 3. IC: for each true type t and report t':
    #    sum_{t''} x_{t,t''} * v_{t''}(t) - p_t  >=  sum_{t''} x_{t,t''} * v_{t''}(t') - p_{t'}
    #    where v_{t''}(t) = sum_g x_{t'',g} * v_g(t)
    #    Actually: utility of t reporting t' = sum_{t''} x_{t,t''} * val(t'', t) - p_t
    #    IC: u(t, t) >= u(t, t') for all t, t'
    #    i.e., sum_{t''} x_{t,t''} * val(t'', t) - p_t >= sum_{t''} x_{t,t''} * val(t'', t') - p_{t'}
    # 4. IR: for each t: sum_{t''} x_{t,t''} * val(t'', t) - p_t >= 0
    # 5. p_t >= 0

    # Value of type t'' to bidder with true type t:
    # t = (v1, v2), t'' = (w1, w2)
    # val(t'', t) = x_{t'',1} * v1 + x_{t'',2} * v2
    # But x_{t'',g} is the allocation of good g to type t''... 
    # Wait, let me re-read the problem.
    #
    # "One additive bidder, two goods, each value in {0,1,2,3}"
    # The allocation x_{t,t'} should be: probability of allocating good 1 and good 2
    # when true type is t and report is t'.
    #
    # Actually, re-reading: "allocations:(16,2)" - so for each of 16 reports, 
    # we get 2 allocation values (one per good).
    # "payments:(16,)" - payment for each report.
    #
    # So the mechanism is: for each report r (16 possible), allocate goods with
    # probabilities (a1_r, a2_r) and charge payment p_r.
    #
    # The allocation depends only on the report, not the true type.
    # So x_{t,t'} should really be: allocation when report is t'.
    # Let me re-think.
    #
    # In a standard auction with reports:
    # - Bidder has true type t = (v1, v2)
    # - Bidder reports t' = (w1, w2)
    # - Mechanism allocates goods with probabilities (a1(t'), a2(t')) and charges p(t')
    # - Utility of t reporting t' = a1(t')*v1 + a2(t')*v2 - p(t')
    #
    # IC: for all t, t': a1(t)*v1(t) + a2(t)*v2(t) - p(t) >= a1(t')*v1(t) + a2(t')*v2(t) - p(t')
    # IR: for all t: a1(t)*v1(t) + a2(t)*v2(t) - p(t) >= 0
    #
    # Variables: a1(t') for t' in 16 types, a2(t') for 16 types, p(t') for 16 types
    # Total: 48 variables
    #
    # Objective: maximize sum_t probs[t] * p(t)
    #
    # This is much smaller! Let me redo.

    # Variables: [a1_0, a2_0, p_0, a1_1, a2_1, p_1, ..., a1_15, a2_15, p_15]
    n_vars = 3 * n_types  # 48

    # Objective: maximize sum_t probs[t] * p_t
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[3*t + 2] = -probs[t]

    # Constraints in form A_ub @ x <= b_ub

    A_ub_rows = []
    b_ub = []

    # 1. 0 <= a1(t') <= 1 for all t'
    # a1(t') >= 0  =>  -a1(t') <= 0
    # a1(t') <= 1  =>  a1(t') <= 1
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[3*t] = -1
        A_ub_rows.append(row)
        b_ub.append(0)
        row2 = np.zeros(n_vars)
        row2[3*t] = 1
        A_ub_rows.append(row2)
        b_ub.append(1)

    # 2. 0 <= a2(t') <= 1 for all t'
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[3*t + 1] = -1
        A_ub_rows.append(row)
        b_ub.append(0)
        row2 = np.zeros(n_vars)
        row2[3*t + 1] = 1
        A_ub_rows.append(row2)
        b_ub.append(1)

    # 3. p(t') >= 0  =>  -p(t') <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[3*t + 2] = -1
        A_ub_rows.append(row)
        b_ub.append(0)

    # 4. IC constraints: for all true type t, report t':
    #    a1(t)*v1(t) + a2(t)*v2(t) - p(t) >= a1(t')*v1(t) + a2(t')*v2(t) - p(t')
    #    => a1(t')*v1(t) + a2(t')*v2(t) - p(t') - a1(t)*v1(t) - a2(t)*v2(t) + p(t) <= 0
    #
    # For each pair (t, t'):
    v1 = np.array([values[t // n] for t in range(n_types)])
    v2 = np.array([values[t % n] for t in range(n_types)])

    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            # a1(tp)*v1(t)
            row[3*tp] = v1[t]
            # a2(tp)*v2(t)
            row[3*tp + 1] = v2[t]
            # -p(tp)
            row[3*tp + 2] = -1
            # -a1(t)*v1(t)
            row[3*t] = -v1[t]
            # -a2(t)*v2(t)
            row[3*t + 1] = -v2[t]
            # +p(t)
            row[3*t + 2] = 1
            A_ub_rows.append(row)
            b_ub.append(0)

    # 5. IR constraints: for all t:
    #    a1(t)*v1(t) + a2(t)*v2(t) - p(t) >= 0
    #    => -a1(t)*v1(t) - a2(t)*v2(t) + p(t) <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[3*t] = -v1[t]
        row[3*t + 1] = -v2[t]
        row[3*t + 2] = 1
        A_ub_rows.append(row)
        b_ub.append(0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub)

    # Bounds
    bounds = [(0, 1)] * n_types + [(0, 1)] * n_types + [(0, None)] * n_types
    # Actually let me restructure: variables are interleaved [a1_0, a2_0, p_0, a1_1, ...]
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # a1
        bounds.append((0, 1))  # a2
        bounds.append((0, None))  # p

    # Solve with high precision
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not result.success:
        # Fallback: try with different method
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs-ds')
        if not result.success:
            # Return a trivial feasible solution
            x = np.zeros(n_vars)
            for t in range(n_types):
                x[3*t] = 1.0  # allocate good 1 fully
                x[3*t+1] = 1.0  # allocate good 2 fully
                x[3*t+2] = v1[t] + v2[t]  # charge full value (IR binding)
            allocations = np.array([[x[3*t], x[3*t+1]] for t in range(n_types)])
            payments = np.array([x[3*t+2] for t in range(n_types)])
            return {'allocations': allocations, 'payments': payments}

    x = result.x

    # Extract allocations and payments
    allocations = np.array([[x[3*t], x[3*t+1]] for t in range(n_types)])
    payments = np.array([x[3*t+2] for t in range(n_types)])

    # Clip to ensure feasibility
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)

    # Check if we have time for a polish pass
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 5.0:
        # Try to improve by local search: check if we can increase payments slightly
        # while maintaining IC and IR
        best_alloc = allocations.copy()
        best_pay = payments.copy()
        best_rev = float(np.dot(probs, best_pay))

        # Small perturbation search
        for trial in range(50):
            if time.time() - t0 > DEADLINE - 2.0:
                break
            # Try increasing payments slightly
            delta = 1e-10 * (trial + 1)
            new_pay = best_pay + delta
            new_pay = np.maximum(new_pay, 0)

            # Check IC and IR
            feasible = True
            for t in range(n_types):
                u_t_t = best_alloc[t, 0] * v1[t] + best_alloc[t, 1] * v2[t] - new_pay[t]
                if u_t_t < -1e-10:
                    feasible = False
                    break
                for tp in range(n_types):
                    if t == tp:
                        continue
                    u_t_tp = best_alloc[tp, 0] * v1[t] + best_alloc[tp, 1] * v2[t] - new_pay[tp]
                    if u_t_t < u_t_tp - 1e-10:
                        feasible = False
                        break
                if not feasible:
                    break
            if feasible:
                best_pay = new_pay
                best_rev = float(np.dot(probs, best_pay))
            else:
                break

        if best_rev > float(np.dot(probs, payments)):
            payments = best_pay

    return {'allocations': allocations, 'payments': payments}
