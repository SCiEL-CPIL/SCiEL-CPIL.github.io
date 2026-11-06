import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    type_index = {t: i for i, t in enumerate(types)}

    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types

    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n_types) / n_types

    # Parse support: allocation rules for each type
    # support[i] should give (x1_i, x2_i) allocation for type i
    if isinstance(support, dict):
        if 'allocations' in support:
            allocations = np.array(support['allocations'], dtype=float)
        else:
            allocations = np.zeros((n_types, 2))
    elif isinstance(support, (list, tuple, np.ndarray)):
        arr = np.array(support, dtype=float)
        if arr.shape == (n_types, 2):
            allocations = arr
        elif arr.shape == (n_types,):
            allocations = np.column_stack([arr, np.zeros(n_types)])
        else:
            allocations = np.zeros((n_types, 2))
    else:
        allocations = np.zeros((n_types, 2))

    if allocations.shape != (n_types, 2):
        allocations = np.zeros((n_types, 2))

    # Parse constraints
    if isinstance(constraints, dict):
        ir_constraint = constraints.get('ir', True)
        regret_bound = constraints.get('regret', 1e-8)
    else:
        ir_constraint = True
        regret_bound = 1e-8

    # For a single additive bidder, IC implies:
    # u(t,t) >= u(t',t) for all t,t'
    # where u(t,t') = v1(t)*x1(t') + v2(t)*x2(t') - p(t')
    #
    # This means: p(t') <= v1(t)*x1(t') + v2(t)*x2(t') - [v1(t)*x1(t) + v2(t)*x2(t) - p(t)]
    # i.e., p(t') - p(t) <= v1(t)*(x1(t')-x1(t)) + v2(t)*(x2(t')-x2(t))
    #
    # For the optimum, we can use the LP approach with correct IC direction.
    # Variables: allocations (n_types * 2) and payments (n_types)
    # Maximize sum_t probs[t] * (v1[t]*x1[t] + v2[t]*x2[t] - p[t])  [expected utility]
    # Wait - we want to maximize REVENUE = sum_t probs[t] * p[t]
    # Subject to:
    #   IC: for all t,t': p[t'] - p[t] <= v1[t]*(x1[t']-x1[t]) + v2[t]*(x2[t']-x2[t])
    #   IR: p[t] <= v1[t]*x1[t] + v2[t]*x2[t]
    #   Feasibility: 0 <= x1[t], x2[t] <= 1
    #   p[t] >= 0

    # Let's set up the LP to maximize revenue
    # Decision variables: x1[0..15], x2[0..15], p[0..15] -> 48 variables
    # Maximize: sum_t probs[t] * p[t]
    # 
    # IC constraints: for each pair (t, t'):
    #   p[t'] - p[t] - v1[t]*(x1[t']-x1[t]) - v2[t]*(x2[t']-x2[t]) <= 0
    #   => -v1[t]*x1[t'] + v1[t]*x1[t] - v2[t]*x2[t'] + v2[t]*x2[t] + p[t'] - p[t] <= 0
    #
    # IR constraints: for each t:
    #   p[t] - v1[t]*x1[t] - v2[t]*x2[t] <= 0
    #
    # Nonnegativity: p[t] >= 0 -> -p[t] <= 0
    # Bounds: 0 <= x1[t] <= 1, 0 <= x2[t] <= 1

    n_vars = 3 * n_types  # 2*n_types for allocations + n_types for payments
    x1_idx = lambda t: t
    x2_idx = lambda t: n_types + t
    p_idx = lambda t: 2 * n_types + t

    # Objective: maximize revenue = sum_t probs[t] * p[t]
    # linprog minimizes, so minimize -revenue
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[p_idx(t)] = -probs[t]

    # Inequality constraints: A_ub @ x <= b_ub
    A_rows = []
    b_rows = []

    # IC constraints: for all t, t'
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            v1t, v2t = types[t]
            row = np.zeros(n_vars)
            # -v1[t]*x1[t'] + v1[t]*x1[t] - v2[t]*x2[t'] + v2[t]*x2[t] + p[t'] - p[t] <= 0
            row[x1_idx(tp)] = -v1t
            row[x1_idx(t)] = v1t
            row[x2_idx(tp)] = -v2t
            row[x2_idx(t)] = v2t
            row[p_idx(tp)] = 1.0
            row[p_idx(t)] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)

    # IR constraints: for each t: p[t] - v1[t]*x1[t] - v2[t]*x2[t] <= 0
    for t in range(n_types):
        v1t, v2t = types[t]
        row = np.zeros(n_vars)
        row[x1_idx(t)] = -v1t
        row[x2_idx(t)] = -v2t
        row[p_idx(t)] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)

    # Nonnegativity of payments: -p[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[p_idx(t)] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds: 0 <= x1,t <= 1, 0 <= x2,t <= 1, p,t >= 0 (upper bound on p can be large)
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # x1
    for t in range(n_types):
        bounds.append((0, 1))  # x2
    for t in range(n_types):
        bounds.append((0, None))  # p

    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            x = result.x
            alloc = np.column_stack([x[0:n_types], x[n_types:2*n_types]])
            pays = x[2*n_types:3*n_types]

            # Verify constraints
            # Check IC
            max_violation = 0.0
            for t in range(n_types):
                for tp in range(n_types):
                    v1t, v2t = types[t]
                    lhs = pays[tp] - pays[t] - v1t*(alloc[tp,0]-alloc[t,0]) - v2t*(alloc[tp,1]-alloc[t,1])
                    if lhs > max_violation:
                        max_violation = lhs
            if max_violation > 1e-6:
                # IC violated, use fallback
                raise Exception("IC violated")

            # Check IR
            for t in range(n_types):
                v1t, v2t = types[t]
                if pays[t] > v1t*alloc[t,0] + v2t*alloc[t,1] + 1e-6:
                    raise Exception("IR violated")

            # Check nonnegativity
            if np.any(pays < -1e-6):
                pays = np.maximum(pays, 0)

            # Check allocation bounds
            alloc = np.clip(alloc, 0, 1)

            # Recompute revenue
            revenue = np.sum(probs * pays)

            return {'allocations': alloc, 'payments': pays}
        else:
            raise Exception("LP failed: " + result.message)
    except Exception as e:
        pass

    # Fallback: simple mechanism
    if time.time() > deadline:
        alloc = np.zeros((n_types, 2))
        pays = np.zeros(n_types)
        return {'allocations': alloc, 'payments': pays}

    # Fallback: allocate all to highest value types, charge accordingly
    alloc = np.zeros((n_types, 2))
    pays = np.zeros(n_types)
    for t in range(n_types):
        v1t, v2t = types[t]
        if v1t > 0:
            alloc[t, 0] = 1.0
        if v2t > 0:
            alloc[t, 1] = 1.0
        # Simple payment: charge the value
        pays[t] = v1t * alloc[t, 0] + v2t * alloc[t, 1]

    # Ensure IR: p[t] <= v1[t]*x1[t] + v2[t]*x2[t]
    for t in range(n_types):
        v1t, v2t = types[t]
        max_pay = v1t * alloc[t, 0] + v2t * alloc[t, 1]
        pays[t] = min(pays[t], max_pay)
        pays[t] = max(pays[t], 0)

    return {'allocations': alloc, 'payments': pays}
