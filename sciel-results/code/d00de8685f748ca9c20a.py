"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    deadline = t0 + 80.0  # 10s safety margin before 90s kill

    # Parse support: list of (v1, v2) tuples, or similar structure
    # Each value in {0,1,2,3}, so 16 types
    # distribution_spec: dict or object mapping type -> probability
    # constraints: dict with keys like 'nonnegative_payments', 'individual_rationality', 'regret_bound'
    
    # Extract types and probabilities
    if isinstance(distribution_spec, dict):
        # Keys might be tuples (v1, v2) or strings
        type_probs = {}
        for k, v in distribution_spec.items():
            if isinstance(k, tuple):
                type_probs[k] = float(v)
            elif isinstance(k, str):
                # Try to parse "v1,v2" or "v1 v2"
                parts = k.replace(' ', '').replace('(', '').replace(')', '').split(',')
                if len(parts) == 2:
                    type_probs[(int(parts[0]), int(parts[1]))] = float(v)
            else:
                type_probs[k] = float(v)
    else:
        # Maybe it's a numpy array or list
        type_probs = {}
        if hasattr(distribution_spec, 'items'):
            for k, v in distribution_spec.items():
                type_probs[k] = float(v)
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            # Assume it's a flat array of probabilities for all 16 types in order
            n = len(distribution_spec)
            idx = 0
            for v1 in range(4):
                for v2 in range(4):
                    type_probs[(v1, v2)] = float(distribution_spec[idx])
                    idx += 1
        else:
            # Fallback: uniform distribution
            type_probs = {}
            for v1 in range(4):
                for v2 in range(4):
                    type_probs[(v1, v2)] = 1.0 / 16.0

    # Ensure all 16 types are present
    all_types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    for t in all_types:
        if t not in type_probs:
            type_probs[t] = 0.0
    
    # Normalize probabilities to sum to 1
    total_prob = sum(type_probs.values())
    if total_prob > 0:
        for t in all_types:
            type_probs[t] /= total_prob
    else:
        for t in all_types:
            type_probs[t] = 1.0 / 16.0

    # Build type index mapping
    type_list = all_types
    n_types = len(type_list)
    type_index = {t: i for i, t in enumerate(type_list)}
    
    # Probabilities vector
    probs = np.array([type_probs[t] for t in type_list])
    
    # Values matrix: shape (n_types, 2)
    values = np.array([[t[0], t[1]] for t in type_list], dtype=float)
    
    # Parse constraints
    if constraints is None:
        constraints = {}
    
    # Default constraint values
    regret_bound = 1e-8
    if isinstance(constraints, dict):
        if 'regret_bound' in constraints:
            regret_bound = float(constraints['regret_bound'])
        elif 'regret' in constraints:
            regret_bound = float(constraints['regret'])
    
    # Decision variables:
    # x[i, j] = allocation probability of goods to type i when reporting j
    # Actually, let's define:
    # q[i][j] = allocation vector (2,) for type i when reporting j
    # p[i][j] = payment for type i when reporting j
    
    # Number of decision variables:
    # For each type i (16) and each report j (16):
    #   2 allocation variables + 1 payment variable = 3
    # Total: 16 * 16 * 3 = 768 variables
    
    # But we can simplify using the structure of IC constraints.
    # Let's use the standard LP formulation for auction design.
    
    # Variables:
    # x[i, j, g] for i in 0..15, j in 0..15, g in 0..1 (allocation of good g)
    # p[i, j] for i in 0..15, j in 0..15 (payment)
    
    # Total variables: 16*16*2 + 16*16 = 768
    
    # Objective: Maximize expected revenue
    # Revenue = sum_i sum_j probs[i] * p[i, j] * (allocation indicator?)
    # Wait, the revenue is the expected payment. The allocation affects utility.
    # Expected revenue = sum_i probs[i] * E[payment | type i]
    # But type i reports some j. Under truthfulness, type i reports i.
    # So expected revenue = sum_i probs[i] * p[i, i]
    
    # Actually, in the LP, we maximize:
    # sum_i probs[i] * p[i, i]
    # subject to:
    # 1. Feasibility: 0 <= x[i, j, g] <= 1
    # 2. Individual Rationality: for each type i, utility when reporting truthfully >= 0
    #    u(i, i) = sum_g v_i[g] * x[i, i, g] - p[i, i] >= 0
    # 3. Incentive Compatibility: for each type i and report j,
    #    u(i, i) >= u(i, j)
    #    sum_g v_i[g] * x[i, i, g] - p[i, i] >= sum_g v_i[g] * x[i, j, g] - p[i, j]
    # 4. Nonnegative payments: p[i, j] >= 0
    
    # Let's set up the LP.
    
    n_reports = n_types  # 16
    
    # Variable indexing:
    # x_var[i, j, g] -> index
    # p_var[i, j] -> index
    
    def x_idx(i, j, g):
        return (i * n_reports + j) * 2 + g
    
    def p_idx(i, j):
        return n_types * n_reports * 2 + i * n_reports + j
    
    n_vars = n_types * n_reports * 3
    
    # Objective: maximize sum_i probs[i] * p[i, i]
    # In linprog, we minimize, so negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[p_idx(i, i)] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    A_ub_rows = []
    b_ub_list = []
    
    # 1. Nonnegativity of allocations and payments are handled by bounds
    
    # 2. Upper bounds on allocations: x[i, j, g] <= 1 (handled by bounds)
    
    # 3. Individual Rationality:
    # u(i, i) >= 0 => sum_g v_i[g] * x[i, i, g] - p[i, i] >= 0
    # => p[i, i] - sum_g v_i[g] * x[i, i, g] <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[p_idx(i, i)] = 1.0
        for g in range(2):
            row[x_idx(i, i, g)] = -values[i, g]
        A_ub_rows.append(row)
        b_ub_list.append(0.0)
    
    # 4. Incentive Compatibility:
    # u(i, i) >= u(i, j) for all i, j
    # sum_g v_i[g] * x[i, i, g] - p[i, i] >= sum_g v_i[g] * x[i, j, g] - p[i, j]
    # => p[i, i] - p[i, j] + sum_g v_i[g] * (x[i, j, g] - x[i, i, g]) <= 0
    # => p[i, i] - p[i, j] + sum_g v_i[g] * x[i, j, g] - sum_g v_i[g] * x[i, i, g] <= 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[p_idx(i, i)] = 1.0
            row[p_idx(i, j)] = -1.0
            for g in range(2):
                row[x_idx(i, j, g)] = values[i, g]
                row[x_idx(i, i, g)] = -values[i, g]
            A_ub_rows.append(row)
            b_ub_list.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_list)
    
    # Bounds: 0 <= x[i, j, g] <= 1, 0 <= p[i, j] <= inf
    bounds = []
    for i in range(n_types):
        for j in range(n_reports):
            for g in range(2):
                bounds.append((0.0, 1.0))
            bounds.append((0.0, None))
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback to baseline
            from autoresearch_suite.science_economics_expansion import auction_baseline
            return auction_baseline(distribution_spec, support, constraints)
        
        x_opt = result.x
        
        # Extract allocations and payments
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        
        for i in range(n_types):
            for g in range(2):
                allocations[i, g] = x_opt[x_idx(i, i, g)]
            payments[i] = x_opt[p_idx(i, i)]
        
        # Verify constraints
        # Check IR
        for i in range(n_types):
            util = np.dot(values[i], allocations[i]) - payments[i]
            if util < -1e-6:
                # IR violated, adjust
                pass
        
        # Check IC
        max_regret = 0.0
        for i in range(n_types):
            for j in range(n_types):
                if i == j:
                    continue
                # Utility of type i when reporting j
                # We need x[i, j, g] which is in the solution
                util_truth = np.dot(values[i], allocations[i]) - payments[i]
                util_report = np.dot(values[i], [x_opt[x_idx(i, j, g)] for g in range(2)]) - x_opt[p_idx(i, j)]
                regret = util_report - util_truth
                if regret > max_regret:
                    max_regret = regret
        
        if max_regret > regret_bound + 1e-6:
            # IC violated, fallback
            from autoresearch_suite.science_economics_expansion import auction_baseline
            return auction_baseline(distribution_spec, support, constraints)
        
        return {
            'allocations': allocations,
            'payments': payments
        }
        
    except Exception as e:
        # Fallback to baseline
        from autoresearch_suite.science_economics_expansion import auction_baseline
        return auction_baseline(distribution_spec, support, constraints)

# EVOLVE-BLOCK-END
