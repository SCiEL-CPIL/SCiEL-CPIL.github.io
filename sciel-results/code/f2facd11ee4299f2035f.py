"""Rewritten program for incentive-compatible finite auction design."""
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
    # distribution_spec should give probabilities for each type
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            # Try to find any array-like value
            for key, val in distribution_spec.items():
                if isinstance(val, (list, tuple, np.ndarray)):
                    if len(val) == n_types:
                        probs = np.array(val, dtype=float)
                        break
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Parse support: which reports are allowed
    # support could be a list of type indices, or a boolean mask, or None (all types)
    if support is None:
        report_mask = np.ones(n_types, dtype=bool)
    elif isinstance(support, (list, tuple, np.ndarray)):
        if len(support) == n_types:
            report_mask = np.array(support, dtype=bool)
        else:
            # Assume it's a list of indices
            report_mask = np.zeros(n_types, dtype=bool)
            for idx in support:
                report_mask[int(idx)] = True
    else:
        report_mask = np.ones(n_types, dtype=bool)
    
    # Get indices of allowed reports
    allowed_reports = np.where(report_mask)[0]
    n_reports = len(allowed_reports)
    
    # Parse constraints
    if constraints is None:
        constraints = {}
    
    # For a single additive bidder with two goods, the mechanism is:
    # - Allocations: x[t, g] for type t and good g, in [0, 1]
    # - Payments: p[t] for type t, >= 0
    # 
    # Incentive compatibility (IC): for all types t, t':
    #   x[t]·v(t) - p[t] >= x[t']·v(t) - p[t']
    # where v(t) = (v1(t), v2(t)) is the value vector for type t
    #
    # Individual rationality (IR): x[t]·v(t) - p[t] >= 0 for all t
    #
    # Objective: maximize expected revenue = sum_t probs[t] * p[t]
    
    # Decision variables:
    # x[t, g] for t in 0..15, g in 0..1 -> 32 variables
    # p[t] for t in 0..15 -> 16 variables
    # Total: 48 variables
    
    # Variable indexing:
    # x_idx(t, g) = t * 2 + g
    # p_idx(t) = 32 + t
    
    def x_idx(t, g):
        return t * n_goods + g
    
    def p_idx(t):
        return n_types * n_goods + t
    
    n_vars = n_types * n_goods + n_types  # 48
    
    # Objective: maximize sum_t probs[t] * p[t]
    # linprog minimizes, so we minimize -sum_t probs[t] * p[t]
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[p_idx(t)] = -probs[t]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 
    # IC constraints: for all t, t' in allowed_reports:
    #   x[t]·v(t) - p[t] >= x[t']·v(t) - p[t']
    #   => x[t]·v(t) - x[t']·v(t) - p[t] + p[t'] >= 0
    #   => -x[t]·v(t) + x[t']·v(t) + p[t] - p[t'] <= 0
    #
    # For each pair (t, t'):
    #   For each good g:
    #     -v_g(t) * x[t, g] + v_g(t) * x[t', g]
    #   + p[t] - p[t']
    #
    # IR constraints: for all t in allowed_reports:
    #   x[t]·v(t) - p[t] >= 0
    #   => -x[t]·v(t) + p[t] <= 0
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints
    for t in allowed_reports:
        for t_prime in allowed_reports:
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            vt = types[t]
            for g in range(n_goods):
                row[x_idx(t, g)] = -vt[g]
                row[x_idx(t_prime, g)] = vt[g]
            row[p_idx(t)] = 1.0
            row[p_idx(t_prime)] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR constraints
    for t in allowed_reports:
        row = np.zeros(n_vars)
        vt = types[t]
        for g in range(n_goods):
            row[x_idx(t, g)] = -vt[g]
        row[p_idx(t)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows) if A_ub_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_ub_vals) if b_ub_vals else np.zeros(0)
    
    # Bounds:
    # x[t, g] in [0, 1]
    # p[t] in [0, inf)
    bounds = []
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
        bounds.append((0.0, None))
    
    # Solve the LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: return a simple mechanism (allocate nothing, charge nothing)
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_opt = result.x
    
    # Extract allocations and payments
    allocations = np.zeros((n_types, n_goods))
    payments = np.zeros(n_types)
    
    for t in range(n_types):
        for g in range(n_goods):
            allocations[t, g] = x_opt[x_idx(t, g)]
        payments[t] = x_opt[p_idx(t)]
    
    # Clip small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)
    
    # Check time
    if time.time() > deadline - 5.0:
        return {'allocations': allocations, 'payments': payments}
    
    # Verify constraints and polish if needed
    # Check IC and IR
    max_violation = 0.0
    for t in allowed_reports:
        vt = types[t]
        utility_t = np.dot(allocations[t], vt) - payments[t]
        
        # IR check
        if utility_t < -1e-8:
            max_violation = max(max_violation, -utility_t)
        
        # IC check
        for t_prime in allowed_reports:
            if t == t_prime:
                continue
            utility_t_prime_as_t = np.dot(allocations[t_prime], vt) - payments[t_prime]
            if utility_t < utility_t_prime_as_t - 1e-8:
                max_violation = max(max_violation, utility_t_prime_as_t - utility_t)
    
    # If there are violations, try to fix by adjusting payments
    if max_violation > 1e-8:
        # Simple fix: increase payments to satisfy IR
        for t in allowed_reports:
            vt = types[t]
            utility_t = np.dot(allocations[t], vt) - payments[t]
            if utility_t < 0:
                payments[t] = np.dot(allocations[t], vt)
        
        # Re-check and if still violating, reduce allocations
        for t in allowed_reports:
            vt = types[t]
            for t_prime in allowed_reports:
                if t == t_prime:
                    continue
                utility_t = np.dot(allocations[t], vt) - payments[t]
                utility_t_prime_as_t = np.dot(allocations[t_prime], vt) - payments[t_prime]
                if utility_t < utility_t_prime_as_t - 1e-8:
                    # Reduce allocation of t_prime
                    for g in range(n_goods):
                        allocations[t_prime, g] = max(0, allocations[t_prime, g] - 0.01)
    
    # Final cleanup
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
