import numpy as np
from itertools import product
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    start_time = time.time()
    timeout = 85.0  # Safety margin before 90s kill

    # Parse support
    # Default to {0, 1, 2, 3} if not provided or invalid
    val_list = [0, 1, 2, 3]
    if support is not None:
        if isinstance(support, (list, tuple, np.ndarray)):
            if len(support) == 2:
                # Assume support is [values_good1, values_good2]
                # Or a flat list?
                # Task says: "two goods, each value in {0,1,2,3}"
                # So support might be [[0,1,2,3], [0,1,2,3]] or similar.
                # Let's try to interpret robustly.
                try:
                    if isinstance(support[0], (list, tuple, np.ndarray)):
                        v1s = np.unique(np.array(support[0]))
                        v2s = np.unique(np.array(support[1]))
                        val_list = [v1s.tolist(), v2s.tolist()]
                    else:
                        # Maybe a single list for both?
                        vals = np.unique(np.array(support))
                        val_list = [vals.tolist(), vals.tolist()]
                except Exception:
                    pass
            elif isinstance(support, dict):
                # e.g., {'good1': [0,1,2,3], 'good2': [0,1,2,3]}
                pass # Fallback to default for now if dict is complex

    # If val_list is still default, ensure it's lists
    if not isinstance(val_list[0], list):
        val_list = [val_list, val_list]

    types = [(i, j) for i in val_list[0] for j in val_list[1]]
    n_types = len(types)
    
    # If n_types != 16, the problem structure is different.
    # The task specifies (16, 2) output.
    # If support changes, output shape might change?
    # "Returns: dict with 'allocations': (16, 2) array"
    # This implies the support is FIXED to {0,1,2,3} x {0,1,2,3}.
    # So we can safely assume n_types=16 and types are fixed.
    # I will revert to hardcoded types to ensure shape compliance.
    
    values = [0, 1, 2, 3]
    types = [(i, j) for i in values for j in values]
    n_types = 16
    
    # Get distribution
    probs = None
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            elif 'values' in distribution_spec and 'probs' in distribution_spec:
                vals = np.array(distribution_spec['values'])
                ps = np.array(distribution_spec['probs'])
                if len(vals) == 16 and len(ps) == 16:
                    probs = ps
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            probs = np.ones(16) / 16.0
        else:
            probs = probs / probs.sum()

    # Index mapping
    type_to_vals = {i: t for i, t in enumerate(types)}

    # LP Variables:
    # x1 (16), x2 (16), p (16) -> 48 vars
    n = 16
    c = np.zeros(3 * n)
    for t in range(n):
        c[32 + t] = -probs[t]  # Maximize sum p(t)*probs[t] -> Minimize -sum

    A_rows = []
    b_rows = []
    
    # IC Constraints:
    # p(t) - p(t') <= v(t).x(t) - v(t).x(t')
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(n):
            if t == t2:
                continue
            row = np.zeros(3 * n)
            row[0 + t] = -v1_t
            row[16 + t] = -v2_t
            row[0 + t2] = v1_t
            row[16 + t2] = v2_t
            row[32 + t] = 1
            row[32 + t2] = -1
            A_rows.append(row)
            b_rows.append(0)
            
    # IR Constraints:
    # p(t) <= v(t).x(t)
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        row = np.zeros(3 * n)
        row[0 + t] = -v1_t
        row[16 + t] = -v2_t
        row[32 + t] = 1
        A_rows.append(row)
        b_rows.append(0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x1
    for _ in range(n):
        bounds.append((0, 1)) # x2
    for _ in range(n):
        bounds.append((0, None)) # p

    # Solve LP with tighter tolerances
    options = {
        'presolve': True,
        'dual_feasibility_tolerance': 1e-12,
        'primal_feasibility_tolerance': 1e-12,
        'dual_gap_tolerance': 1e-12
    }
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception:
        result = None

    if result is None or not result.success:
        # Fallback: Uniform allocation, zero payments
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}

    x = result.x
    x1 = x[0:16]
    x2 = x[16:32]
    p = x[32:48]
    
    allocations = np.column_stack((x1, x2))
    payments = p
    
    # Clip to ensure validity
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Repair step: Ensure IC and IR strictly with a small margin
    # IR: p(t) <= v(t).x(t)
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        if payments[t] > val_t:
            payments[t] = val_t - 1e-12

    # IC: p(t) - p(t') <= v(t).x(t) - v(t).x(t')
    # If violated, lower p(t)
    # Note: Lowering p(t) helps IC for (t, t') but hurts IC for (t', t)?
    # IC(t', t): p(t') - p(t) <= v(t').x(t') - v(t').x(t)
    # If we lower p(t), LHS increases. This might violate IC(t', t).
    # So we need to be careful.
    # However, the LP solution is a global optimum for the convex problem.
    # Any violation is due to numerical error.
    # The "repair" by lowering p(t) might break other constraints.
    # A safer repair: Scale down payments slightly?
    # Or, since the violation is < 1e-12, and tolerance is 1e-8,
    # we might not need to repair if we trust the solver.
    # But to be safe, let's check max violation.
    
    max_violation = 0
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(n):
            if t == t2: continue
            v1_t2, v2_t2 = type_to_vals[t2]
            lhs = payments[t] - payments[t2]
            rhs = v1_t * (allocations[t, 0] - allocations[t2, 0]) + v2_t * (allocations[t, 1] - allocations[t2, 1])
            if lhs > rhs:
                max_violation = max(max_violation, lhs - rhs)
                
    if max_violation > 1e-10:
        # Significant violation, likely due to fallback or solver issue.
        # Fallback to safe mechanism
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
        
    return {'allocations': allocations, 'payments': payments}
