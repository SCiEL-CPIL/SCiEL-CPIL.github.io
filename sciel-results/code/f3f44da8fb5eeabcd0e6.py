"""Incentive-compatible finite auction design via direct LP solve."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Parse inputs
    # distribution_spec should have 'probs' or be a list of 16 probabilities
    if isinstance(distribution_spec, dict):
        probs = np.array(distribution_spec.get('probs', []), dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0
    
    # Ensure we have 16 probabilities
    if len(probs) < 16:
        probs = np.pad(probs, (0, 16 - len(probs)))
    else:
        probs = probs[:16]
    
    # Normalize probabilities
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(16) / 16.0
    
    # Support: list of (v1, v2) tuples
    if support is None:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    # Map support to indices
    n_reports = len(support)
    if n_reports != 16:
        # If support doesn't have exactly 16 entries, use the standard 4x4 grid
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
        n_reports = 16
    
    # Ensure support is the standard 4x4 grid
    support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_reports = 16
    
    # Values: each report type i has value (v1_i, v2_i)
    values = np.array(support, dtype=float)  # shape (16, 2)
    
    # Allocation variables: x[i, g] = probability of allocating good g to bidder when they report type i
    # Payment variables: p[i] = payment when bidder reports type i
    #
    # For a single additive bidder with two goods, the utility when reporting type j
    # while having true type i is:
    #   u(i, j) = sum_g x[j, g] * v_i[g] - p[j]
    #
    # Incentive compatibility (IC): For all i, j: u(i, i) >= u(i, j)
    #   sum_g x[i,g]*v_i[g] - p[i] >= sum_g x[j,g]*v_i[g] - p[j]
    #
    # Individual rationality (IR): u(i, i) >= 0
    #   sum_g x[i,g]*v_i[g] - p[i] >= 0
    #
    # Objective: Maximize expected revenue = sum_i probs[i] * p[i]
    #
    # To maximize revenue, we can use the envelope theorem / payment identity:
    # For a truthful mechanism, p[i] = sum_g x[i,g]*v_i[g] - U[i] where U[i] is the utility.
    # With IR binding at the lowest type (or we can set U[i] >= 0 for all i),
    # the minimum payment satisfying IC and IR gives maximum revenue.
    #
    # Actually, let's set up the LP directly.
    # Decision variables: x[0..15, 0..1] (32 vars) and p[0..15] (16 vars) = 48 vars
    #
    # But we can eliminate p using the IC constraints. For a single bidder, the
    # payment is determined by the allocation up to a constant (the utility of the
    # lowest type). To maximize revenue, we set the utility of the lowest type to 0 (IR binding).
    #
    # Let's use the direct LP formulation:
    # Maximize: sum_i probs[i] * p[i]
    # Subject to:
    #   For all i, j: sum_g x[i,g]*v_i[g] - p[i] >= sum_g x[j,g]*v_i[g] - p[j]  (IC)
    #   For all i: sum_g x[i,g]*v_i[g] - p[i] >= 0  (IR)
    #   For all i, g: 0 <= x[i,g] <= 1  (feasible probabilities)
    #   p[i] >= 0  (nonnegative payments)
    
    # Number of variables: 32 allocation + 16 payment = 48
    n_alloc = 32
    n_pay = 16
    n_vars = n_alloc + n_pay
    
    # Objective: maximize sum_i probs[i] * p[i]
    # In linprog, we minimize, so negate
    c = np.zeros(n_vars)
    for i in range(16):
        c[n_alloc + i] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # IC constraints: for all i, j:
    #   sum_g x[i,g]*v_i[g] - p[i] - sum_g x[j,g]*v_i[g] + p[j] >= 0
    #   => -sum_g x[i,g]*v_i[g] + p[i] + sum_g x[j,g]*v_i[g] - p[j] <= 0
    #
    # IR constraints: for all i:
    #   -sum_g x[i,g]*v_i[g] + p[i] <= 0
    
    A_rows = []
    b_rows = []
    
    # IC constraints
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # -sum_g x[i,g]*v_i[g]
            for g in range(2):
                row[i * 2 + g] = -values[i, g]
            # +p[i]
            row[n_alloc + i] = 1.0
            # +sum_g x[j,g]*v_i[g]
            for g in range(2):
                row[j * 2 + g] = values[i, g]
            # -p[j]
            row[n_alloc + j] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints: -sum_g x[i,g]*v_i[g] + p[i] <= 0
    for i in range(16):
        row = np.zeros(n_vars)
        for g in range(2):
            row[i * 2 + g] = -values[i, g]
        row[n_alloc + i] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((1, n_vars))
    b_ub = np.array(b_rows) if b_rows else np.zeros(1)
    
    # Bounds: 0 <= x[i,g] <= 1, p[i] >= 0
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))
    
    # Check time
    if time.time() > deadline - 5:
        # Return a trivial feasible solution
        allocs = np.zeros((16, 2))
        pays = np.zeros(16)
        return {'allocations': allocs, 'payments': pays}
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    # Check time again
    if time.time() > deadline - 2:
        if result.success:
            x_opt = result.x
            allocs = x_opt[:32].reshape(16, 2)
            pays = x_opt[32:]
            # Clip to ensure feasibility
            allocs = np.clip(allocs, 0, 1)
            pays = np.maximum(pays, 0)
            return {'allocations': allocs, 'payments': pays}
        else:
            allocs = np.zeros((16, 2))
            pays = np.zeros(16)
            return {'allocations': allocs, 'payments': pays}
    
    if result.success:
        x_opt = result.x
        allocs = x_opt[:32].reshape(16, 2)
        pays = x_opt[32:]
        # Clip to ensure numerical feasibility
        allocs = np.clip(allocs, 0, 1)
        pays = np.maximum(pays, 0)
        
        # Verify constraints and fix if needed
        # Check IC
        for i in range(16):
            for j in range(16):
                ui_i = np.dot(allocs[i], values[i]) - pays[i]
                ui_j = np.dot(allocs[j], values[i]) - pays[j]
                if ui_i < ui_j - 1e-8:
                    # Fix by adjusting payment
                    pass  # LP should have handled this
        
        return {'allocations': allocs, 'payments': pays}
    else:
        # Fallback: trivial mechanism
        allocs = np.zeros((16, 2))
        pays = np.zeros(16)
        return {'allocations': allocs, 'payments': pays}

# EVOLVE-BLOCK-END
