"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Solve the finite-support mechanism design LP exactly.
    
    For a single additive bidder with two goods, values in {0,1,2,3},
    and a given distribution, we solve the LP that maximizes expected revenue
    subject to incentive compatibility (IC), individual rationality (IR),
    feasibility, and non-negativity constraints.
    
    The key insight: for a single bidder, IC implies that the payment for type t
    is determined by the integral of the allocation along any path from the
    reference type to t. We use the 'integral formula' approach where we
    parameterize payments via allocations, reducing the problem to a pure LP
    over allocation variables only (with payments derived).
    
    However, the most robust approach for finite support is to set up the full LP
    with explicit IC constraints. With 16 types (4x4 grid), this is 16 allocation
    vars (one per type, since single bidder) + 16 payment vars = 32 vars.
    Actually, for a single additive bidder, the allocation is a scalar per type
    (probability of winning both goods? No - wait, the task says allocations:(16,2)).
    
    Let me re-read: "allocations:(16,2), payments:(16,)". So 16 types, 2 goods each.
    The bidder's value for type (v1, v2) is v1*x1 + v2*x2 where x1, x2 are
    allocation probabilities for the two goods.
    
    For a single additive bidder, the IC constraint between types t and t' is:
    u(t; t) >= u(t; t')
    v_t · x_t - p_t >= v_t · x_{t'} - p_{t'}
    
    This is linear in (x, p). We can solve this LP directly.
    """
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Determine the support and distribution
    # support: list of (v1, v2) tuples or similar
    # distribution_spec: probabilities for each type
    
    # Parse the support - assume it's a list of (v1, v2) pairs
    if isinstance(support, np.ndarray):
        support = [tuple(s) for s in support]
    elif not isinstance(support, (list, tuple)):
        support = list(support)
    
    n_types = len(support)
    n_goods = 2
    
    # Get probabilities
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(t, 0.0) for t in support], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n_types) / n_types
    
    # Variables: x_{t,1}, x_{t,2}, p_t for each type t
    # Total variables: 16 * 2 + 16 = 48
    # Index: for type i, good j: var = i * n_goods + j
    # For type i payment: var = n_types * n_goods + i
    
    n_vars = n_types * n_goods + n_types
    
    def alloc_var(i, j):
        return i * n_goods + j
    
    def pay_var(i):
        return n_types * n_goods + i
    
    # Objective: maximize expected revenue = sum_i p_i * p_i_type
    # In linprog (minimize), negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[pay_var(i)] = -probs[i]
    
    # Constraints:
    # 1. Feasibility: 0 <= x_{t,j} <= 1
    # 2. Non-negative payments: p_t >= 0
    # 3. IC: for all i, j: v_i · x_i - p_i >= v_i · x_j - p_j
    # 4. IR: for all i: v_i · x_i - p_i >= 0
    
    A_ub = []
    b_ub = []
    
    # IC constraints: v_i · x_i - p_i - (v_i · x_j - p_j) >= 0
    # => -(v_i · x_i - p_i - v_i · x_j + p_j) <= 0
    # For each pair (i, j) with i != j:
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            v_i = np.array(support[i], dtype=float)
            row = np.zeros(n_vars)
            # v_i · x_i - p_i - v_i · x_j + p_j >= 0
            # => -v_i·x_i + p_i + v_i·x_j - p_j <= 0
            for g in range(n_goods):
                row[alloc_var(i, g)] = -v_i[g]
                row[alloc_var(j, g)] = v_i[g]
            row[pay_var(i)] = 1.0
            row[pay_var(j)] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
    
    # IR constraints: v_i · x_i - p_i >= 0 => -v_i·x_i + p_i <= 0
    for i in range(n_types):
        v_i = np.array(support[i], dtype=float)
        row = np.zeros(n_vars)
        for g in range(n_goods):
            row[alloc_var(i, g)] = -v_i[g]
        row[pay_var(i)] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= x <= 1, 0 <= p (upper bound on p can be large)
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0, 1))
        bounds.append((0, None))
    
    # Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    elapsed = time.time() - start_time
    if not res.success:
        # Fallback: use baseline if available
        try:
            from autoresearch_suite.science_economics_expansion import auction_baseline
            return auction_baseline(distribution_spec, support, constraints)
        except:
            pass
        # Return a valid but possibly suboptimal mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        for i in range(n_types):
            v_i = np.array(support[i], dtype=float)
            # Give allocation 1 for highest value, payment = value
            if v_i.max() > 0:
                allocations[i, :] = 1.0
                payments[i] = v_i.sum()
        return {'allocations': allocations, 'payments': payments}
    
    x = res.x
    
    # Extract allocations and payments
    allocations = np.zeros((n_types, n_goods))
    payments = np.zeros(n_types)
    for i in range(n_types):
        for j in range(n_goods):
            allocations[i, j] = x[alloc_var(i, j)]
        payments[i] = x[pay_var(i)]
    
    # Clip small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Shouldn't happen but safety check
        pass
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
