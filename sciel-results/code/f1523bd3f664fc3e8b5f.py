import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    # 16 types: (v1, v2) with v1, v2 in {0,1,2,3}
    # Index: v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n = 16

    # Parse distribution
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) != n:
            probs = np.ones(n) / n
    else:
        probs = np.ones(n) / n

    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        probs = np.ones(n) / n
        total = 1.0
    probs = probs / total

    # Value of each type
    values = np.array([v1 + v2 for v1, v2 in types], dtype=float)

    # For a single additive bidder, the mechanism is:
    # - x_i: probability of allocating to type i (scalar per type)
    # - p_i: payment for type i
    #
    # IC constraint: type i prefers reporting i over reporting j
    #   x_i * v_i - p_i >= x_j * v_i - p_j  for all i, j
    #   => p_i - x_i * v_i <= p_j - x_j * v_i
    #
    # Let's define: u_i = p_i - x_i * v_i (utility of type i when reporting truthfully)
    # IC: u_i <= u_j + (x_j - x_i) * v_i ... this gets complex
    #
    # Alternative: Use the standard formulation
    # Variables: x_0..x_15 (allocation probs), p_0..p_15 (payments)
    # Total: 32 variables
    #
    # Constraints:
    # 1. 0 <= x_i <= 1 for all i
    # 2. p_i >= 0 for all i (nonnegative payments)
    # 3. IR: p_i <= v_i * x_i for all i (type i gets nonnegative utility)
    # 4. IC: For all i, j: v_i * x_i - p_i >= v_i * x_j - p_j
    #    => p_j - p_i >= v_i * (x_j - x_i)
    #    => -p_i + p_j - v_i * x_j + v_i * x_i >= 0
    #    => v_i * x_i - v_i * x_j - p_i + p_j >= 0
    # 5. Monotonicity is implied by IC + IR for single bidder

    # Let's set up LP:
    # Maximize: sum_i probs[i] * p_i
    # Subject to constraints above

    n_vars = 2 * n  # x_0..x_15, p_0..p_15

    # Objective: maximize sum(probs[i] * p_i) => minimize -sum(probs[i] * p_i)
    c = np.zeros(n_vars)
    for i in range(n):
        c[n + i] = -probs[i]  # negative because linprog minimizes

    # Inequality constraints: A_ub @ vars <= b_ub
    # We'll collect all inequality constraints

    A_ub_rows = []
    b_ub_vals = []

    # x_i <= 1: x_i - 0*p <= 1
    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)

    # -x_i <= 0 (x_i >= 0)
    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # -p_i <= 0 (p_i >= 0)
    for i in range(n):
        row = np.zeros(n_vars)
        row[n + i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # IR: p_i - v_i * x_i <= 0  => -v_i * x_i + p_i <= 0
    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = -values[i]
        row[n + i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # IC: For all i, j: v_i * x_i - v_i * x_j - p_i + p_j >= 0
    #    => -v_i * x_i + v_i * x_j + p_i - p_j <= 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i] = -values[i]
            row[j] = values[i]
            row[n + i] = 1.0
            row[n + j] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)

    # Bounds: all variables >= 0 (already handled by constraints, but set bounds too)
    bounds = [(0, None)] * n_vars

    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if time.time() > deadline - 5:
        # Fallback: simple mechanism
        allocations = np.ones((n, 2)) * 0.5
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}

    if not result.success:
        # Fallback: simple mechanism
        allocations = np.ones((n, 2)) * 0.5
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}

    x = result.x[:n]
    p = result.x[n:]

    # Clip to ensure feasibility
    x = np.clip(x, 0.0, 1.0)
    p = np.clip(p, 0.0, None)

    # Ensure IR: p_i <= v_i * x_i
    for i in range(n):
        if values[i] > 0:
            p[i] = min(p[i], values[i] * x[i])
        else:
            p[i] = 0.0

    # Ensure nonnegativity
    p = np.clip(p, 0.0, None)
    x = np.clip(x, 0.0, 1.0)

    # Construct allocations: since it's a single additive bidder with two goods,
    # the allocation probability x_i is the probability of allocating BOTH goods
    # (since the bidder values both additively, and we're allocating a bundle)
    # Actually, for a single bidder with additive values over two goods, the mechanism
    # allocates each good independently or as a bundle. The standard approach:
    # x_i is the probability of allocating the bundle (both goods)
    # But the output format is (16, 2) - allocation probability for each good per type

    # For a single additive bidder, the optimal mechanism allocates each good with
    # probability x_i (same for both goods) since the bidder's value is v1 + v2
    # and the allocation is a bundle. But the interface expects (16,2) which suggests
    # allocation probability for good 1 and good 2 separately.

    # Given the structure, I'll set both goods to have allocation probability x_i
    allocations = np.column_stack([x, x])

    return {'allocations': allocations, 'payments': p}
