import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism for a single additive bidder
    with two goods, values in {0,1,2,3}. Returns allocations (16,2) and payments (16,).
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

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

    # Value of each type (additive: v1 + v2)
    values = np.array([v1 + v2 for v1, v2 in types], dtype=float)

    # LP formulation for single additive bidder with two goods
    # Variables: x_0..x_15 (allocation probability of bundle), p_0..p_15 (payments)
    #
    # For a single additive bidder, the optimal mechanism allocates the bundle (both goods)
    # with some probability x_i. The allocation for each good is the same as x_i.
    #
    # Objective: maximize sum_i probs[i] * p_i
    #
    # Constraints:
    # 1. 0 <= x_i <= 1
    # 2. p_i >= 0
    # 3. IR: p_i <= v_i * x_i (utility >= 0)
    # 4. IC: v_i * x_i - p_i >= v_i * x_j - p_j for all i,j
    #    => p_j - p_i >= v_i * (x_j - x_i)
    #    => -v_i * x_i + v_i * x_j + p_i - p_j <= 0

    n_vars = 2 * n  # x_0..x_15, p_0..p_15

    # Objective: minimize -sum(probs[i] * p_i)
    c = np.zeros(n_vars)
    for i in range(n):
        c[n + i] = -probs[i]

    # Inequality constraints: A_ub @ vars <= b_ub
    A_ub_rows = []
    b_ub_vals = []

    # x_i <= 1
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

    # IR: p_i - v_i * x_i <= 0  =>  -v_i * x_i + p_i <= 0
    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = -values[i]
        row[n + i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # IC: For all i != j: v_i * x_i - p_i >= v_i * x_j - p_j
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

    bounds = [(0, None)] * n_vars

    # Solve LP with tighter tolerances for better precision
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12,
                              'time_limit': deadline - time.time() - 2})

    # Check time
    if time.time() > deadline - 3:
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}

    if not result.success:
        # Fallback: allocate nothing, collect nothing
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}

    x = result.x[:n].copy()
    p = result.x[n:].copy()

    # Clean up numerical noise
    x = np.clip(x, 0.0, 1.0)
    p = np.clip(p, 0.0, None)

    # Verify IC constraints with regret check
    max_regret = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            util_i = values[i] * x[i] - p[i]
            util_j = values[i] * x[j] - p[j]
            regret = util_j - util_i
            if regret > max_regret:
                max_regret = regret

    # If regret exceeds tolerance, recompute payments using envelope theorem
    if max_regret > 1e-10:
        # Compute utilities u_i such that:
        # u_i >= u_j + (v_i - v_j) * x_i for all j (IC constraint in utility space)
        # u_i >= 0 (IR)
        # p_i = v_i * x_i - u_i
        #
        # The minimal u that satisfies IC is:
        # u_i = max(0, max_j [u_j + (v_i - v_j) * x_i])
        #
        # This is a system of difference constraints. We solve it by value iteration.
        
        u = np.zeros(n)
        
        # Iterate to convergence
        for _ in range(200):
            changed = False
            for i in range(n):
                if i == 0:
                    continue  # u_0 = 0 always (type with v=0)
                # u_i >= u_j + (v_i - v_j) * x_i for all j
                # = u_j + v_i * x_i - v_j * x_i
                best = 0.0  # IR: u_i >= 0
                for j in range(n):
                    if j == i:
                        continue
                    candidate = u[j] + (values[i] - values[j]) * x[i]
                    if candidate > best:
                        best = candidate
                if best > u[i] + 1e-14:
                    u[i] = best
                    changed = True
            if not changed:
                break
        
        # Recompute payments from utilities
        for i in range(n):
            p[i] = values[i] * x[i] - u[i]
        
        # Enforce IR and nonnegativity
        for i in range(n):
            if values[i] > 0:
                p[i] = min(p[i], values[i] * x[i])
            else:
                p[i] = 0.0
            p[i] = max(0.0, p[i])
    
    # Final verification and cleanup
    x = np.clip(x, 0.0, 1.0)
    p = np.clip(p, 0.0, None)
    
    # Ensure IR one more time
    for i in range(n):
        if values[i] > 0:
            p[i] = min(p[i], values[i] * x[i])
        else:
            p[i] = 0.0
        p[i] = max(0.0, p[i])

    # Construct allocations: both goods allocated with probability x_i
    allocations = np.column_stack([x, x])

    return {'allocations': allocations, 'payments': p}
