"""Incentive-compatible finite auction design solver using LP with efficient formulation."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit

    # Values are in {0, 1, 2, 3}
    values = [0, 1, 2, 3]

    # Parse distribution_spec
    if isinstance(distribution_spec, dict):
        if 'goods' in distribution_spec:
            p1 = np.array(distribution_spec['goods'][0], dtype=float)
            p2 = np.array(distribution_spec['goods'][1], dtype=float)
        elif 'p1' in distribution_spec and 'p2' in distribution_spec:
            p1 = np.array(distribution_spec['p1'], dtype=float)
            p2 = np.array(distribution_spec['p2'], dtype=float)
        elif 'probs' in distribution_spec:
            probs = distribution_spec['probs']
            if isinstance(probs, (list, tuple)) and len(probs) == 2:
                p1 = np.array(probs[0], dtype=float)
                p2 = np.array(probs[1], dtype=float)
            else:
                p1 = np.array(probs, dtype=float)
                p2 = np.array(probs, dtype=float)
        else:
            p1 = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
            p2 = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 2:
            p1 = np.array(distribution_spec[0], dtype=float)
            p2 = np.array(distribution_spec[1], dtype=float)
        else:
            p1 = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
            p2 = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    else:
        p1 = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        p2 = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

    # Ensure probabilities are valid
    p1 = np.maximum(p1, 0)
    p2 = np.maximum(p2, 0)
    if p1.sum() > 0:
        p1 = p1 / p1.sum()
    else:
        p1 = np.array([0.25, 0.25, 0.25, 0.25])
    if p2.sum() > 0:
        p2 = p2 / p2.sum()
    else:
        p2 = np.array([0.25, 0.25, 0.25, 0.25])

    # Parse support
    if support is None:
        support_list = [(v1, v2) for v1 in values for v2 in values]
    elif isinstance(support, np.ndarray):
        if support.ndim == 1:
            if len(support) % 2 == 0:
                support_list = [(int(support[i]), int(support[i+1])) for i in range(0, len(support), 2)]
            else:
                support_list = [(int(v), 0) for v in support]
        elif support.ndim == 2:
            support_list = [(int(row[0]), int(row[1])) for row in support]
        else:
            support_list = [(v1, v2) for v1 in values for v2 in values]
    elif isinstance(support, (list, tuple)):
        if len(support) > 0:
            if isinstance(support[0], (list, tuple)):
                support_list = [(int(s[0]), int(s[1])) for s in support]
            else:
                if len(support) % 2 == 0:
                    support_list = [(int(support[i]), int(support[i+1])) for i in range(0, len(support), 2)]
                else:
                    support_list = [(int(v), 0) for v in support]
        else:
            support_list = [(v1, v2) for v1 in values for v2 in values]
    else:
        support_list = [(v1, v2) for v1 in values for v2 in values]

    # Remove duplicates and sort
    support_set = set(support_list)
    support_list = sorted(support_set)
    n_reports = len(support_list)

    # Build probability for each report
    probs = np.zeros(n_reports)
    for i, (v1, v2) in enumerate(support_list):
        probs[i] = p1[v1] * p2[v2]

    # LP formulation:
    # Variables: x1_0..x1_{n-1}, x2_0..x2_{n-1}, t_0..t_{n-1}
    # n_vars = 3 * n_reports
    #
    # Objective: maximize sum_i probs[i] * t_i
    #
    # Constraints:
    # 1. 0 <= x1_i <= 1, 0 <= x2_i <= 1
    # 2. t_i >= 0
    # 3. IR: v1_i*x1_i + v2_i*x2_i - t_i >= 0  =>  -v1_i*x1_i - v2_i*x2_i + t_i <= 0
    # 4. IC: For all i, j: v1_i*x1_i + v2_i*x2_i - t_i >= v1_i*x1_j + v2_i*x2_j - t_j
    #    => -v1_i*x1_i - v2_i*x2_i + t_i + v1_i*x1_j + v2_i*x2_j - t_j <= 0

    n_vars = 3 * n_reports

    # Objective: maximize sum probs[i] * t_i => minimize -sum probs[i] * t_i
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[3*i + 2] = -probs[i]

    # Build constraint matrix
    A_ub_rows = []
    b_ub_rows = []

    # x1_i <= 1
    for i in range(n_reports):
        row = np.zeros(n_vars)
        row[3*i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(1.0)

    # x2_i <= 1
    for i in range(n_reports):
        row = np.zeros(n_vars)
        row[3*i + 1] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(1.0)

    # -t_i <= 0  (t_i >= 0)
    for i in range(n_reports):
        row = np.zeros(n_vars)
        row[3*i + 2] = -1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)

    # -x1_i <= 0, -x2_i <= 0  (x >= 0)
    for i in range(n_reports):
        row = np.zeros(n_vars)
        row[3*i] = -1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
        row = np.zeros(n_vars)
        row[3*i + 1] = -1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)

    # IR: -v1_i*x1_i - v2_i*x2_i + t_i <= 0
    for i, (v1, v2) in enumerate(support_list):
        row = np.zeros(n_vars)
        row[3*i] = -float(v1)
        row[3*i + 1] = -float(v2)
        row[3*i + 2] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)

    # IC: for all i != j:
    # -v1_i*x1_i - v2_i*x2_i + t_i + v1_i*x1_j + v2_i*x2_j - t_j <= 0
    for i, (v1, v2) in enumerate(support_list):
        for j in range(n_reports):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[3*i] = -float(v1)
            row[3*i + 1] = -float(v2)
            row[3*i + 2] = 1.0
            row[3*j] = float(v1)
            row[3*j + 1] = float(v2)
            row[3*j + 2] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)

    # Bounds
    bounds = []
    for i in range(n_reports):
        bounds.append((0, 1))   # x1
        bounds.append((0, 1))   # x2
        bounds.append((0, None)) # t

    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if result.status == 0 and time.time() < deadline:
            x = result.x
            allocations = np.zeros((n_reports, 2))
            payments = np.zeros(n_reports)
            for i in range(n_reports):
                allocations[i, 0] = x[3*i]
                allocations[i, 1] = x[3*i + 1]
                payments[i] = x[3*i + 2]

            # Clamp to valid range
            allocations = np.clip(allocations, 0, 1)
            payments = np.maximum(payments, 0)

            # Verify constraints and fix small violations
            for i, (v1, v2) in enumerate(support_list):
                # Check IR
                util = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
                if util < -1e-9:
                    payments[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1]
                    payments[i] = max(payments[i], 0)

                # Check IC
                util_ii = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
                for j, (v1j, v2j) in enumerate(support_list):
                    if i == j:
                        continue
                    util_ij = v1 * allocations[j, 0] + v2 * allocations[j, 1] - payments[j]
                    if util_ii < util_ij - 1e-9:
                        # Adjust payment to fix IC
                        payments[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1] - util_ij
                        payments[i] = max(payments[i], 0)

            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            # LP failed or timed out - use fallback
            pass
    except Exception:
        pass

    # Fallback: threshold mechanism (IC by construction)
    allocations = np.zeros((n_reports, 2))
    payments = np.zeros(n_reports)

    for i, (v1, v2) in enumerate(support_list):
        if v1 >= 2:
            allocations[i, 0] = 1.0
            payments[i] += 2.0
        if v2 >= 2:
            allocations[i, 1] = 1.0
            payments[i] += 2.0

    payments = np.maximum(payments, 0)

    return {
        'allocations': allocations,
        'payments': payments
    }
