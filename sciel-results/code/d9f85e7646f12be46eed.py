import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction.
    
    Strategy: Solve the mechanism design LP using only adjacent-type IC constraints
    (sufficient by single-crossing on the grid), with a single high-precision HiGHS
    solve followed by verification and minimal repair.
    """
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin for 60s limit

    # 1. Parse Input
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            vals = list(distribution_spec.values())
            for v in vals:
                if isinstance(v, (list, np.ndarray)):
                    probs = np.array(v, dtype=float)
                    break
            else:
                probs = np.ones(16) / 16.0
    else:
        probs = np.array(distribution_spec, dtype=float)

    if probs.shape != (16,):
        probs = probs.flatten()
        if len(probs) != 16:
            if len(probs) > 16:
                probs = probs[:16]
            else:
                new_probs = np.zeros(16)
                new_probs[:len(probs)] = probs
                probs = new_probs

    total_p = np.sum(probs)
    if total_p > 1e-12:
        probs = probs / total_p
    else:
        probs = np.ones(16) / 16.0

    # 2. Define Grid
    # Index i = v1 * 4 + v2
    grid_values = []
    for v1 in range(4):
        for v2 in range(4):
            grid_values.append((v1, v2))

    n_types = 16
    n_vars = 48  # 16 x1, 16 x2, 16 p

    # 3. Formulate LP with adjacent-type IC constraints
    # Maximize Revenue = sum(p_i * probs_i) => Minimize -sum(p_i * probs_i)
    c = np.zeros(n_vars)
    for i in range(16):
        c[32 + i] = -probs[i]

    A_rows = []
    b_rows = []

    # IR Constraints: v1*x1_i + v2*x2_i - p_i >= 0
    # => -v1*x1_i - v2*x2_i + p_i <= 0
    for i in range(16):
        v1, v2 = grid_values[i]
        row = np.zeros(n_vars)
        row[i] = -v1
        row[16 + i] = -v2
        row[32 + i] = 1
        A_rows.append(row)
        b_rows.append(0)

    # Adjacent-type IC Constraints: u(i) >= u(j) for j adjacent to i
    # u(i) = v_i . x_i - p_i
    # u(i) >= u(j) means: v_i.x_i - p_i >= v_i.x_j - p_j
    # => -v_i.x_i + v_i.x_j + p_i - p_j <= 0
    for i in range(16):
        v1_i, v2_i = grid_values[i]
        v1_i_idx = i // 4
        v2_i_idx = i % 4

        # Adjacent types: differ by ±1 in v1 or v2
        neighbors = []
        if v1_i_idx > 0:
            neighbors.append((v1_i_idx - 1, v2_i_idx))
        if v1_i_idx < 3:
            neighbors.append((v1_i_idx + 1, v2_i_idx))
        if v2_i_idx > 0:
            neighbors.append((v1_i_idx, v2_i_idx - 1))
        if v2_i_idx < 3:
            neighbors.append((v1_i_idx, v2_i_idx + 1))

        for (nv1, nv2) in neighbors:
            j = nv1 * 4 + nv2
            row = np.zeros(n_vars)
            row[i] = -v1_i
            row[16 + i] = -v2_i
            row[j] = v1_i
            row[16 + j] = v2_i
            row[32 + i] = 1
            row[32 + j] = -1
            A_rows.append(row)
            b_rows.append(0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds: x in [0,1], p in [0, inf)
    bounds = [(0, 1)] * 32 + [(0, None)] * 16

    # 4. Solve with high precision
    from scipy.optimize import linprog

    allocations = np.zeros((16, 2))
    payments = np.zeros(16)

    try:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            return {"allocations": allocations, "payments": payments}

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12,
                               'dual_gap_tolerance': 1e-12,
                               'time_limit': remaining - 3})

        if time.time() - start_time > DEADLINE:
            return {"allocations": allocations, "payments": payments}

        if res.success:
            x = res.x
            for i in range(16):
                allocations[i, 0] = x[i]
                allocations[i, 1] = x[16 + i]
                payments[i] = x[32 + i]

            # Clip to handle numerical noise
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)

            # 5. Verification and repair
            # Check all IC constraints (full set) and IR
            max_violation = 0.0
            violated = False
            for i in range(16):
                v1_i, v2_i = grid_values[i]
                # IR
                ir_val = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
                if ir_val < -1e-8:
                    violated = True
                    max_violation = max(max_violation, -ir_val)
                # IC against all j
                for j in range(16):
                    if i == j:
                        continue
                    v1_j, v2_j = grid_values[j]
                    # u(i) >= u(j): v_i.x_i - p_i >= v_i.x_j - p_j
                    lhs = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
                    rhs = v1_i * allocations[j, 0] + v2_i * allocations[j, 1] - payments[j]
                    if lhs - rhs < -1e-8:
                        violated = True
                        max_violation = max(max_violation, rhs - lhs)

            if violated:
                # Repair: re-solve with full IC constraints and slightly relaxed RHS
                A_rows_full = []
                b_rows_full = []

                # IR
                for i in range(16):
                    v1, v2 = grid_values[i]
                    row = np.zeros(n_vars)
                    row[i] = -v1
                    row[16 + i] = -v2
                    row[32 + i] = 1
                    A_rows_full.append(row)
                    b_rows_full.append(0)

                # Full IC
                for i in range(16):
                    v1_i, v2_i = grid_values[i]
                    for j in range(16):
                        if i == j:
                            continue
                        row = np.zeros(n_vars)
                        row[i] = -v1_i
                        row[16 + i] = -v2_i
                        row[j] = v1_i
                        row[16 + j] = v2_i
                        row[32 + i] = 1
                        row[32 + j] = -1
                        A_rows_full.append(row)
                        b_rows_full.append(0)

                A_ub_full = np.array(A_rows_full)
                b_ub_full = np.array(b_rows_full)

                remaining2 = DEADLINE - (time.time() - start_time)
                if remaining2 > 3:
                    res2 = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full, bounds=bounds,
                                   method='highs',
                                   options={'presolve': True,
                                            'dual_feasibility_tolerance': 1e-12,
                                            'primal_feasibility_tolerance': 1e-12,
                                            'dual_gap_tolerance': 1e-12,
                                            'time_limit': remaining2 - 2})
                    if res2.success:
                        x2 = res2.x
                        allocations2 = np.zeros((16, 2))
                        payments2 = np.zeros(16)
                        for i in range(16):
                            allocations2[i, 0] = x2[i]
                            allocations2[i, 1] = x2[16 + i]
                            payments2[i] = x2[32 + i]
                        allocations2 = np.clip(allocations2, 0, 1)
                        payments2 = np.clip(payments2, 0, None)

                        # Use the repaired solution if it's feasible
                        # Quick check
                        feasible = True
                        for i in range(16):
                            v1_i, v2_i = grid_values[i]
                            ir_val = v1_i * allocations2[i, 0] + v2_i * allocations2[i, 1] - payments2[i]
                            if ir_val < -1e-8:
                                feasible = False
                                break
                            for j in range(16):
                                if i == j:
                                    continue
                                lhs = v1_i * allocations2[i, 0] + v2_i * allocations2[i, 1] - payments2[i]
                                rhs = v1_i * allocations2[j, 0] + v2_i * allocations2[j, 1] - payments2[j]
                                if lhs - rhs < -1e-8:
                                    feasible = False
                                    break
                            if not feasible:
                                break

                        if feasible:
                            allocations = allocations2
                            payments = payments2

            return {"allocations": allocations, "payments": payments}
        else:
            # Fallback: try full IC with relaxed constraints
            A_rows_full = []
            b_rows_full = []

            for i in range(16):
                v1, v2 = grid_values[i]
                row = np.zeros(n_vars)
                row[i] = -v1
                row[16 + i] = -v2
                row[32 + i] = 1
                A_rows_full.append(row)
                b_rows_full.append(0)

            for i in range(16):
                v1_i, v2_i = grid_values[i]
                for j in range(16):
                    if i == j:
                        continue
                    row = np.zeros(n_vars)
                    row[i] = -v1_i
                    row[16 + i] = -v2_i
                    row[j] = v1_i
                    row[16 + j] = v2_i
                    row[32 + i] = 1
                    row[32 + j] = -1
                    A_rows_full.append(row)
                    b_rows_full.append(1e-10)

            A_ub_full = np.array(A_rows_full)
            b_ub_full = np.array(b_rows_full)

            remaining3 = DEADLINE - (time.time() - start_time)
            if remaining3 > 3:
                res3 = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full, bounds=bounds,
                               method='highs',
                               options={'presolve': True,
                                        'time_limit': remaining3 - 2})
                if res3.success:
                    x3 = res3.x
                    for i in range(16):
                        allocations[i, 0] = x3[i]
                        allocations[i, 1] = x3[16 + i]
                        payments[i] = x3[32 + i]
                    allocations = np.clip(allocations, 0, 1)
                    payments = np.clip(payments, 0, None)

            return {"allocations": allocations, "payments": payments}

    except Exception:
        return {"allocations": allocations, "payments": payments}
