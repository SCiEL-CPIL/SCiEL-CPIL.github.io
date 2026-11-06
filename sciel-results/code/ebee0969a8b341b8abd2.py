import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction.
    
    Strategy: Direct LP with standard IC constraints (no epsilon), solved via HiGHS,
    followed by a high-precision polish pass with tighter tolerances.
    """
    start_time = time.time()
    DEADLINE = 85.0

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
    grid_values = []
    for v1 in range(4):
        for v2 in range(4):
            grid_values.append((v1, v2))

    n_types = 16
    n_vars = 48  # 16 x1, 16 x2, 16 p

    # 3. Formulate LP
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

    # IC Constraints (standard, no epsilon): u(i) >= u(j)
    # v_i.x_i - p_i >= v_i.x_j - p_j
    # -v_i.x_i + v_i.x_j + p_i - p_j <= 0
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
            A_rows.append(row)
            b_rows.append(0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds: x in [0,1], p in [0, inf)
    bounds = [(0, 1)] * 32 + [(0, None)] * 16

    # 4. Solve with HiGHS
    from scipy.optimize import linprog

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})

        if time.time() - start_time > DEADLINE:
            return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}

        if res.success:
            x = res.x
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            for i in range(16):
                allocations[i, 0] = x[i]
                allocations[i, 1] = x[16 + i]
                payments[i] = x[32 + i]
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)
            
            # Clip tiny negative values that may arise from numerical error
            payments[payments < 0] = 0
            allocations[allocations < 0] = 0

            # 5. Polish pass: re-solve with tighter tolerances and a small perturbation to break degeneracy
            if time.time() - start_time < DEADLINE - 5.0:
                try:
                    # Add tiny perturbation to objective to break degeneracy
                    c_pert = c.copy()
                    np.random.seed(42)
                    for i in range(16):
                        c_pert[32 + i] += np.random.uniform(-1e-10, 1e-10)
                    
                    res2 = linprog(c_pert, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                                   options={'presolve': True,
                                            'dual_feasibility_tolerance': 1e-14,
                                            'primal_feasibility_tolerance': 1e-14,
                                            'time_limit': 20.0})
                    if res2.success:
                        x2 = res2.x
                        alloc2 = np.zeros((16, 2))
                        pay2 = np.zeros(16)
                        for i in range(16):
                            alloc2[i, 0] = x2[i]
                            alloc2[i, 1] = x2[16 + i]
                            pay2[i] = x2[32 + i]
                        alloc2 = np.clip(alloc2, 0, 1)
                        pay2 = np.clip(pay2, 0, None)
                        # Use the better solution
                        rev1 = np.sum(payments * probs)
                        rev2 = np.sum(pay2 * probs)
                        if rev2 > rev1 - 1e-12:
                            allocations = alloc2
                            payments = pay2
                except Exception:
                    pass
                
                # 6. Local refinement: try to increase payments slightly while maintaining feasibility
                if time.time() - start_time < DEADLINE - 2.0:
                    try:
                        # Check current solution feasibility with strict tolerance
                        def check_feasibility(alloc, pay, tol=1e-10):
                            # Check IR: v1*x1 + v2*x2 - p >= 0
                            for i in range(16):
                                v1, v2 = grid_values[i]
                                surplus = v1 * alloc[i, 0] + v2 * alloc[i, 1] - pay[i]
                                if surplus < -tol:
                                    return False
                            # Check IC: u(i) >= u(j) for all i,j
                            for i in range(16):
                                v1_i, v2_i = grid_values[i]
                                u_i = v1_i * alloc[i, 0] + v2_i * alloc[i, 1] - pay[i]
                                for j in range(16):
                                    if i == j:
                                        continue
                                    v1_j, v2_j = grid_values[j]
                                    u_j = v1_i * alloc[j, 0] + v2_i * alloc[j, 1] - pay[j]
                                    if u_i - u_j < -tol:
                                        return False
                            # Check bounds
                            if np.any(alloc < -tol) or np.any(alloc > 1 + tol):
                                return False
                            if np.any(pay < -tol):
                                return False
                            return True
                        
                        # Try small increments to payments
                        best_alloc = allocations.copy()
                        best_pay = payments.copy()
                        best_rev = np.sum(best_pay * probs)
                        
                        for _ in range(3):  # 3 refinement iterations
                            improved = False
                            for i in range(16):
                                # Try increasing payment[i] by small delta
                                for delta in [1e-8, 1e-9, 1e-10]:
                                    test_pay = best_pay.copy()
                                    test_pay[i] += delta
                                    test_alloc = best_alloc.copy()
                                    
                                    if check_feasibility(test_alloc, test_pay, tol=1e-12):
                                        new_rev = np.sum(test_pay * probs)
                                        if new_rev > best_rev + 1e-14:
                                            best_alloc = test_alloc
                                            best_pay = test_pay
                                            best_rev = new_rev
                                            improved = True
                                            break
                            if not improved:
                                break
                        
                        if best_rev > np.sum(payments * probs) - 1e-14:
                            allocations = best_alloc
                            payments = best_pay
                    except Exception:
                        pass

            return {"allocations": allocations, "payments": payments}
        else:
            # Fallback: try without presolve
            try:
                res_fb = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                                 options={'presolve': False})
                if res_fb.success:
                    x = res_fb.x
                    allocations = np.zeros((16, 2))
                    payments = np.zeros(16)
                    for i in range(16):
                        allocations[i, 0] = x[i]
                        allocations[i, 1] = x[16 + i]
                        payments[i] = x[32 + i]
                    allocations = np.clip(allocations, 0, 1)
                    payments = np.clip(payments, 0, None)
                    return {"allocations": allocations, "payments": payments}
            except Exception:
                pass
            return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}

    except Exception:
        # Final fallback: return zero mechanism
        return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
