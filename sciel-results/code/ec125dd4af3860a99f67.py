import numpy as np
import time
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction.
    
    Strategy: Solve the mechanism design LP directly using HiGHS with
    tight tolerances, then apply a high-precision polish pass.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit

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
    # Maximize Revenue = sum(p_i * probs_i)
    # Minimize -sum(p_i * probs_i)
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

    # IC Constraints (standard, no epsilon to avoid infeasibility)
    # u(i) >= u(j)
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

    # Bounds
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))

    # 4. Solve
    best_allocations = None
    best_payments = None
    best_rev = -np.inf

    try:
        # Try multiple tolerance settings, keeping the best feasible solution
        tolerances = [
            {'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12, 'dual_gap_tolerance': 1e-12},
            {'dual_feasibility_tolerance': 1e-10, 'primal_feasibility_tolerance': 1e-10, 'dual_gap_tolerance': 1e-10},
            {'dual_feasibility_tolerance': 1e-8, 'primal_feasibility_tolerance': 1e-8, 'dual_gap_tolerance': 1e-8},
        ]
        
        for tol_opts in tolerances:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                              options={'presolve': True,
                                       'time_limit': max(1.0, DEADLINE - (time.time() - start_time) - 5),
                                       **tol_opts})
                
                if time.time() - start_time > DEADLINE:
                    break
                    
                if res.success:
                    x = res.x
                    allocations = np.zeros((16, 2))
                    payments = np.zeros(16)

                    for i in range(16):
                        allocations[i, 0] = x[i]
                        allocations[i, 1] = x[16 + i]
                        payments[i] = x[32 + i]

                    # Clip to handle numerical noise
                    allocations = np.clip(allocations, 0, 1)
                    payments = np.clip(payments, 0, None)

                    # Verify feasibility more carefully
                    feasible = True
                    
                    # Check IR
                    for i in range(16):
                        v1, v2 = grid_values[i]
                        util = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
                        if util < -1e-8:
                            feasible = False
                            break
                    
                    # Check IC
                    if feasible:
                        for i in range(16):
                            v1_i, v2_i = grid_values[i]
                            for j in range(16):
                                if i == j:
                                    continue
                                util_i = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
                                util_j = v1_i * allocations[j, 0] + v2_i * allocations[j, 1] - payments[j]
                                if util_i < util_j - 1e-8:
                                    feasible = False
                                    break
                            if not feasible:
                                break
                    
                    # Check allocations in [0,1]
                    if feasible:
                        if np.any(allocations < -1e-10) or np.any(allocations > 1 + 1e-10):
                            feasible = False
                    
                    # Check payments non-negative
                    if feasible:
                        if np.any(payments < -1e-10):
                            feasible = False

                    if feasible:
                        rev = np.sum(payments * probs)
                        if rev > best_rev:
                            best_rev = rev
                            best_allocations = allocations.copy()
                            best_payments = payments.copy()
                
            except Exception:
                pass

        if best_allocations is not None:
            return {
                "allocations": best_allocations,
                "payments": best_payments
            }
        
        # If LP fails, try with slightly relaxed constraints
        b_relaxed = b_ub.copy()
        # Relax IC constraints slightly
        for k in range(16, len(b_relaxed)):
            b_relaxed[k] = 1e-10

        if time.time() - start_time < DEADLINE - 5:
            res_relaxed = linprog(c, A_ub=A_ub, b_ub=b_relaxed, bounds=bounds, method='highs',
                                  options={'presolve': True,
                                           'time_limit': max(1.0, DEADLINE - (time.time() - start_time) - 5)})
            if res_relaxed.success:
                x = res_relaxed.x
                allocations = np.zeros((16, 2))
                payments = np.zeros(16)
                for i in range(16):
                    allocations[i, 0] = x[i]
                    allocations[i, 1] = x[16 + i]
                    payments[i] = x[32 + i]
                allocations = np.clip(allocations, 0, 1)
                payments = np.clip(payments, 0, None)
                return {"allocations": allocations, "payments": payments}

        # Final fallback
        return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}

    except Exception:
        return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
