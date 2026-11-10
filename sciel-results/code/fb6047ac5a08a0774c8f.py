import time
import numpy as np
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: specification of the type distribution
        support: support of types (values in {0,1,2,3} for each good)
        constraints: additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    WALL_LIMIT = 90.0
    SAFETY_MARGIN = 12.0
    DEADLINE = WALL_LIMIT - SAFETY_MARGIN  # 78s

    def time_exceeded():
        return (time.time() - start_time) >= DEADLINE

    # Values are in {0,1,2,3} for each of 2 goods -> 16 types
    values_per_good = [0, 1, 2, 3]
    n_types = 16

    # Enumerate all type profiles: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in values_per_good for v2 in values_per_good]
    type_index = {t: i for i, t in enumerate(types)}

    # Get distribution from distribution_spec
    probs = None
    if isinstance(distribution_spec, dict):
        for key in ('probs', 'distribution', 'weights', 'prob', 'p'):
            if key in distribution_spec:
                try:
                    probs = np.array(distribution_spec[key], dtype=float).flatten()
                    if len(probs) == n_types:
                        break
                except Exception:
                    probs = None
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        try:
            probs = np.array(distribution_spec, dtype=float).flatten()
        except Exception:
            probs = None

    if probs is None or len(probs) != n_types or not np.all(np.isfinite(probs)):
        probs = np.ones(n_types) / n_types
    else:
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            probs = np.ones(n_types) / n_types
        else:
            probs = probs / total
            probs = np.maximum(probs, 0.0)
            probs = probs / probs.sum()

    # Build fallback mechanism (Myerson-style threshold)
    def build_fallback():
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        
        for i, (v1, v2) in enumerate(types):
            if v1 >= 2:
                allocations[i, 0] = 1.0
            if v2 >= 2:
                allocations[i, 1] = 1.0

        for i, (v1, v2) in enumerate(types):
            x1 = allocations[i, 0]
            x2 = allocations[i, 1]
            sum_x1_lower = 0.0
            for v1_lower in range(v1):
                idx = type_index[(v1_lower, v2)]
                sum_x1_lower += allocations[idx, 0]
            sum_x2_lower = 0.0
            for v2_lower in range(v2):
                idx = type_index[(v1, v2_lower)]
                sum_x2_lower += allocations[idx, 1]
            payments[i] = v1 * x1 + v2 * x2 - sum_x1_lower - sum_x2_lower
        
        payments = np.maximum(payments, 0)
        for i in range(n_types):
            max_pay = types[i][0]*allocations[i,0] + types[i][1]*allocations[i,1]
            if payments[i] > max_pay + 1e-10:
                payments[i] = max_pay
        
        return {"allocations": allocations, "payments": payments}

    # Try LP-based optimal mechanism with sparse matrix for efficiency
    try:
        from scipy.optimize import linprog
        import scipy.sparse as sp

        n_vars = n_types * 3  # x1, x2, p for each type

        # Objective: maximize sum_i probs[i] * p[i]
        c = np.zeros(n_vars)
        for i in range(n_types):
            c[3*i + 2] = -probs[i]
        
        # Build IC constraints using lattice structure with sparse representation
        # For all pairs (i, j): u_i(t_i) >= u_i(t_j) for all types t_i
        # u_i(t) = v_i[0]*x_t[0] + v_i[1]*x_t[1] - p_t
        # Need: v_i[0]*x_i[0] + v_i[1]*x_i[1] - p_i >= v_i[0]*x_j[0] + v_i[1]*x_j[1] - p_j
        # Rearranged: -v_i[0]*x_i[0] - v_i[1]*x_i[1] + v_i[0]*x_j[0] + v_i[1]*x_j[1] + p_i - p_j <= 0
        
        ic_rows = []
        ic_cols = []
        ic_vals = []
        
        for i in range(n_types):
            vi0, vi1 = types[i]
            for j in range(n_types):
                if i == j:
                    continue
                row_idx = len(ic_rows)
                # Coefficient for x_i[0]: -vi0
                ic_rows.append(row_idx); ic_cols.append(3*i + 0); ic_vals.append(-vi0)
                # Coefficient for x_i[1]: -vi1
                ic_rows.append(row_idx); ic_cols.append(3*i + 1); ic_vals.append(-vi1)
                # Coefficient for x_j[0]: vi0
                ic_rows.append(row_idx); ic_cols.append(3*j + 0); ic_vals.append(vi0)
                # Coefficient for x_j[1]: vi1
                ic_rows.append(row_idx); ic_cols.append(3*j + 1); ic_vals.append(vi1)
                # Coefficient for p_i: 1
                ic_rows.append(row_idx); ic_cols.append(3*i + 2); ic_vals.append(1.0)
                # Coefficient for p_j: -1
                ic_rows.append(row_idx); ic_cols.append(3*j + 2); ic_vals.append(-1.0)
        
        n_ic = len(types) * (len(types) - 1)
        
        # IR constraints: p_i - v_i·x_i <= 0
        ir_rows = []
        ir_cols = []
        ir_vals = []
        for i in range(n_types):
            vi0, vi1 = types[i]
            row_idx = len(ir_rows)
            ir_rows.append(row_idx); ir_cols.append(3*i + 0); ir_vals.append(-vi0)
            ir_rows.append(row_idx); ir_cols.append(3*i + 1); ir_vals.append(-vi1)
            ir_rows.append(row_idx); ir_cols.append(3*i + 2); ir_vals.append(1.0)
        
        n_ir = n_types
        
        # Combine into one sparse matrix
        all_rows = np.concatenate([ic_rows, ir_rows])
        all_cols = np.concatenate([ic_cols, ir_cols])
        all_vals = np.concatenate([ic_vals, ir_vals])
        
        n_constraints = n_ic + n_ir
        A_ub = sp.csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_constraints, n_vars))
        b_ub = np.zeros(n_constraints)
        
        bounds = []
        for i in range(n_types):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.0, None))
        
        if time_exceeded():
            return build_fallback()
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success and result.x is not None:
            x = result.x
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            for i in range(n_types):
                allocations[i, 0] = max(0.0, min(1.0, x[3*i + 0]))
                allocations[i, 1] = max(0.0, min(1.0, x[3*i + 1]))
                payments[i] = max(0.0, x[3*i + 2])
            
            # Verify IC and IR
            valid = True
            for i in range(n_types):
                vi0, vi1 = types[i]
                util_i = vi0*allocations[i,0] + vi1*allocations[i,1] - payments[i]
                # IR: util_i >= 0
                if util_i < -1e-8:
                    valid = False
                    break
                for j in range(n_types):
                    if i == j:
                        continue
                    util_i_as_j = vi0*allocations[j,0] + vi1*allocations[j,1] - payments[j]
                    if util_i < util_i_as_j - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return {"allocations": allocations, "payments": payments}
    
    except Exception:
        pass
    
    return build_fallback()
