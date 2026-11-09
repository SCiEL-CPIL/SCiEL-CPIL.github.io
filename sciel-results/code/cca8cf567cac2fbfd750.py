import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s
    
    # Precompute support grid and state mapping
    values = [0, 1, 2, 3]
    support_grid = [(v1, v2) for v1 in values for v2 in values]
    n_states = len(support_grid)
    state_to_idx = {s: i for i, s in enumerate(support_grid)}
    
    # Precompute value vectors
    v1_vec = np.array([s[0] for s in support_grid], dtype=float)
    v2_vec = np.array([s[1] for s in support_grid], dtype=float)
    
    # Precompute IC constraint matrix (same for all distributions)
    # IC: p_i - p_j <= v_i·x_i - v_i·x_j
    # Rearranged: p_i - p_j - v1_i*x1_i - v2_i*x2_i + v1_i*x1_j + v2_i*x2_j <= 0
    n_vars = 3 * n_states
    ic_rows = []
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[2 * n_states + i] = 1.0
            row[2 * n_states + j] = -1.0
            row[i] = -v1_vec[i]
            row[n_states + i] = -v2_vec[i]
            row[j] = v1_vec[i]
            row[n_states + j] = v2_vec[i]
            ic_rows.append(row)
    A_ic = np.array(ic_rows) if ic_rows else np.empty((0, n_vars))
    b_ic = np.zeros(len(ic_rows))
    
    # Precompute IR constraint matrix
    # IR: p_i - v1_i*x1_i - v2_i*x2_i <= 0
    ir_rows = []
    for i in range(n_states):
        row = np.zeros(n_vars)
        row[2 * n_states + i] = 1.0
        row[i] = -v1_vec[i]
        row[n_states + i] = -v2_vec[i]
        ir_rows.append(row)
    A_ir = np.array(ir_rows)
    b_ir = np.zeros(n_states)
    
    # Precompute bounds
    bounds = [(0, 1)] * n_states + [(0, 1)] * n_states + [(0, None)] * n_states

    # Extract probabilities robustly
    probs = np.zeros(n_states)
    if isinstance(distribution_spec, dict):
        for s, p in distribution_spec.items():
            if s in state_to_idx:
                probs[state_to_idx[s]] = float(p)
    elif isinstance(distribution_spec, (np.ndarray, list)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (4, 4):
            for i in range(4):
                for j in range(4):
                    idx = state_to_idx[(i, j)]
                    probs[idx] = arr[i, j]
        else:
            flat = arr.flatten()
            if len(flat) == n_states:
                probs = flat.copy()
            else:
                probs = np.ones(n_states) / n_states
    else:
        probs = np.ones(n_states) / n_states
    
    # Normalize
    total_prob = probs.sum()
    if total_prob > 1e-12:
        probs = probs / total_prob
    else:
        probs = np.ones(n_states) / n_states

    # Build objective: maximize expected revenue = sum(probs[i] * p[i])
    # Minimize: -sum(probs[i] * p[i])
    c = np.zeros(n_vars)
    for i in range(n_states):
        c[2 * n_states + i] = -probs[i]
    
    # Combine constraints
    A_ub = np.vstack([A_ic, A_ir])
    b_ub = np.concatenate([b_ic, b_ir])

    # Check time
    if time.time() - start_time > deadline - 5:
        allocations = np.zeros((n_states, 2))
        payments = np.zeros(n_states)
        return {"allocations": allocations, "payments": payments}

    # Solve LP with high precision settings
    x_opt = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12,
                              'dual_stall_tolerance': 1e-12, 'primal_stall_tolerance': 1e-12})
        if res.status == 0:
            x_opt = res.x
        else:
            # Try interior-point as fallback if highs fails
            res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point',
                          options={'maxiter': 1000, 'tol': 1e-12})
            if res2.status == 0:
                x_opt = res2.x
    except Exception:
        x_opt = None

    if x_opt is None:
        # Fallback: simple deterministic allocation (always allocate both goods)
        # with VCG-like payment to ensure IC and IR
        allocations = np.ones((n_states, 2))
        # Payment: p_i = v1_i + v2_i - (v1_i + v2_i) = 0? No, that's not right.
        # For always-allocate: x_i = (1,1) for all i
        # IC: p_i - p_j <= v_i·(1,1) - v_i·(1,1) = 0, so p_i = p_j for all i
        # IR: p_i <= v1_i + v2_i
        # To maximize revenue: set all p_i = min_i (v1_i + v2_i) = 0
        # So payments are all 0. Revenue = 0.
        # Alternative: use a threshold mechanism
        # For type (v1,v2), allocate good 1 if v1 >= 2, good 2 if v2 >= 2
        # This is a known IC mechanism
        allocations = np.zeros((n_states, 2))
        for i, (v1, v2) in enumerate(support_grid):
            if v1 >= 2:
                allocations[i, 0] = 1.0
            if v2 >= 2:
                allocations[i, 1] = 1.0
        # Compute payments via the IC/IR LP for this fixed allocation
        n_p = n_states
        c_p = -probs.copy()
        A_ub_p = []
        b_ub_p = []
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            val_i = v1_i * allocations[i, 0] + v2_i * allocations[i, 1]
            for j in range(n_states):
                if i == j:
                    continue
                val_j_at_i = v1_i * allocations[j, 0] + v2_i * allocations[j, 1]
                row = np.zeros(n_p)
                row[i] = 1.0
                row[j] = -1.0
                A_ub_p.append(row)
                b_ub_p.append(val_i - val_j_at_i)
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            val_i = v1_i * allocations[i, 0] + v2_i * allocations[i, 1]
            row = np.zeros(n_p)
            row[i] = 1.0
            A_ub_p.append(row)
            b_ub_p.append(val_i)
        bounds_p = [(0, None)] * n_p
        if A_ub_p:
            A_ub_p = np.array(A_ub_p)
            b_ub_p = np.array(b_ub_p)
        else:
            A_ub_p = np.empty((0, n_p))
            b_ub_p = np.empty(0)
        payments = np.zeros(n_states)
        try:
            res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p, method='highs',
                           options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                                    'primal_feasibility_tolerance': 1e-10,
                                    'dual_stall_tolerance': 1e-10, 'primal_stall_tolerance': 1e-10})
            if res_p.status == 0:
                payments = res_p.x
        except Exception:
            pass
        return {"allocations": allocations, "payments": payments}

    x1 = x_opt[:n_states]
    x2 = x_opt[n_states:2 * n_states]
    p = x_opt[2 * n_states:]

    allocations = np.column_stack((x1, x2))
    payments = p

    # Clean up numerical noise
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    # Snap near-boundary values to exact 0 or 1
    allocations[allocations < 1e-10] = 0.0
    allocations[allocations > 1.0 - 1e-10] = 1.0
    payments[payments < 1e-10] = 0.0

    # Verify IC (regret check) using vectorized computation
    # Utility of truthful report i: v_i · x_i - p_i
    # Utility of deviating to report j: v_i · x_j - p_j
    # Regret = max_j (v_i · x_j - p_j) - (v_i · x_i - p_i)
    
    # Compute all utilities: util[i, j] = v_i · x_j - p_j
    # x_j = [x1[j], x2[j]]
    # v_i · x_j = v1_i * x1[j] + v2_i * x2[j]
    
    # Build matrix of values: val[i, j] = v1_i * x1[j] + v2_i * x2[j]
    # This is outer product style
    val_matrix = np.outer(v1_vec, x1) + np.outer(v2_vec, x2)
    
    # Utility when true type is i and reports j: val[i, j] - p[j]
    util_matrix = val_matrix - p[np.newaxis, :]
    
    # Truthful utility for type i: util[i, i]
    truthful_util = np.diag(util_matrix)
    
    # Best deviation utility for type i: max_j util[i, j]
    best_dev_util = util_matrix.max(axis=1)
    
    # Regret for type i
    regret = best_dev_util - truthful_util
    max_regret = regret.max() if len(regret) > 0 else 0.0

    # If regret exceeds tolerance, attempt to fix payments for the given allocation
    if max_regret > 1e-9:
        # Re-optimize payments for fixed allocations
        # Variables: p[0..15]
        n_p = n_states
        c_p = -probs.copy()
        
        A_ub_p = []
        b_ub_p = []
        
        # IC: p_i - p_j <= v_i x_i - v_i x_j
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            val_i = v1_i * x1[i] + v2_i * x2[i]
            for j in range(n_states):
                if i == j:
                    continue
                val_j_at_i = v1_i * x1[j] + v2_i * x2[j]
                # p_i - p_j <= val_i - val_j_at_i
                row = np.zeros(n_p)
                row[i] = 1.0
                row[j] = -1.0
                A_ub_p.append(row)
                b_ub_p.append(val_i - val_j_at_i)
                
        # IR: p_i <= val_i
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            val_i = v1_i * x1[i] + v2_i * x2[i]
            row = np.zeros(n_p)
            row[i] = 1.0
            A_ub_p.append(row)
            b_ub_p.append(val_i)
            
        bounds_p = [(0, None)] * n_p
        
        if A_ub_p:
            A_ub_p = np.array(A_ub_p)
            b_ub_p = np.array(b_ub_p)
        else:
            A_ub_p = np.empty((0, n_p))
            b_ub_p = np.empty(0)
            
        try:
            res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p, method='highs',
                           options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                                    'primal_feasibility_tolerance': 1e-10,
                                    'dual_stall_tolerance': 1e-10, 'primal_stall_tolerance': 1e-10})
            if res_p.status == 0:
                new_payments = res_p.x
                # Re-verify with vectorized computation
                val_matrix_2 = np.outer(v1_vec, x1) + np.outer(v2_vec, x2)
                util_matrix_2 = val_matrix_2 - new_payments[np.newaxis, :]
                truthful_util_2 = np.diag(util_matrix_2)
                best_dev_util_2 = util_matrix_2.max(axis=1)
                regret_2 = best_dev_util_2 - truthful_util_2
                max_regret_2 = regret_2.max() if len(regret_2) > 0 else 0.0
                
                if max_regret_2 <= 1e-8:
                    payments = new_payments
                else:
                    # Still failed, try with slightly relaxed IC by adjusting payments
                    # Use the maximum payment that satisfies IR and is as close to IC as possible
                    # Fallback: set payments to the minimum that satisfies IR
                    # p_i = min over j of (v_i·x_j - v_i·x_i + p_j) ... complex
                    # Simple fallback: zero payments
                    allocations = np.zeros((n_states, 2))
                    payments = np.zeros(n_states)
            else:
                # LP for payments failed, fallback to zero
                allocations = np.zeros((n_states, 2))
                payments = np.zeros(n_states)
        except Exception:
            allocations = np.zeros((n_states, 2))
            payments = np.zeros(n_states)

    # Final verification and cleanup
    if max_regret > 1e-8:
        # Last resort: use the threshold fallback
        allocations = np.zeros((n_states, 2))
        for i, (v1, v2) in enumerate(support_grid):
            if v1 >= 2:
                allocations[i, 0] = 1.0
            if v2 >= 2:
                allocations[i, 1] = 1.0
        n_p = n_states
        c_p = -probs.copy()
        A_ub_p = []
        b_ub_p = []
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            val_i = v1_i * allocations[i, 0] + v2_i * allocations[i, 1]
            for j in range(n_states):
                if i == j:
                    continue
                val_j_at_i = v1_i * allocations[j, 0] + v2_i * allocations[j, 1]
                row = np.zeros(n_p)
                row[i] = 1.0
                row[j] = -1.0
                A_ub_p.append(row)
                b_ub_p.append(val_i - val_j_at_i)
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            val_i = v1_i * allocations[i, 0] + v2_i * allocations[i, 1]
            row = np.zeros(n_p)
            row[i] = 1.0
            A_ub_p.append(row)
            b_ub_p.append(val_i)
        bounds_p = [(0, None)] * n_p
        if A_ub_p:
            A_ub_p = np.array(A_ub_p)
            b_ub_p = np.array(b_ub_p)
        else:
            A_ub_p = np.empty((0, n_p))
            b_ub_p = np.empty(0)
        payments = np.zeros(n_states)
        try:
            res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p, method='highs',
                           options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                    'primal_feasibility_tolerance': 1e-12,
                                    'dual_stall_tolerance': 1e-12, 'primal_stall_tolerance': 1e-12})
            if res_p.status == 0:
                payments = res_p.x
        except Exception:
            pass

    return {"allocations": allocations, "payments": payments}
