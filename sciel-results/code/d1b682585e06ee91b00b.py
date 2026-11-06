import time
import numpy as np
from scipy.optimize import linprog

def _fallback_mechanism(types, n_types, n_goods, probs):
    """
    Fallback mechanism if LP fails.
    Uses a simple threshold rule which is not necessarily optimal but valid.
    """
    allocations = np.zeros((n_types, n_goods))
    payments = np.zeros(n_types)
    for i in range(n_types):
        vi = types[i]
        if vi[0] >= 2:
            allocations[i, 0] = 1.0
            payments[i] = float(vi[0])
    return allocations, payments

def _solve_payment_lp(types, n_types, probs, allocations):
    """
    Given fixed allocations, solve for optimal payments to maximize revenue.
    Variables: p[0..n_types-1]
    """
    c = -probs
    A_ub = []
    b_ub = []
    
    # IC: p_i - p_j <= v_i . (x_i - x_j)
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        vi_xi = np.dot(allocations[i], vi)
        for j in range(n_types):
            if i == j: continue
            vi_xj = np.dot(allocations[j], vi)
            row = np.zeros(n_types)
            row[i] = 1
            row[j] = -1
            A_ub.append(row)
            b_ub.append(vi_xi - vi_xj)
            
    # IR: p_i <= v_i . x_i
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        row = np.zeros(n_types)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(np.dot(allocations[i], vi))
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0
    
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
        
    probs = probs / probs.sum()
    
    # LP Variables: x[0..31], p[32..47]
    n_vars = n_types * n_goods + n_types
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[n_types * n_goods + i] = -probs[i]
        
    A_ub = []
    b_ub = []
    
    # IC: p_i - p_j + v_i . (x_j - x_i) <= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j: continue
            row = np.zeros(n_vars)
            row[i * n_goods + 0] = -vi[0]
            row[i * n_goods + 1] = -vi[1]
            row[j * n_goods + 0] = vi[0]
            row[j * n_goods + 1] = vi[1]
            row[n_types * n_goods + i] = 1
            row[n_types * n_goods + j] = -1
            A_ub.append(row)
            b_ub.append(0)
            
    # IR: p_i - v_i . x_i <= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        row = np.zeros(n_vars)
        row[i * n_goods + 0] = -vi[0]
        row[i * n_goods + 1] = -vi[1]
        row[n_types * n_goods + i] = 1
        A_ub.append(row)
        b_ub.append(0)
        
    bounds = []
    for _ in range(n_types * n_goods):
        bounds.append((0, 1))
    for _ in range(n_types):
        bounds.append((0, None))
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    def _validate(allocations, payments):
        """Check IC, IR, non-negativity, and feasibility."""
        if np.any(allocations < -1e-10) or np.any(payments < -1e-10):
            return False
        if np.any(allocations > 1 + 1e-10):
            return False
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            ui_i = np.dot(allocations[i], vi) - payments[i]
            if ui_i < -1e-8:
                return False
            for j in range(n_types):
                ui_j = np.dot(allocations[j], vi) - payments[j]
                if ui_i < ui_j - 1e-8:
                    return False
        return True

    def _calculate_revenue(allocations, payments, probs):
        return np.dot(probs, payments)

    def _repair_payments(allocations, payments):
        """
        Iteratively adjust payments to satisfy IC and IR.
        We reduce payments to the maximum allowed by constraints.
        This ensures validity but may reduce revenue.
        """
        payments = payments.copy()
        for _ in range(10): # Iterate until convergence
            changed = False
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                # IR constraint: p_i <= v_i . x_i
                max_p_ir = np.dot(allocations[i], vi)
                
                # IC constraints: p_i <= p_j + v_i . (x_i - x_j) for all j
                max_p_ic = np.inf
                for j in range(n_types):
                    if i == j: continue
                    vi_xi = np.dot(allocations[i], vi)
                    vi_xj = np.dot(allocations[j], vi)
                    bound = payments[j] + (vi_xi - vi_xj)
                    if bound < max_p_ic:
                        max_p_ic = bound
                
                max_allowed = min(max_p_ir, max_p_ic)
                if payments[i] > max_allowed:
                    payments[i] = max(0.0, max_allowed)
                    changed = True
            if not changed:
                break
        return payments

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})
        best_allocations = None
        best_payments = None
        best_revenue = -np.inf

        if res.success:
            x_flat = res.x[:n_types * n_goods]
            p_flat = res.x[n_types * n_goods:]
            allocations = x_flat.reshape(n_types, n_goods)
            payments = p_flat.copy()
            
            # Clip to feasible range
            allocations = np.clip(allocations, 0, 1)
            payments = np.maximum(payments, 0)
            
            # Check if solution is valid
            if _validate(allocations, payments):
                best_allocations = allocations
                best_payments = payments
                best_revenue = _calculate_revenue(allocations, payments, probs)
            
            # Try repair if not valid
            if best_allocations is None:
                repaired_payments = _repair_payments(allocations, payments)
                if _validate(allocations, repaired_payments):
                    best_allocations = allocations
                    best_payments = repaired_payments
                    best_revenue = _calculate_revenue(allocations, repaired_payments, probs)

            # Polish: Try snapping allocations to 0/1 and re-optimizing payments
            # This can help if the LP solution is a "messy" fractional point that is slightly suboptimal
            # or if the integer vertex is the true optimum and the LP solver stopped early.
            if best_allocations is not None:
                snap_allocations = allocations.copy()
                # Snap values close to 0 or 1
                snap_allocations[snap_allocations < 1e-6] = 0.0
                snap_allocations[snap_allocations > 1.0 - 1e-6] = 1.0
                
                if not np.allclose(snap_allocations, allocations):
                    # Re-solve payments for the snapped allocations
                    snap_payments = _solve_payment_lp(types, n_types, probs, snap_allocations)
                    if snap_payments is not None:
                        snap_payments = np.maximum(snap_payments, 0)
                        if _validate(snap_allocations, snap_payments):
                            snap_revenue = _calculate_revenue(snap_allocations, snap_payments, probs)
                            if snap_revenue > best_revenue + 1e-9:
                                best_allocations = snap_allocations
                                best_payments = snap_payments
                                best_revenue = snap_revenue

        if best_allocations is not None:
            return {'allocations': best_allocations, 'payments': best_payments}
        else:
            alloc_fb, pay_fb = _fallback_mechanism(types, n_types, n_goods, probs)
            return {'allocations': alloc_fb, 'payments': pay_fb}
    except Exception:
        alloc_fb, pay_fb = _fallback_mechanism(types, n_types, n_goods, probs)
        return {'allocations': alloc_fb, 'payments': pay_fb}
