"""Incentive-compatible finite auction design via utility-variable LP with local repair."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=np.float64)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=np.float64)
        else:
            probs = None
            for key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 16:
                    probs = np.array(val, dtype=np.float64)
                    break
            if probs is None:
                probs = np.ones(16, dtype=np.float64) / 16.0
    else:
        probs = np.array(distribution_spec, dtype=np.float64)
        if len(probs) != 16:
            probs = np.ones(16, dtype=np.float64) / 16.0
    
    # Normalize probabilities
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(16, dtype=np.float64) / 16.0
    
    # Parse support
    if support is None or len(support) != 16:
        support_list = [(i // 4, i % 4) for i in range(16)]
    else:
        support_list = []
        for s in support:
            if isinstance(s, (list, tuple, np.ndarray)):
                support_list.append((float(s[0]), float(s[1])))
            else:
                support_list.append((float(s), 0.0))
        if len(support_list) != 16:
            support_list = [(i // 4, i % 4) for i in range(16)]
    
    n_types = 16
    n_goods = 2
    
    # Variables: x[i][j] for i in 0..15, j in 0..1 (32 vars)
    #           u[i] for i in 0..15 (utility, 16 vars)
    # Total: 48 vars
    #
    # Key insight: For a single additive bidder, IC + IR imply:
    #   p(v) = v·x(v) - u(v)
    #   IR: u(v) >= 0
    #   IC: u(v) >= v·x(v') - p(v') = v·x(v') - v'·x(v') + u(v')
    #       => u(v) - u(v') >= v·x(v') - v'·x(v') for all v, v'
    #
    # Objective: Maximize E[p(v)] = E[v·x(v) - u(v)]
    #
    # This eliminates explicit payment variables and the nonnegativity constraint on p.
    # Instead we have u >= 0 (IR) and the IC differences.
    
    n_x = 32
    n_u = 16
    n_vars = n_x + n_u
    
    # Objective: maximize sum_i probs[i] * (v_i·x(i) - u(i))
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(16):
        v1, v2 = support_list[i]
        c[i * 2 + 0] = -probs[i] * v1
        c[i * 2 + 1] = -probs[i] * v2
        c[n_x + i] = probs[i]  # because we're negating -u(i)
    
    # Inequality constraints: A_ub @ vars <= b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # Constraint 1: x[i][j] <= 1
    for i in range(16):
        for j in range(2):
            row = np.zeros(n_vars)
            row[i * 2 + j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
    
    # Constraint 2: -u[i] <= 0 (i.e., u[i] >= 0) [IR]
    for i in range(16):
        row = np.zeros(n_vars)
        row[n_x + i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # Constraint 3: IC: u(i) - u(j) >= v_i·x(j) - v_j·x(j) for all i != j
    # i.e., -u(i) + u(j) + v_i·x(j) - v_j·x(j) <= 0
    for i in range(16):
        v1_i, v2_i = support_list[i]
        for j in range(16):
            if i == j:
                continue
            v1_j, v2_j = support_list[j]
            row = np.zeros(n_vars)
            # -u(i)
            row[n_x + i] = -1.0
            # +u(j)
            row[n_x + j] = 1.0
            # +v_i·x(j)
            row[j * 2 + 0] = v1_i
            row[j * 2 + 1] = v2_i
            # -v_j·x(j)
            row[j * 2 + 0] -= v1_j
            row[j * 2 + 1] -= v2_j
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds: x[i][j] in [0,1], u[i] >= 0 (unbounded above)
    bounds = []
    for i in range(16):
        for j in range(2):
            bounds.append((0.0, 1.0))
    for i in range(16):
        bounds.append((0.0, None))
    
    # Solve LP with tight tolerances
    try:
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
            options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                     'primal_feasibility_tolerance': 1e-12,
                     'integrality_tolerance': 1e-12}
        )
    except Exception:
        result = None
    
    # Check time
    if time.time() > deadline:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    if result is None or not result.success:
        # Fallback: deterministic second-price style
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(16):
            v1, v2 = support_list[i]
            allocations[i] = [1.0 if v1 >= 1 else 0.0, 1.0 if v2 >= 1 else 0.0]
            payments[i] = 0.0
        return {'allocations': allocations, 'payments': payments}
    
    # Extract solution
    x_vals = result.x[:n_x].reshape(16, 2)
    u_vals = result.x[n_x:]
    
    # Compute payments: p(i) = v_i·x(i) - u(i)
    p_vals = np.zeros(16)
    for i in range(16):
        v1, v2 = support_list[i]
        p_vals[i] = v1 * x_vals[i, 0] + v2 * x_vals[i, 1] - u_vals[i]
    
    # Clip allocations to [0, 1] with tiny tolerance
    x_vals = np.clip(x_vals, 0.0, 1.0)
    
    # Ensure payments are non-negative
    p_vals = np.clip(p_vals, 0.0, None)
    
    # Local repair pass: check IC and IR, fix tiny violations by adjusting utilities
    if time.time() < deadline - 3.0:
        # Verify and repair
        max_violation = 0.0
        violations = []
        for i in range(16):
            v1_i, v2_i = support_list[i]
            util_i = v1_i * x_vals[i, 0] + v1_i * 0 + v2_i * x_vals[i, 1] - p_vals[i]
            # IR: util_i >= 0
            if util_i < -1e-10:
                max_violation = max(max_violation, -util_i)
                violations.append(('IR', i, -util_i))
            for j in range(16):
                if i == j:
                    continue
                v1_j, v2_j = support_list[j]
                util_i_report_j = v1_i * x_vals[j, 0] + v2_i * x_vals[j, 1] - p_vals[j]
                if util_i < util_i_report_j - 1e-10:
                    viol = util_i_report_j - util_i
                    max_violation = max(max_violation, viol)
                    violations.append(('IC', i, j, viol))
        
        if max_violation > 0 and max_violation < 1e-6:
            # Tiny violations: try to fix by adjusting utilities
            # For each IC violation u(i) - u(j) >= v_i·x(j) - v_j·x(j),
            # we need to increase u(i) or decrease u(j)
            # Simple approach: increase u(i) for violated types
            for _ in range(3):  # A few iterations
                changed = False
                for i in range(16):
                    v1_i, v2_i = support_list[i]
                    util_i = v1_i * x_vals[i, 0] + v2_i * x_vals[i, 1] - p_vals[i]
                    for j in range(16):
                        if i == j:
                            continue
                        v1_j, v2_j = support_list[j]
                        util_i_report_j = v1_i * x_vals[j, 0] + v2_i * x_vals[j, 1] - p_vals[j]
                        if util_i < util_i_report_j - 1e-12:
                            # Need to increase u(i) or decrease u(j)
                            # Increase u(i) by the violation amount
                            needed = util_i_report_j - util_i + 1e-12
                            u_vals[i] += needed
                            p_vals[i] = v1_i * x_vals[i, 0] + v2_i * x_vals[i, 1] - u_vals[i]
                            p_vals[i] = max(0.0, p_vals[i])
                            changed = True
                            break
                    if changed:
                        break
                if not changed:
                    break
            
            # Re-verify after repair
            max_violation2 = 0.0
            for i in range(16):
                v1_i, v2_i = support_list[i]
                util_i = v1_i * x_vals[i, 0] + v2_i * x_vals[i, 1] - p_vals[i]
                if util_i < -1e-9:
                    max_violation2 = max(max_violation2, -util_i)
                for j in range(16):
                    if i == j:
                        continue
                    v1_j, v2_j = support_list[j]
                    util_i_report_j = v1_i * x_vals[j, 0] + v2_i * x_vals[j, 1] - p_vals[j]
                    if util_i < util_i_report_j - 1e-9:
                        max_violation2 = max(max_violation2, util_i_report_j - util_i)
            
            if max_violation2 <= 1e-8:
                # Repair successful
                pass
            else:
                # Repair failed, fall back to original
                u_vals = result.x[n_x:]
                p_vals = np.zeros(16)
                for i in range(16):
                    v1, v2 = support_list[i]
                    p_vals[i] = v1 * x_vals[i, 0] + v2 * x_vals[i, 1] - u_vals[i]
                p_vals = np.clip(p_vals, 0.0, None)
        elif max_violation >= 1e-6:
            # Large violation: conservative fallback
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            return {'allocations': allocations, 'payments': payments}
    
    # Final verification
    max_violation_final = 0.0
    for i in range(16):
        v1_i, v2_i = support_list[i]
        util_i = v1_i * x_vals[i, 0] + v2_i * x_vals[i, 1] - p_vals[i]
        if util_i < -1e-9:
            max_violation_final = max(max_violation_final, -util_i)
        for j in range(16):
            if i == j:
                continue
            v1_j, v2_j = support_list[j]
            util_i_report_j = v1_i * x_vals[j, 0] + v2_i * x_vals[j, 1] - p_vals[j]
            if util_i < util_i_report_j - 1e-9:
                max_violation_final = max(max_violation_final, util_i_report_j - util_i)
    
    if max_violation_final > 1e-8:
        # Significant violation: conservative fallback
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    return {'allocations': x_vals, 'payments': p_vals}
# EVOLVE-BLOCK-END
