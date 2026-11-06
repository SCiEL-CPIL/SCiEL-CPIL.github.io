<<<<<<< SEARCH
"""Incentive-compatible finite auction design via dual LP formulation."""
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
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.array(distribution_spec.get('values', [1/16]*16), dtype=float)
    else:
        probs = np.array(distribution_spec, dtype=float)
    
    # Normalize probabilities
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0
    
    # Parse support
    if isinstance(support, (list, tuple, np.ndarray)):
        support_list = list(support)
    else:
        support_list = [(i, j) for i in range(4) for j in range(4)]
    
    n_reports = len(support_list)
    if n_reports != 16:
        if n_reports < 16:
            support_list = support_list + [(3, 3)] * (16 - n_reports)
            probs = np.concatenate([probs, np.zeros(16 - len(probs))])
        else:
            support_list = support_list[:16]
            probs = probs[:16]
    
    # Build value matrix V[i] = [v1_i, v2_i]
    V = np.array(support_list, dtype=float)  # shape (16, 2)
    
    # Primal LP formulation:
    # Variables: x[i][j] for i in 0..15, j in 0..1 (32 vars), u[i] for i in 0..15 (16 vars)
    # Maximize: sum_i probs[i] * (V[i] @ x[i] - u[i])
    # Subject to:
    #   0 <= x[i][j] <= 1
    #   u[i] >= 0 (IR)
    #   u[i] - u[j] >= V[i] @ x[j] - V[j] @ x[j] for all i != j (IC)
    #   p[i] = V[i] @ x[i] - u[i] >= 0 (nonnegative payments)
    
    n_x = 32
    n_u = 16
    n_vars = n_x + n_u
    
    # Objective: maximize sum_i probs[i] * (V[i] @ x[i] - u[i])
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(16):
        c[i * 2 + 0] = -probs[i] * V[i, 0]
        c[i * 2 + 1] = -probs[i] * V[i, 1]
        c[n_x + i] = probs[i]  # negating -u[i]
    
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
    
    # Constraint 3: IC: u[i] - u[j] >= V[i] @ x[j] - V[j] @ x[j] for all i != j
    # i.e., -u[i] + u[j] + V[i] @ x[j] - V[j] @ x[j] <= 0
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[n_x + i] = -1.0  # -u[i]
            row[n_x + j] = 1.0   # +u[j]
            # V[i] @ x[j] - V[j] @ x[j] = (V[i] - V[j]) @ x[j]
            row[j * 2 + 0] = V[i, 0] - V[j, 0]
            row[j * 2 + 1] = V[i, 1] - V[j, 1]
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # Constraint 4: Nonnegative payments: p[i] = V[i] @ x[i] - u[i] >= 0
    # i.e., -V[i] @ x[i] + u[i] <= 0
    for i in range(16):
        row = np.zeros(n_vars)
        row[i * 2 + 0] = -V[i, 0]
        row[i * 2 + 1] = -V[i, 1]
        row[n_x + i] = 1.0
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
    
    # Solve LP
    try:
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
            options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                     'primal_feasibility_tolerance': 1e-10}
        )
    except Exception:
        result = None
    
    # Check time
    if time.time() > deadline:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    if result is None or not result.success:
        # Fallback: simple deterministic mechanism
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
    
    # Compute payments: p(i) = V[i] @ x[i] - u[i]
    p_vals = np.zeros(16)
    for i in range(16):
        p_vals[i] = V[i, 0] * x_vals[i, 0] + V[i, 1] * x_vals[i, 1] - u_vals[i]
    
    # Clip allocations to [0, 1]
    x_vals = np.clip(x_vals, 0.0, 1.0)
    
    # Ensure payments are non-negative
    p_vals = np.clip(p_vals, 0.0, None)
    
    # Verification: check IC and IR
    max_violation = 0.0
    for i in range(16):
        util_i = V[i, 0] * x_vals[i, 0] + V[i, 1] * x_vals[i, 1] - p_vals[i]
        if util_i < -1e-9:
            max_violation = max(max_violation, -util_i)
        for j in range(16):
            if i == j:
                continue
            util_i_report_j = V[i, 0] * x_vals[j, 0] + V[i, 1] * x_vals[j, 1] - p_vals[j]
            if util_i < util_i_report_j - 1e-9:
                max_violation = max(max_violation, util_i_report_j - util_i)
    
    if max_violation > 1e-8:
        # Significant violation: use conservative fallback
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    return {'allocations': x_vals, 'payments': p_vals}
=======
"""Incentive-compatible finite auction design via dual LP formulation."""
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
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.array(distribution_spec.get('values', [1/16]*16), dtype=float)
    else:
        probs = np.array(distribution_spec, dtype=float)
    
    # Normalize probabilities
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0
    
    # Parse support
    if isinstance(support, (list, tuple, np.ndarray)):
        support_list = list(support)
    else:
        support_list = [(i, j) for i in range(4) for j in range(4)]
    
    n_reports = len(support_list)
    if n_reports != 16:
        if n_reports < 16:
            support_list = support_list + [(3, 3)] * (16 - n_reports)
            probs = np.concatenate([probs, np.zeros(16 - len(probs))])
        else:
            support_list = support_list[:16]
            probs = probs[:16]
    
    # Build value matrix V[i] = [v1_i, v2_i]
    V = np.array(support_list, dtype=float)  # shape (16, 2)
    
    # Dual LP formulation for mechanism design
    # Primal: max sum_i p_i * (v_i^T x_i - u_i)
    # Subject to: 0 <= x_i <= 1, u_i >= 0, u_i - u_j >= (v_i - v_j)^T x_j, p_i = v_i^T x_i - u_i >= 0
    #
    # Dual variables:
    # alpha_i >= 0: for x_i <= 1 constraints (2 per good, so 32 total)
    # beta_i >= 0: for u_i >= 0 (IR) constraints (16 total)  
    # gamma_ij >= 0: for IC constraints u_i - u_j >= (v_i-v_j)^T x_j (240 total)
    # delta_i >= 0: for p_i >= 0 constraints (16 total)
    #
    # Dual: min sum_i sum_j alpha_ij + sum_i beta_i + sum_i delta_i
    # Subject to: for each i,j: alpha_ij + sum_k gamma_ki * (v_k - v_i)_j + delta_i * v_i_j >= p_i * v_i_j
    #             for each i: beta_i - sum_j gamma_ij + delta_i >= p_i
    #             alpha, beta, gamma, delta >= 0
    
    n_reports = 16
    n_goods = 2
    
    # Dual objective: minimize sum of all dual variables
    # Variables: alpha (32), beta (16), gamma (240), delta (16) = 304 variables
    n_alpha = n_reports * n_goods  # 32
    n_beta = n_reports  # 16
    n_gamma = n_reports * (n_reports - 1)  # 240
    n_delta = n_reports  # 16
    n_dual_vars = n_alpha + n_beta + n_gamma + n_delta
    
    # Objective coefficients (all 1.0 for minimization)
    c_dual = np.ones(n_dual_vars)
    
    # Constraints: A_ub @ dual_vars <= b_ub
    # For each report i and good j: alpha_ij + sum_k gamma_ki*(v_k - v_i)_j + delta_i*v_i_j >= p_i*v_i_j
    # Rearranged: -alpha_ij - sum_k gamma_ki*(v_k - v_i)_j - delta_i*v_i_j <= -p_i*v_i_j
    
    A_ub_rows = []
    b_ub_vals = []
    
    # Build gamma index mapping: gamma[k,i] for k != i, indexed as k*(n_reports-1) + offset
    gamma_idx = {}
    idx = 0
    for k in range(n_reports):
        for i in range(n_reports):
            if k != i:
                gamma_idx[(k, i)] = idx
                idx += 1
    
    # Constraints for allocation bounds: for each i, j
    for i in range(n_reports):
        for j in range(n_goods):
            row = np.zeros(n_dual_vars)
            # alpha_ij coefficient: -1
            row[i * n_goods + j] = -1.0
            # gamma_ki coefficients: -(v_k - v_i)_j for each k != i
            for k in range(n_reports):
                if k != i:
                    row[n_alpha + n_beta + gamma_idx[(k, i)]] = -(V[k, j] - V[i, j])
            # delta_i coefficient: -v_i_j
            row[n_alpha + n_beta + n_gamma + i] = -V[i, j]
            A_ub_rows.append(row)
            b_ub_vals.append(-probs[i] * V[i, j])
    
    # Constraints for utility/IR: for each i: beta_i - sum_j gamma_ij + delta_i >= p_i
    # Rearranged: -beta_i + sum_j gamma_ij - delta_i <= -p_i
    for i in range(n_reports):
        row = np.zeros(n_dual_vars)
        # beta_i coefficient: -1
        row[n_alpha + i] = -1.0
        # gamma_ij coefficients: +1 for each j != i
        for j in range(n_reports):
            if j != i:
                row[n_alpha + n_beta + gamma_idx[(i, j)]] = 1.0
        # delta_i coefficient: -1
        row[n_alpha + n_beta + n_gamma + i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(-probs[i])
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds: all dual variables >= 0
    bounds = [(0.0, None)] * n_dual_vars
    
    # Solve dual LP
    try:
        result = linprog(
            c_dual, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
            options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                     'primal_feasibility_tolerance': 1e-10}
        )
    except Exception:
        result = None
    
    # Check time
    if time.time() > deadline:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    if result is None or not result.success:
        # Fallback: simple deterministic mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(16):
            v1, v2 = support_list[i]
            allocations[i] = [1.0 if v1 >= 1 else 0.0, 1.0 if v2 >= 1 else 0.0]
            payments[i] = 0.0
        return {'allocations': allocations, 'payments': payments}
    
    # Extract dual solution
    dual_sol = result.x
    alpha = dual_sol[:n_alpha].reshape(n_reports, n_goods)
    beta = dual_sol[n_alpha:n_alpha + n_beta]
    gamma_flat = dual_sol[n_alpha + n_beta:n_alpha + n_beta + n_gamma]
    delta = dual_sol[n_alpha + n_beta + n_gamma:]
    
    # Recover primal solution from complementary slackness
    # x_i_j = 1 if alpha_ij > 0, else 0 (for interior)
    # u_i = sum_j gamma_ij - beta_i + delta_i (from IC/IR binding)
    
    # For each report i, compute x_i as the argmax of the dual objective contribution
    # x_i_j = 1 if the dual constraint is tight (alpha_ij > 0 or gamma/delta make it binding)
    
    # Simpler approach: use the dual solution to compute payments and allocations
    # From complementary slackness:
    # If alpha_ij > 0, then x_i_j = 1
    # If gamma_ki > 0, then u_i - u_k = (v_k - v_i)^T x_k
    
    # Compute allocations: x_i_j = 1 if alpha_ij > threshold
    threshold = 1e-6
    x_vals = np.zeros((n_reports, n_goods))
    for i in range(n_reports):
        for j in range(n_goods):
            if alpha[i, j] > threshold:
                x_vals[i, j] = 1.0
    
    # Compute utilities from dual variables
    # u_i = sum_j gamma_ij - beta_i + delta_i (when constraints are binding)
    u_vals = np.zeros(n_reports)
    for i in range(n_reports):
        gamma_sum = 0.0
        for j in range(n_reports):
            if j != i:
                gamma_sum += gamma_flat[gamma_idx[(i, j)]]
        u_vals[i] = gamma_sum - beta[i] + delta[i]
    
    # Compute payments: p_i = v_i^T x_i - u_i
    p_vals = np.zeros(n_reports)
    for i in range(n_reports):
        p_vals[i] = V[i, 0] * x_vals[i, 0] + V[i, 1] * x_vals[i, 1] - u_vals[i]
    
    # Clip allocations to [0, 1]
    x_vals = np.clip(x_vals, 0.0, 1.0)
    
    # Ensure payments are non-negative
    p_vals = np.clip(p_vals, 0.0, None)
    
    # Verification: check IC and IR
    max_violation = 0.0
    for i in range(16):
        util_i = V[i, 0] * x_vals[i, 0] + V[i, 1] * x_vals[i, 1] - p_vals[i]
        if util_i < -1e-9:
            max_violation = max(max_violation, -util_i)
        for j in range(16):
            if i == j:
                continue
            util_i_report_j = V[i, 0] * x_vals[j, 0] + V[i, 1] * x_vals[j, 1] - p_vals[j]
            if util_i < util_i_report_j - 1e-9:
                max_violation = max(max_violation, util_i_report_j - util_i)
    
    if max_violation > 1e-8:
        # Significant violation: use conservative fallback
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    return {'allocations': x_vals, 'payments': p_vals}
>>>>>>> REPLACE
