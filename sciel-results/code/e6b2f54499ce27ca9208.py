"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

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
    DEADLINE = 80.0  # 10s safety margin before 90s kill
    
    # Values are in {0,1,2,3} for each of 2 goods -> 16 types
    values_per_good = [0, 1, 2, 3]
    n_types = 16
    
    # Enumerate all type profiles: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in values_per_good for v2 in values_per_good]
    type_index = {t: i for i, t in enumerate(types)}
    
    # Helper: check if we're still within deadline
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    # Get distribution from distribution_spec
    # Try to extract probabilities
    probs = None
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'])
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'])
        elif 'weights' in distribution_spec:
            probs = np.array(distribution_spec['weights'])
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec)
    
    if probs is None or len(probs) != n_types:
        # Default: uniform distribution
        probs = np.ones(n_types) / n_types
    else:
        probs = probs / probs.sum()
    
    # Try to solve the LP for the optimal mechanism
    # Variables: x[i,j] for allocation of good j to type i, p[i] for payment to type i
    # Total variables: 16*2 + 16 = 48
    
    # Objective: maximize sum_i probs[i] * p[i]
    # Subject to:
    # 1. IC: for all i, i': v_i * x_i - p_i >= v_i * x_i' - p_i'
    #    i.e., p_i - v_i * x_i <= p_i' - v_i * x_i'
    # 2. IR: p_i <= v_i * x_i (i.e., p_i - v_i * x_i <= 0)
    # 3. Nonnegativity: x_i >= 0, p_i >= 0
    # 4. x_i in [0,1] for each good (fractional allocation allowed)
    
    try:
        from scipy.optimize import linprog
        import scipy.sparse as sp
        
        # Decision variables: [x_0_0, x_0_1, p_0, x_1_0, x_1_1, p_1, ...]
        # Index: for type i, goods at 3*i, 3*i+1, payment at 3*i+2
        n_vars = n_types * 3
        
        # Objective: maximize sum_i probs[i] * p[i] -> minimize -sum_i probs[i] * p[i]
        c = np.zeros(n_vars)
        for i in range(n_types):
            c[3*i + 2] = -probs[i]
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_rows = []
        b_rows = []
        
        # IC constraints: For single additive bidder on a lattice, it suffices to
        # check IC for adjacent types (differing by 1 in one coordinate).
        # This reduces constraints from 240 to ~48, making the LP much faster.
        for i in range(n_types):
            v1, v2 = types[i]
            # Check neighbors: (v1±1, v2) and (v1, v2±1)
            for dv1, dv2 in [(1,0), (-1,0), (0,1), (0,-1)]:
                nv1, nv2 = v1 + dv1, v2 + dv2
                if 0 <= nv1 <= 3 and 0 <= nv2 <= 3:
                    ip = type_index[(nv1, nv2)]
                    # IC: p_i - v_i·x_i <= p_i' - v_i·x_i'
                    # p_i - p_i' - v_i·x_i + v_i·x_i' <= 0
                    row = np.zeros(n_vars)
                    row[3*i + 2] = 1.0    # p_i: +1
                    row[3*ip + 2] = -1.0  # p_i': -1
                    row[3*i + 0] = -v1     # x_i_0: -v1
                    row[3*i + 1] = -v2     # x_i_1: -v2
                    row[3*ip + 0] = v1     # x_i'_0: +v1
                    row[3*ip + 1] = v2     # x_i'_1: +v2
                    A_rows.append(row)
                    b_rows.append(0.0)
        
        # IR constraints: p_i - v_i·x_i <= 0
        for i in range(n_types):
            row = np.zeros(n_vars)
            row[3*i + 2] = 1.0
            row[3*i + 0] = -types[i][0]
            row[3*i + 1] = -types[i][1]
            A_rows.append(row)
            b_rows.append(0.0)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        # Bounds: x in [0,1], p in [0, inf)
        bounds = []
        for i in range(n_types):
            bounds.append((0, 1))  # x_i_0
            bounds.append((0, 1))  # x_i_1
            bounds.append((0, None))  # p_i
        
        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success and time_ok():
            x = result.x
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            for i in range(n_types):
                allocations[i, 0] = x[3*i + 0]
                allocations[i, 1] = x[3*i + 1]
                payments[i] = x[3*i + 2]
            
            # Verify IC for all pairs (full verification since we only used adjacent)
            ic_ok = True
            for i in range(n_types):
                v_i = types[i]
                util_i = v_i[0]*allocations[i,0] + v_i[1]*allocations[i,1] - payments[i]
                for ip in range(n_types):
                    if i == ip:
                        continue
                    util_i_report_ip = v_i[0]*allocations[ip,0] + v_i[1]*allocations[ip,1] - payments[ip]
                    if util_i < util_i_report_ip - 1e-8:
                        ic_ok = False
                        break
                if not ic_ok:
                    break
            
            if ic_ok:
                # Check IR
                ir_ok = True
                for i in range(n_types):
                    max_pay = types[i][0]*allocations[i,0] + types[i][1]*allocations[i,1]
                    if payments[i] > max_pay + 1e-8:
                        ir_ok = False
                        break
                if ir_ok:
                    return {"allocations": allocations, "payments": payments}
        else:
            # LP failed or timeout, use simple mechanism
            pass
    
    except Exception:
        pass
    
    # Fallback: simple dominant strategy mechanism
    # Allocate good j to type i if v_i_j > threshold
    # For single bidder, use Myerson-style: allocate if virtual value positive
    # Simple: allocate good 1 if v1 >= 2, good 2 if v2 >= 2
    # Payment: second-price style
    
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    for i, (v1, v2) in enumerate(types):
        # Allocate good 1 if value >= 2
        if v1 >= 2:
            allocations[i, 0] = 1.0
        # Allocate good 2 if value >= 2
        if v2 >= 2:
            allocations[i, 1] = 1.0
    
    # Compute payments using envelope theorem / IC
    # For discrete types, payment for type i is:
    # p(i) = v_i · x_i - sum over lower types of (v_i - v_i') · (x_i - x_i')
    # Simplified: use the IC binding constraint
    
    # For a single additive bidder, the payment can be computed as:
    # p(v1, v2) = v1*x1(v1,v2) + v2*x2(v1,v2) - integral of x1 dv1 - integral of x2 dv2
    # For discrete: p(v1,v2) = v1*x1 + v2*x2 - sum_{v1'<v1} x1(v1',v2) - sum_{v2'<v2} x2(v1,v2')
    
    for i, (v1, v2) in enumerate(types):
        x1 = allocations[i, 0]
        x2 = allocations[i, 1]
        # Sum of x1 for lower v1 (same v2)
        sum_x1_lower = 0.0
        for v1_lower in range(v1):
            idx = type_index[(v1_lower, v2)]
            sum_x1_lower += allocations[idx, 0]
        # Sum of x2 for lower v2 (same v1)
        sum_x2_lower = 0.0
        for v2_lower in range(v2):
            idx = type_index[(v1, v2_lower)]
            sum_x2_lower += allocations[idx, 1]
        
        payments[i] = v1 * x1 + v2 * x2 - sum_x1_lower - sum_x2_lower
    
    # Ensure nonnegative payments
    payments = np.maximum(payments, 0)
    
    # Check IR: p_i <= v_i · x_i
    for i in range(n_types):
        max_pay = types[i][0]*allocations[i,0] + types[i][1]*allocations[i,1]
        if payments[i] > max_pay + 1e-10:
            payments[i] = max_pay
    
    return {"allocations": allocations, "payments": payments}

# EVOLVE-BLOCK-END
