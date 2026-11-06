import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Parse inputs
    # support: list of 16 types (v1, v2) where vi in {0,1,2,3}
    # distribution_spec: probabilities for each type
    # constraints: additional constraints (may be empty)
    
    # Determine types and probabilities
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n = len(types)  # 16
    
    # Get probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        # Try to get probabilities
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            # Default: uniform
            probs = np.ones(n) / n
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == n:
            probs = probs / probs.sum()
        else:
            probs = np.ones(n) / n
    else:
        probs = np.ones(n) / n
    
    # Ensure probs sum to 1
    probs = probs / probs.sum()
    
    # Variables: q1(v) for v in types (16), q2(v) for v in types (16), p(v) for v in types (16)
    # Total: 48 variables
    # Indexing: q1[0:16], q2[16:32], p[32:48]
    
    n_vars = 3 * n  # 48
    
    # Objective: maximize sum_v probs[v] * p[v]
    # linprog minimizes, so minimize -sum_v probs[v] * p[v]
    c = np.zeros(n_vars)
    c[32:48] = -probs  # negative for maximization
    
    # Constraints:
    # 1. 0 <= q1(v) <= 1 for all v
    # 2. 0 <= q2(v) <= 1 for all v
    # 3. p(v) >= 0 for all v
    # 4. IR: p(v) <= v1*q1(v) + v2*q2(v) for all v
    # 5. IC: for all v, v': p(v) - p(v') <= u(v,v) - u(v',v')
    #    u(v,v) = v1*q1(v) + v2*q2(v)
    #    u(v',v') = v1'*q1(v') + v2'*q2(v')
    #    p(v) - p(v') - (v1*q1(v) + v2*q2(v) - v1'*q1(v') - v2'*q2(v')) <= 0
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # q1
    for i in range(n):
        bounds.append((0, 1))  # q2
    for i in range(n):
        bounds.append((0, None))  # p >= 0
    
    A_ub = []
    b_ub = []
    
    # IR constraints: for each v, -v1*q1(v) - v2*q2(v) + p(v) <= 0
    for i in range(n):
        v1, v2 = types[i]
        row = np.zeros(n_vars)
        row[i] = -v1  # -v1 * q1(v)
        row[n + i] = -v2  # -v2 * q2(v)
        row[2*n + i] = 1  # + p(v)
        A_ub.append(row)
        b_ub.append(0)
    
    # IC constraints: for all v, v'
    # p(v) - p(v') <= u(v,v) - u(v',v')
    # p(v) - p(v') - v1*q1(v) - v2*q2(v) + v1'*q1(v') + v2'*q2(v') <= 0
    for i in range(n):  # true type v
        v1, v2 = types[i]
        for j in range(n):  # reported type v'
            v1p, v2p = types[j]
            row = np.zeros(n_vars)
            # - v1*q1(v) - v2*q2(v)
            row[i] = -v1
            row[n + i] = -v2
            # + v1'*q1(v') + v2'*q2(v')
            row[j] = v1p
            row[n + j] = v2p
            # p(v) - p(v')
            row[2*n + i] = 1
            row[2*n + j] = -1
            A_ub.append(row)
            b_ub.append(0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Solve LP
    # Use tight tolerances to avoid numerical violations
    options = {'presolve': 'highs', 'primal_feasibility_tolerance': 1e-12, 'dual_feasibility_tolerance': 1e-12, 'time_limit': 80.0}
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    
    if not res.success:
        # Fallback: return zeros
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}
    
    x = res.x
    q1 = x[0:n]
    q2 = x[n:2*n]
    
    # Post-processing: Refine payments to ensure strict feasibility
    # We fix q1, q2 and solve for optimal p
    # Maximize sum pi(v) p(v)
    # Subject to:
    # p(v) >= 0
    # p(v) <= u(v)  (IR)
    # p(v) - p(v') <= u(v) - u(v') (IC)
    
    # Calculate utilities u(v) = v1*q1(v) + v2*q2(v)
    u = np.zeros(n)
    for i in range(n):
        v1, v2 = types[i]
        u[i] = v1 * q1[i] + v2 * q2[i]
        
    # Secondary LP for payments
    c_p = -probs  # Maximize revenue => Minimize -revenue
    A_ub_p = []
    b_ub_p = []
    
    # IR: p(v) - u(v) <= 0 => p(v) <= u(v)
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        A_ub_p.append(row)
        b_ub_p.append(u[i])
        
    # IC: p(v) - p(v') - (u(v) - u(v')) <= 0
    for i in range(n):
        for j in range(n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = -1
            A_ub_p.append(row)
            b_ub_p.append(u[i] - u[j])
            
    A_ub_p = np.array(A_ub_p)
    b_ub_p = np.array(b_ub_p)
    
    bounds_p = [(0, None)] * n
    
    res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p, method='highs', options=options)
    
    if res_p.success:
        p = res_p.x
    else:
        # If secondary LP fails, use original p but clamp to IR
        p = x[2*n:3*n]
        for i in range(n):
            p[i] = min(p[i], u[i])
            p[i] = max(p[i], 0)
            
    allocations = np.column_stack([q1, q2])
    payments = p
    
    return {'allocations': allocations, 'payments': payments}

