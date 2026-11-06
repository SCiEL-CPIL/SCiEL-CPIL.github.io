<<<<<<< SEARCH
def _solve_reduced_lp(values, probs, n_types, deadline):
    """
    Solves the optimal mechanism problem using the reduced LP formulation.
    Variables: x[v, g] for v in 0..15, g in 0..1 (32 variables)
    Objective: Maximize Expected Revenue (substituting payment identity)
    Constraints:
      1. 0 <= x <= 1
      2. Monotonicity: x[v, g] <= x[v+e_g, g]
      3. Non-negative Payments: p_v(x) >= 0
    """
    start_time = time.time()
    if start_time > deadline - 5.0:
        return None

    n_vars = n_types * 2  # 32 variables
    
    # 1. Objective Coefficients
    # E[R] = sum_v probs[v] * p_v
    # p_v = v1*x[v,0] + v2*x[v,1] - sum_{u1<v1} x[u1,v2,0] - sum_{u2<v2} x[v1,u2,1]
    
    obj_coeffs = np.zeros(n_vars)
    
    for v in range(n_types):
        v1, v2 = int(values[v][0]), int(values[v][1])
        p_v = probs[v]
        
        # Direct terms: +p_v * v1 * x[v,0] and +p_v * v2 * x[v,1]
        obj_coeffs[v*2 + 0] += p_v * v1
        obj_coeffs[v*2 + 1] += p_v * v2
        
        # Subsidy terms for Good 0: -p_v * x[u1, v2, 0] for u1 < v1
        for u1 in range(v1):
            idx_u = u1 * 4 + v2
            obj_coeffs[idx_u * 2 + 0] -= p_v
            
        # Subsidy terms for Good 1: -p_v * x[v1, u2, 1] for u2 < v2
        for u2 in range(v2):
            idx_u = v1 * 4 + u2
            obj_coeffs[idx_u * 2 + 1] -= p_v

    # linprog minimizes, so we negate the objective
    c = -obj_coeffs

    # 2. Bounds
    bounds = [(0.0, 1.0)] * n_vars

    # 3. Constraints (A_ub @ x <= b_ub)
    A_ub_rows = []
    b_ub_rows = []
    
    # 3a. Monotonicity Constraints
    # Good 0: x[(v1,v2),0] <= x[(v1+1,v2),0]
    for v2 in range(4):
        for v1 in range(3):
            i_curr = v1 * 4 + v2
            i_next = (v1 + 1) * 4 + v2
            row = np.zeros(n_vars)
            row[i_curr * 2 + 0] = 1.0
            row[i_next * 2 + 0] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
    # Good 1: x[(v1,v2),1] <= x[(v1,v2+1),1]
    for v1 in range(4):
        for v2 in range(3):
            i_curr = v1 * 4 + v2
            i_next = v1 * 4 + (v2 + 1)
            row = np.zeros(n_vars)
            row[i_curr * 2 + 1] = 1.0
            row[i_next * 2 + 1] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
    # 3b. Non-Negative Payment Constraints
    # p_v >= 0  =>  -p_v <= 0
    # p_v = v1*x[v,0] + v2*x[v,1] - sum_{u1<v1} x[u1,v2,0] - sum_{u2<v2} x[v1,u2,1]
    # -p_v = -v1*x[v,0] - v2*x[v,1] + sum_{u1<v1} x[u1,v2,0] + sum_{u2<v2} x[v1,u2,1] <= 0
    
    for v in range(n_types):
        v1, v2 = int(values[v][0]), int(values[v][1])
        
        # If v1=0 and v2=0, p_v is always 0. -0 <= 0 is 0<=0, which is redundant but harmless.
        # We can skip (0,0) or include it. Including it is fine.
        
        row = np.zeros(n_vars)
        
        # Coefficients for x[v, g]
        if v1 > 0:
            row[v*2 + 0] = -v1
        if v2 > 0:
            row[v*2 + 1] = -v2
            
        # Coefficients for subsidy terms (positive in -p_v)
        for u1 in range(v1):
            idx_u = u1 * 4 + v2
            row[idx_u * 2 + 0] += 1.0
            
        for u2 in range(v2):
            idx_u = v1 * 4 + u2
            row[idx_u * 2 + 1] += 1.0
            
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
        
    if A_ub_rows:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)
    else:
        A_ub = None
        b_ub = None
        
    # 4. Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})
        
        if not res.success:
            return None
            
        x_flat = res.x
        x_alloc = x_flat.reshape(n_types, 2)
        x_alloc = np.clip(x_alloc, 0.0, 1.0)
        
        # 5. Calculate Payments using IC Identity
        payments = np.zeros(n_types)
        for v in range(n_types):
            v1, v2 = int(values[v][0]), int(values[v][1])
            p_v = v1 * x_alloc[v, 0] + v2 * x_alloc[v, 1]
            
            for u1 in range(v1):
                idx_u = u1 * 4 + v2
                p_v -= x_alloc[idx_u, 0]
                
            for u2 in range(v2):
                idx_u = v1 * 4 + u2
                p_v -= x_alloc[idx_u, 1]
                
            payments[v] = p_v
            
        # Ensure non-negative payments (clip tiny negatives from numerical error)
        payments = np.clip(payments, 0.0, None)
        
        return (x_alloc, payments)
        
    except Exception:
        return None
=======
def _solve_reduced_lp(values, probs, n_types, deadline):
    """
    Solves the optimal mechanism LP with explicit utility variables.
    Variables: x[v,g] (32) + u[v] (16) = 48 total.
    Uses envelope-theorem IC constraints directly.
    """
    if time.time() > deadline - 5.0:
        return None

    n_x = n_types * 2
    n_u = n_types
    n_vars = n_x + n_u

    # Objective: Maximize sum_v probs[v] * (v1*x[v,0] + v2*x[v,1] - u[v])
    c = np.zeros(n_vars)
    for v in range(n_types):
        v1, v2 = int(values[v][0]), int(values[v][1])
        p_v = probs[v]
        c[v*2 + 0] -= p_v * v1
        c[v*2 + 1] -= p_v * v2
        c[n_x + v] += p_v

    # Bounds
    bounds = [(0.0, 1.0)] * n_x + [(0.0, None)] * n_u

    A_ub_rows = []
    b_ub_rows = []

    # IC via envelope: u[v] - u[v-e_g] >= x[v-e_g, g]
    for v in range(n_types):
        v1, v2 = int(values[v][0]), int(values[v][1])
        for g in range(2):
            if v[g] > 0:
                ve = list(v)
                ve[g] -= 1
                idx_ve = ve[0] * 4 + ve[1]
                row = np.zeros(n_vars)
                row[n_x + v] = -1.0
                row[n_x + idx_ve] = 1.0
                row[idx_ve * 2 + g] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)

    # u[0,0] = 0
    row = np.zeros(n_vars)
    row[n_x + 0] = -1.0
    A_ub_rows.append(row)
    b_ub_rows.append(0.0)

    # Non-negative payment: -v1*x[v,0] - v2*x[v,1] + u[v] <= 0
    for v in range(n_types):
        v1, v2 = int(values[v][0]), int(values[v][1])
        row = np.zeros(n_vars)
        row[v*2 + 0] = -v1
        row[v*2 + 1] = -v2
        row[n_x + v] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})
        if not res.success:
            return None

        x_alloc = res.x[:n_x].reshape(n_types, 2)
        x_alloc = np.clip(x_alloc, 0.0, 1.0)
        u_flat = res.x[n_x:n_x + n_u]

        payments = np.zeros(n_types)
        for v in range(n_types):
            v1, v2 = int(values[v][0]), int(values[v][1])
            payments[v] = v1 * x_alloc[v, 0] + v2 * x_alloc[v, 1] - u_flat[v]
        payments = np.clip(payments, 0.0, None)

        return (x_alloc, payments)
    except Exception:
        return None
>>>>>>> REPLACE
