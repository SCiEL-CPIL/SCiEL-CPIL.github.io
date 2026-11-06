<<<<<<< SEARCH
def _solve_lp(probs, types, n_types, n_goods):
    """Solve the revenue-maximizing LP for a given probability distribution."""
    n_vars = n_types * n_goods + n_types
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[n_types * n_goods + i] = -probs[i]

    A_ub = []
    b_ub = []

    # IC constraints: p_i - p_j + v_i . (x_j - x_i) <= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i * n_goods + 0] = -vi[0]
            row[i * n_goods + 1] = -vi[1]
            row[j * n_goods + 0] = vi[0]
            row[j * n_goods + 1] = vi[1]
            row[n_types * n_goods + i] = 1.0
            row[n_types * n_goods + j] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # IR constraints: p_i - v_i . x_i <= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        row = np.zeros(n_vars)
        row[i * n_goods + 0] = -vi[0]
        row[i * n_goods + 1] = -vi[1]
        row[n_types * n_goods + i] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    bounds = []
    for _ in range(n_types * n_goods):
        bounds.append((0.0, 1.0))
    for _ in range(n_types):
        bounds.append((0.0, None))

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                  options={'presolve': True,
                           'dual_feasibility_tolerance': 1e-12,
                           'primal_feasibility_tolerance': 1e-12})
    if not res.success:
        return None, None

    x_flat = res.x[:n_types * n_goods]
    p_flat = res.x[n_types * n_goods:]
    allocations = x_flat.reshape(n_types, n_goods)
    payments = p_flat.copy()
    return allocations, payments
=======
def _solve_lp(probs, types, n_types, n_goods):
    """Solve the revenue-maximizing LP using utility variables for better conditioning."""
    # Variables: x_{i,k} for i in range(n_types), k in range(n_goods), then u_i for i in range(n_types)
    # Total: n_types * n_goods + n_types
    n_x = n_types * n_goods
    n_u = n_types
    n_vars = n_x + n_u
    
    # Objective: maximize sum_i probs[i] * (v_i . x_i - u_i)
    # = sum_i probs[i] * sum_k v_{i,k} x_{i,k} - sum_i probs[i] * u_i
    c = np.zeros(n_vars)
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for k in range(n_goods):
            c[i * n_goods + k] = probs[i] * vi[k]
        c[n_x + i] = -probs[i]
    
    A_ub = []
    b_ub = []
    
    # IC constraints: u_j <= u_i + v_j . x_i - v_i . x_i
    # Rearranged: u_j - u_i - v_j . x_i + v_i . x_i <= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j], dtype=float)
            row = np.zeros(n_vars)
            # -v_j . x_i + v_i . x_i
            for k in range(n_goods):
                row[i * n_goods + k] = vi[k] - vj[k]
            # u_j - u_i
            row[n_x + j] = 1.0
            row[n_x + i] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
    
    # IR constraints: u_i >= 0 (since p_i = v_i . x_i - u_i and p_i >= 0 means u_i <= v_i . x_i)
    # Actually IR is u_i >= 0. But we also need p_i >= 0 which is u_i <= v_i . x_i
    # Let's add both:
    # 1) u_i >= 0 -> handled by lower bound on u_i
    # 2) p_i >= 0 -> v_i . x_i - u_i >= 0 -> u_i - v_i . x_i <= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        row = np.zeros(n_vars)
        for k in range(n_goods):
            row[i * n_goods + k] = -vi[k]
        row[n_x + i] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    bounds = []
    for _ in range(n_x):
        bounds.append((0.0, 1.0))
    for _ in range(n_u):
        bounds.append((0.0, None))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                  options={'presolve': True,
                           'dual_feasibility_tolerance': 1e-12,
                           'primal_feasibility_tolerance': 1e-12})
    if not res.success:
        return None, None
    
    x_flat = res.x[:n_x]
    u_flat = res.x[n_x:]
    allocations = x_flat.reshape(n_types, n_goods)
    
    # Compute payments: p_i = v_i . x_i - u_i
    payments = np.zeros(n_types)
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        payments[i] = np.dot(allocations[i], vi) - u_flat[i]
    
    # Clamp small numerical errors
    payments = np.maximum(payments, 0.0)
    
    return allocations, payments
>>>>>>> REPLACE

