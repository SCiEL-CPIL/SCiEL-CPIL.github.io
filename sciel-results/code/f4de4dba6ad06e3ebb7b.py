<<<<<<< SEARCH
def _solve_mechanism_lp(types, n_types, probs, type_to_idx, deadline):
    """
    Solve the mechanism design LP to maximize expected revenue subject to:
    - Feasibility: allocations in [0,1]
    - Individual rationality: expected utility >= 0
    - Incentive compatibility: truth-telling is optimal
    - Non-negative payments
    
    We use the standard formulation where we optimize over allocations x_{i,j}
    (probability of allocating good j to type i) and derive payments from IC/IR.
    
    For a single additive bidder with two goods, the IC constraint for type t
    reporting type t' is:
    x(t')·v(t) - p(t') >= x(t)·v(t) - p(t)
    
    The standard approach: fix x and set p(t) = min over t' of [x(t')·v(t) - u(t')]
    where u(t) is the utility. With IR, u(t) >= 0.
    
    Actually, let's use the direct revelation form. We choose:
    - x_{i,j} for i in types, j in {0,1} (allocation probabilities)
    - p_i for i in types (payments)
    
    Maximize: sum_i probs[i] * p_i
    
    Subject to:
    - 0 <= x_{i,j} <= 1 for all i, j
    - p_i >= 0 for all i
    - IC: for all i, i': 
      x(i')·v(i) - p(i') <= x(i)·v(i) - p(i)
      i.e., p(i) - p(i') <= x(i)·v(i) - x(i')·v(i)
    - IR: for all i:
      x(i)·v(i) - p(i) >= 0
      i.e., p(i) <= x(i)·v(i)
    
    This is an LP. Let's set it up.
    
    Variables: x_{0,0}, x_{0,1}, ..., x_{15,0}, x_{15,1}, p_0, ..., p_{15}
    Total: 2*16 + 16 = 48 variables
    """
    n_goods = 2
    
    # Variable indexing:
    # x[i][j] -> index 2*i + j
    # p[i] -> index 2*n_types + i
    def x_idx(i, j):
        return 2 * i + j
    
    def p_idx(i):
        return 2 * n_types + i
    
    n_vars = 2 * n_types + n_types  # 32 + 16 = 48
    
    # Objective: maximize sum_i probs[i] * p_i
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[p_idx(i)] = -probs[i]
    
    # Inequality constraints: A_ub @ vars <= b_ub
    A_ub_rows = []
    b_ub_rows = []
    
    # IC constraints: for all i, i' (i != i'):
    # p(i) - p(i') <= x(i)·v(i) - x(i')·v(i)
    # p(i) - p(i') - sum_j x(i,j)*v(i,j) + sum_j x(i',j)*v(i',j) <= 0
    for i in range(n_types):
        for ip in range(n_types):
            if i == ip:
                continue
            row = np.zeros(n_vars)
            row[p_idx(i)] = 1
            row[p_idx(ip)] = -1
            for j in range(n_goods):
                row[x_idx(i, j)] = -types[i][j]
                row[x_idx(ip, j)] = types[ip][j]
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
    
    # IR constraints: for all i:
    # p(i) <= x(i)·v(i)
    # p(i) - sum_j x(i,j)*v(i,j) <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[p_idx(i)] = 1
        for j in range(n_goods):
            row[x_idx(i, j)] = -types[i][j]
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
    
    # Non-negativity handled by bounds
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None
    
    # Bounds: 0 <= x_{i,j} <= 1, 0 <= p_i (no upper bound, but we can set a reasonable one)
    # Upper bound for payments: max possible value is 3+3=6, so p <= 6 is safe
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, 6.0))
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            x_vals = res.x[:2*n_types].reshape(n_types, n_goods)
            p_vals = res.x[2*n_types:]
            
            # Clip to ensure feasibility
            x_vals = np.clip(x_vals, 0, 1)
            p_vals = np.clip(p_vals, 0, None)
            
            return {'allocations': x_vals, 'payments': p_vals}
        else:
            return None
    except Exception:
        return None
=======
def _solve_mechanism_lp(types, n_types, probs, type_to_idx, deadline):
    """
    Solve the mechanism design LP to maximize expected revenue subject to:
    - Feasibility: allocations in [0,1]
    - Individual rationality: expected utility >= 0
    - Incentive compatibility: truth-telling is optimal
    - Non-negative payments
    
    We add a small epsilon to the RHS of IC and IR constraints to ensure
    strict feasibility, which helps pass the strict regret<=1e-8 check.
    """
    n_goods = 2
    epsilon = 1e-7  # Safety margin for strict feasibility
    
    # Variable indexing:
    # x[i][j] -> index 2*i + j
    # p[i] -> index 2*n_types + i
    def x_idx(i, j):
        return 2 * i + j
    
    def p_idx(i):
        return 2 * n_types + i
    
    n_vars = 2 * n_types + n_types  # 32 + 16 = 48
    
    # Objective: maximize sum_i probs[i] * p_i
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[p_idx(i)] = -probs[i]
    
    # Inequality constraints: A_ub @ vars <= b_ub
    A_ub_rows = []
    b_ub_rows = []
    
    # IC constraints: for all i, i' (i != i'):
    # p(i) - p(i') <= x(i)·v(i) - x(i')·v(i) - epsilon
    for i in range(n_types):
        for ip in range(n_types):
            if i == ip:
                continue
            row = np.zeros(n_vars)
            row[p_idx(i)] = 1
            row[p_idx(ip)] = -1
            for j in range(n_goods):
                row[x_idx(i, j)] = -types[i][j]
                row[x_idx(ip, j)] = types[ip][j]
            A_ub_rows.append(row)
            b_ub_rows.append(-epsilon)
    
    # IR constraints: for all i:
    # p(i) <= x(i)·v(i) - epsilon
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[p_idx(i)] = 1
        for j in range(n_goods):
            row[x_idx(i, j)] = -types[i][j]
        A_ub_rows.append(row)
        b_ub_rows.append(-epsilon)
    
    # Non-negativity handled by bounds
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None
    
    # Bounds: 0 <= x_{i,j} <= 1, 0 <= p_i
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, 6.0))
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            x_vals = res.x[:2*n_types].reshape(n_types, n_goods)
            p_vals = res.x[2*n_types:]
            
            # Clip to ensure feasibility
            x_vals = np.clip(x_vals, 0, 1)
            p_vals = np.clip(p_vals, 0, None)
            
            return {'allocations': x_vals, 'payments': p_vals}
        else:
            # If strict LP fails, try without epsilon (relaxed)
            # This is a fallback for cases where the feasible region is very tight
            A_ub_rows_relax = []
            b_ub_rows_relax = []
            for i in range(n_types):
                for ip in range(n_types):
                    if i == ip:
                        continue
                    row = np.zeros(n_vars)
                    row[p_idx(i)] = 1
                    row[p_idx(ip)] = -1
                    for j in range(n_goods):
                        row[x_idx(i, j)] = -types[i][j]
                        row[x_idx(ip, j)] = types[ip][j]
                    A_ub_rows_relax.append(row)
                    b_ub_rows_relax.append(0.0)
            for i in range(n_types):
                row = np.zeros(n_vars)
                row[p_idx(i)] = 1
                for j in range(n_goods):
                    row[x_idx(i, j)] = -types[i][j]
                A_ub_rows_relax.append(row)
                b_ub_rows_relax.append(0.0)
            
            A_ub_relax = np.array(A_ub_rows_relax)
            b_ub_relax = np.array(b_ub_rows_relax)
            res_relax = linprog(c, A_ub=A_ub_relax, b_ub=b_ub_relax, bounds=bounds, method='highs')
            if res_relax.success:
                x_vals = res_relax.x[:2*n_types].reshape(n_types, n_goods)
                p_vals = res_relax.x[2*n_types:]
                x_vals = np.clip(x_vals, 0, 1)
                p_vals = np.clip(p_vals, 0, None)
                return {'allocations': x_vals, 'payments': p_vals}
            return None
    except Exception:
        return None
>>>>>>> REPLACE

