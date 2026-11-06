"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if isinstance(dist, dict) and 'probs' in dist:
                probs = np.array(dist['probs'], dtype=float)
            elif isinstance(dist, (list, tuple, np.ndarray)):
                probs = np.array(dist, dtype=float)
            else:
                probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Ensure probs is normalized
    probs = probs.flatten()
    if len(probs) != n_types:
        probs = np.ones(n_types) / n_types
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(n_types) / n_types
    
    # Parse support: which reports are allowed
    # support could be a list of types, or a dict, or None (all allowed)
    if support is None:
        allowed_types = list(range(n_types))
    elif isinstance(support, dict):
        if 'types' in support:
            allowed_types = support['types']
        elif 'reports' in support:
            allowed_types = support['reports']
        else:
            allowed_types = list(range(n_types))
    elif isinstance(support, (list, tuple, np.ndarray)):
        allowed_types = list(support)
    else:
        allowed_types = list(range(n_types))
    
    # Convert allowed types to indices
    type_to_idx = {t: i for i, t in enumerate(types)}
    allowed_indices = []
    for at in allowed_types:
        if isinstance(at, (list, tuple)):
            if len(at) == 2 and at[0] in values and at[1] in values:
                idx = type_to_idx.get(tuple(at), None)
                if idx is not None:
                    allowed_indices.append(idx)
        elif isinstance(at, int) and 0 <= at < n_types:
            allowed_indices.append(at)
    
    if not allowed_indices:
        allowed_indices = list(range(n_types))
    
    # Parse constraints
    regret_bound = 1e-8
    if isinstance(constraints, dict):
        if 'regret' in constraints:
            regret_bound = float(constraints['regret'])
        elif 'regret_bound' in constraints:
            regret_bound = float(constraints['regret_bound'])
    
    # Formulate LP over expected utilities u_i (16 vars) and allocation probabilities x_{i,k} (32 vars)
    # Total variables: 16 + 32 = 48
    # Objective: maximize expected revenue = sum_i probs[i] * (sum_k v_{i,k} * x_{i,k} - u_i)
    # 
    # Incentive compatibility via discrete envelope theorem:
    # For all pairs (i, j): u_j <= u_i + sum_k (v_{j,k} - v_{i,k}) * x_{i,k}
    # Rearranged: u_j - u_i - sum_k (v_{j,k} - v_{i,k}) * x_{i,k} <= 0
    #
    # Individual rationality: u_i >= 0 for all i
    #
    # Feasible probabilities: 0 <= x_{i,k} <= 1 for all i, k
    #
    # Nonnegative payments: payment_i = sum_k v_{i,k} * x_{i,k} - u_i >= 0
    # This is equivalent to u_i <= sum_k v_{i,k} * x_{i,k}
    
    # Variable ordering: [u_0, ..., u_15, x_{0,0}, x_{0,1}, x_{1,0}, x_{1,1}, ..., x_{15,0}, x_{15,1}]
    n_u = n_types
    n_x = n_types * n_goods
    n_vars = n_u + n_x
    
    def u_idx(i):
        return i
    
    def x_idx(i, k):
        return n_u + i * n_goods + k
    
    # Objective: maximize sum_i probs[i] * (sum_k v_{i,k} * x_{i,k} - u_i)
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[u_idx(i)] = -probs[i]
        for k in range(n_goods):
            c[x_idx(i, k)] = probs[i] * types[i][k]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub_list = []
    b_ub_list = []
    
    # Incentive compatibility: u_j - u_i - sum_k (v_{j,k} - v_{i,k}) * x_{i,k} <= 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[u_idx(j)] = 1.0
            row[u_idx(i)] = -1.0
            for k in range(n_goods):
                row[x_idx(i, k)] = -(types[j][k] - types[i][k])
            A_ub_list.append(row)
            b_ub_list.append(0.0)
    
    # Nonnegative payments: u_i <= sum_k v_{i,k} * x_{i,k}
    # => u_i - sum_k v_{i,k} * x_{i,k} <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[u_idx(i)] = 1.0
        for k in range(n_goods):
            row[x_idx(i, k)] = -types[i][k]
        A_ub_list.append(row)
        b_ub_list.append(0.0)
    
    # Upper bounds on x: x_{i,k} <= 1
    # Lower bounds: x_{i,k} >= 0, u_i >= 0
    bounds = []
    for i in range(n_types):
        bounds.append((0, None))  # u_i >= 0
    for i in range(n_types):
        for k in range(n_goods):
            bounds.append((0, 1.0))  # 0 <= x_{i,k} <= 1
    
    A_ub = np.array(A_ub_list) if A_ub_list else None
    b_ub = np.array(b_ub_list) if b_ub_list else None
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: return a simple mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_opt = result.x
    u_opt = x_opt[:n_u]
    x_alloc = x_opt[n_u:]
    
    # Extract allocations and payments
    allocations = np.zeros((n_types, n_goods))
    payments = np.zeros(n_types)
    for i in range(n_types):
        for k in range(n_goods):
            allocations[i, k] = x_alloc[i * n_goods + k]
        payments[i] = sum(types[i][k] * allocations[i, k] for k in range(n_goods)) - u_opt[i]
        # Ensure nonnegative payments
        payments[i] = max(0.0, payments[i])
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
