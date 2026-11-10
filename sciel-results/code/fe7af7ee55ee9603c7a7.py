"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

# EVOLVE-BLOCK-START
def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with 1 additive bidder, 2 goods.
    Values in {0,1,2,3}. Three fixed distributions.
    Returns: {'allocations': (16,2) array of probabilities, 'payments': (16,) array}
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Parse distribution spec
    # Assume distribution_spec specifies the probability distribution over the 16 types
    # Each type is (v1, v2) where v1, v2 in {0,1,2,3}
    
    # Generate all 16 types
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = len(types)  # 16
    
    # Parse probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'])
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if dist == 'uniform':
                probs = np.ones(n_types) / n_types
            elif dist == 'geometric':
                # Example: P(v1=k, v2=l) proportional to 0.7^k * 0.7^l
                probs = np.array([0.7**v1 * 0.7**v2 for v1, v2 in types])
                probs /= probs.sum()
            else:
                probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec)
        if len(probs) == n_types:
            probs = probs / probs.sum()
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # Ensure probabilities are valid
    probs = np.maximum(probs, 0)
    probs = probs / probs.sum()
    
    # Variables:
    # x[i] = probability of allocating good 1 to type i (16 vars)
    # y[i] = probability of allocating good 2 to type i (16 vars)
    # p[i] = payment for type i (16 vars)
    # Total: 48 variables
    # Order: [x_0..x_15, y_0..y_15, p_0..p_15]
    
    n_vars = 3 * n_types  # 48
    
    # Objective: Maximize expected revenue = sum_i probs[i] * p[i]
    # c = [0,...,0, 0,...,0, probs[0],...,probs[15]]
    c = np.zeros(n_vars)
    c[2*n_types:3*n_types] = probs  # Maximize revenue (linprog minimizes, so use -c)
    c = -c  # Negate for minimization
    
    # Constraints:
    # 1. Feasibility: 0 <= x[i] <= 1, 0 <= y[i] <= 1
    # 2. Nonnegative payments: p[i] >= 0
    # 3. Individual rationality: For each type i, expected utility >= 0
    #    u_i = x[i]*v1_i + y[i]*v2_i - p[i] >= 0
    #    => p[i] <= x[i]*v1_i + y[i]*v2_i
    #    => -x[i]*v1_i - y[i]*v2_i + p[i] <= 0
    # 4. Incentive compatibility (truthfulness): For each type i and each possible report j,
    #    u_i(report as j) <= u_i(report truthfully as i)
    #    u_i(report as j) = x[j]*v1_i + y[j]*v2_i - p[j]
    #    u_i(truthful) = x[i]*v1_i + y[i]*v2_i - p[i]
    #    IC: x[j]*v1_i + y[j]*v2_i - p[j] <= x[i]*v1_i + y[i]*v2_i - p[i]
    #    => (x[j]-x[i])*v1_i + (y[j]-y[i])*v2_i - p[j] + p[i] <= 0
    
    # Build inequality constraints A_ub @ vars <= b_ub
    A_rows = []
    b_rows = []
    
    # IR constraints: -v1_i * x[i] - v2_i * y[i] + p[i] <= 0
    for i in range(n_types):
        v1, v2 = types[i]
        row = np.zeros(n_vars)
        row[i] = -v1           # x[i]
        row[n_types + i] = -v2 # y[i]
        row[2*n_types + i] = 1 # p[i]
        A_rows.append(row)
        b_rows.append(0)
    
    # IC constraints: for each i (true type) and j (reported type)
    # (x[j]-x[i])*v1_i + (y[j]-y[i])*v2_i - p[j] + p[i] <= 0
    for i in range(n_types):
        v1_i, v2_i = types[i]
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # x[j] * v1_i
            row[j] = v1_i
            # -x[i] * v1_i
            row[i] = -v1_i
            # y[j] * v2_i
            row[n_types + j] = v2_i
            # -y[i] * v2_i
            row[n_types + i] = -v2_i
            # -p[j]
            row[2*n_types + j] = -1
            # +p[i]
            row[2*n_types + i] = 1
            A_rows.append(row)
            b_rows.append(0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x[i] <= 1, 0 <= y[i] <= 1, p[i] >= 0
    bounds = []
    for i in range(n_types):
        bounds.append((0, 1))  # x[i]
    for i in range(n_types):
        bounds.append((0, 1))  # y[i]
    for i in range(n_types):
        bounds.append((0, None))  # p[i]
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: return a trivial feasible solution
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
        
        x = result.x[:n_types]
        y = result.x[n_types:2*n_types]
        p = result.x[2*n_types:3*n_types]
        
        # Clip to ensure feasibility
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        p = np.maximum(p, 0)
        
        # Verify IR
        for i in range(n_types):
            v1, v2 = types[i]
            utility = x[i]*v1 + y[i]*v2 - p[i]
            if utility < -1e-8:
                # Adjust payment to satisfy IR
                p[i] = max(0, x[i]*v1 + y[i]*v2)
        
        # Verify IC with small tolerance
        for i in range(n_types):
            v1_i, v2_i = types[i]
            for j in range(n_types):
                if i == j:
                    continue
                util_report = x[j]*v1_i + y[j]*v2_i - p[j]
                util_truth = x[i]*v1_i + y[i]*v2_i - p[i]
                if util_report - util_truth > 1e-8:
                    # IC violated; fall back to trivial solution
                    allocations = np.zeros((n_types, 2))
                    payments = np.zeros(n_types)
                    return {'allocations': allocations, 'payments': payments}
        
        allocations = np.column_stack((x, y))
        return {'allocations': allocations, 'payments': p}
        
    except Exception as e:
        # Fallback: return a trivial feasible solution
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
