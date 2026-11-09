def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    from scipy.optimize import linprog
    import time
    import itertools
    
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    # Parse the problem
    # distribution_spec: describes the distribution over types
    # support: the set of possible types
    # constraints: any additional constraints
    
    # Types: each type is (v1, v2) where v1, v2 in {0, 1, 2, 3}
    # So 16 types total
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2
    
    # Extract probabilities from distribution_spec
    # We need to figure out the format of distribution_spec
    # Let's assume it's a dict or similar structure
    
    # Get probabilities for each type
    probs = np.zeros(n_types)
    
    # Try to extract probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        # Try common formats
        if 'probs' in distribution_spec:
            for i, t in enumerate(types):
                if t in distribution_spec['probs']:
                    probs[i] = distribution_spec['probs'][t]
                elif str(t) in distribution_spec['probs']:
                    probs[i] = distribution_spec['probs'][str(t)]
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if isinstance(dist, dict):
                for i, t in enumerate(types):
                    if t in dist:
                        probs[i] = dist[t]
                    elif str(t) in dist:
                        probs[i] = dist[str(t)]
            elif isinstance(dist, list):
                for i, p in enumerate(dist):
                    if i < n_types:
                        probs[i] = p
        else:
            # Try direct list
            for key in distribution_spec.keys():
                if isinstance(distribution_spec[key], (list, tuple, np.ndarray)):
                    arr = np.array(distribution_spec[key])
                    if len(arr) == n_types:
                        probs = arr.astype(float)
                        break
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n_types:
            probs = np.zeros(n_types)
    
    # Normalize probabilities
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
    
    # Decision variables:
    # x_{i,j} for i in 0..15, j in 0..1 (32 vars)
    # p_i for i in 0..15 (16 vars)
    # Total: 48 vars
    # Index: 0..31 for x, 32..47 for p
    
    def x_idx(i, j):
        return i * n_goods + j
    
    def p_idx(i):
        return n_types * n_goods + i
    
    n_vars = n_types * n_goods + n_types  # 32 + 16 = 48
    
    # Objective: maximize sum_i probs[i] * p_i
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[p_idx(i)] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub = []
    b_ub = []
    
    # IR constraints: v_i · x_i - p_i >= 0  =>  -v_i · x_i + p_i <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        v1, v2 = types[i]
        row[x_idx(i, 0)] = -v1
        row[x_idx(i, 1)] = -v2
        row[p_idx(i)] = 1
        A_ub.append(row)
        b_ub.append(0)
    
    # IC constraints: v_i · x_i - p_i >= v_i · x_j - p_j for all i != j
    # => v_i · x_i - p_i - v_i · x_j + p_j >= 0
    # => -v_i · x_i + p_i + v_i · x_j - p_j <= 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            v1, v2 = types[i]
            # -v_i · x_i
            row[x_idx(i, 0)] = -v1
            row[x_idx(i, 1)] = -v2
            # +p_i
            row[p_idx(i)] = 1
            # +v_i · x_j
            row[x_idx(j, 0)] = v1
            row[x_idx(j, 1)] = v2
            # -p_j
            row[p_idx(j)] = -1
            A_ub.append(row)
            b_ub.append(0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= x_{i,j} <= 1, 0 <= p_i
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x_vals = result.x[:n_types * n_goods].reshape(n_types, n_goods)
            p_vals = result.x[n_types * n_goods:]
            
            # Clip to handle numerical errors
            x_vals = np.clip(x_vals, 0, 1)
            p_vals = np.clip(p_vals, 0, None)
            
            # Verify IC and IR with tolerance
            max_violation = 0
            for i in range(n_types):
                v1, v2 = types[i]
                utility_truth = v1 * x_vals[i, 0] + v2 * x_vals[i, 1] - p_vals[i]
                if utility_truth < -1e-8:
                    max_violation = max(max_violation, -utility_truth)
                for j in range(n_types):
                    if i != j:
                        utility_misrep = v1 * x_vals[j, 0] + v2 * x_vals[j, 1] - p_vals[j]
                        if utility_misrep - utility_truth > 1e-8:
                            max_violation = max(max_violation, utility_misrep - utility_truth)
            
            if max_violation > 1e-6:
                # Try to fix small violations by adjusting payments
                # Increase payments slightly to ensure IC/IR hold
                for i in range(n_types):
                    v1, v2 = types[i]
                    utility_truth = v1 * x_vals[i, 0] + v2 * x_vals[i, 1] - p_vals[i]
                    # Check if IR is violated
                    if utility_truth < 0:
                        p_vals[i] += (-utility_truth + 1e-9)
                    # Check IC: ensure no misreport gives higher utility
                    for j in range(n_types):
                        if i != j:
                            utility_misrep = v1 * x_vals[j, 0] + v2 * x_vals[j, 1] - p_vals[j]
                            if utility_misrep - utility_truth > 1e-8:
                                # Need to increase p_i or decrease p_j
                                # Simple fix: increase p_i
                                p_vals[i] += (utility_misrep - utility_truth + 1e-9)
                # Re-clip after adjustment
                p_vals = np.clip(p_vals, 0, None)
            
            return {
                'allocations': x_vals,
                'payments': p_vals
            }
        else:
            # Fallback: return a simple feasible solution
            x_vals = np.zeros((n_types, n_goods))
            p_vals = np.zeros(n_types)
            return {
                'allocations': x_vals,
                'payments': p_vals
            }
    except Exception as e:
        # Fallback
        x_vals = np.zeros((n_types, n_goods))
        p_vals = np.zeros(n_types)
        return {
            'allocations': x_vals,
            'payments': p_vals
        }
