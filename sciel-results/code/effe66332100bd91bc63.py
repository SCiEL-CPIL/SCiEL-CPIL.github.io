import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    # Define the support: 16 types (v1, v2) with v1, v2 in {0, 1, 2, 3}
    # Order: v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution_spec to get probabilities for each type
    probs = np.zeros(n_types)
    if isinstance(distribution_spec, (list, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
        else:
            # Fallback to uniform if length mismatch
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
            elif str(t) in distribution_spec:
                probs[i] = distribution_spec[str(t)]
            else:
                probs[i] = 1.0 / n_types
    elif isinstance(distribution_spec, str):
        spec = distribution_spec.lower()
        if "uniform" in spec:
            probs = np.ones(n_types) / n_types
        elif "skewed" in spec or "biased" in spec:
            # Example of a skewed distribution: higher values more likely
            # v1, v2 independent, P(v) proportional to (v+1)
            weights = np.array([(v1 + 1) * (v2 + 1) for v1, v2 in types])
            probs = weights / np.sum(weights)
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
        
    # Normalize probabilities
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(n_types) / n_types

    # Variables:
    # x[0..31]: allocations for each type and good. 
    # x[type_idx * 2 + good_idx]
    # p[32..47]: payments for each type.
    # p[type_idx + 32]
    
    n_vars = n_types * n_goods + n_types
    c = np.zeros(n_vars)
    # Maximize revenue: sum(P(v) * p_v)
    # Minimize -revenue
    for i in range(n_types):
        c[32 + i] = -probs[i]
        
    # Constraints: A_ub x <= b_ub
    # 1. x <= 1
    # 2. x >= 0 (handled by bounds)
    # 3. p >= 0 (handled by bounds)
    # 4. IR: p_v <= v . x_v  =>  p_v - v . x_v <= 0
    # 5. IC: p_v - p_v' <= v . x_v - v . x_v'  =>  p_v - p_v' - v . x_v + v . x_v' <= 0
    
    A_ub_rows = []
    b_ub_vals = []
    
    # 1. x <= 1
    for i in range(n_types):
        for g in range(n_goods):
            row = np.zeros(n_vars)
            row[i * n_goods + g] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
            
    # 4. IR: p_v - sum(v_g * x_vg) <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[32 + i] = 1.0 # p_v
        v1, v2 = types[i]
        row[i * n_goods + 0] = -v1
        row[i * n_goods + 1] = -v2
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
        
    # 5. IC: p_v - p_v' - v . x_v + v . x_v' <= 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[32 + i] = 1.0    # p_v
            row[32 + j] = -1.0   # p_v'
            
            v1, v2 = types[i]
            # - v . x_v
            row[i * n_goods + 0] = -v1
            row[i * n_goods + 1] = -v2
            
            # + v . x_v'
            row[j * n_goods + 0] = v1
            row[j * n_goods + 1] = v2
            
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds for variables
    # x: [0, 1]
    # p: [0, inf)
    bounds = []
    for i in range(n_types * n_goods):
        bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
        
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback: Simple VCG-like or uniform allocation?
        # If LP fails, return a valid but potentially suboptimal mechanism.
        # Allocate 1 unit of each good to all types? No, might violate IR if p is low.
        # Safest fallback: Allocate 0, Pay 0. Revenue 0.
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
    else:
        x_opt = res.x[:n_types * n_goods].reshape(n_types, n_goods)
        p_opt = res.x[n_types * n_goods:]
        
        # Clip small numerical errors
        x_opt = np.clip(x_opt, 0, 1)
        p_opt = np.clip(p_opt, 0, None)
        
        allocations = x_opt
        payments = p_opt

    return {
        'allocations': allocations,
        'payments': payments
    }
