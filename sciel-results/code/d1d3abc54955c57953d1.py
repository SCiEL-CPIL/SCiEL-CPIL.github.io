import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    # Define the support: 16 types (v1, v2) with v1, v2 in {0, 1, 2, 3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution_spec to get probabilities for each type
    probs = np.zeros(n_types)
    if isinstance(distribution_spec, (list, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
        else:
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

    # New Approach: LP over allocations only, using payment identity
    # Variables: x[i * 2 + g] for i in 0..15, g in 0..1
    n_vars = n_types * n_goods
    
    # Objective: Maximize Revenue = sum_i pi_i * p_i
    # p_i = v_i . x_i - sum_g sum_{m=1}^{v_{i,g}} x_{type(m-1, v_{-g}), g}
    # Coeff for x_{k,g} in objective:
    # It appears in p_i if:
    # 1. k == i: coeff is pi_i * v_{i,g}
    # 2. k is a "predecessor" of i in dimension g: 
    #    i.e., v_{k,g} = v_{i,g} - m for some m>=1? 
    #    Actually, the term is x_{(v_{i,1}, ..., v_{i,g}-m, ...), g}.
    #    So for a fixed variable x_{k,g}, it contributes to p_i with coeff -pi_i if 
    #    v_{k,g} < v_{i,g} and v_{k,h} = v_{i,h} for h != g.
    #    And specifically, it is subtracted for each step from v_{k,g}+1 to v_{i,g}.
    #    Wait, the sum is sum_{m=1}^{v_{i,g}} x_{(v_{i,g}-m, ...), g}.
    #    So x_{k,g} is included in the sum for type i if v_{k,g} < v_{i,g} and v_{k,h} = v_{i,h} for h!=g.
    #    And it is included exactly once for each such i?
    #    Yes, for a fixed k, and fixed i with v_{i,g} > v_{k,g} and v_{i,h}=v_{k,h}, 
    #    x_{k,g} is one of the terms in the sum for i.
    #    So the coeff of x_{k,g} in p_i is -1 if v_{i,g} > v_{k,g} and v_{i,h}=v_{k,h} for h!=g.
    #    Otherwise 0 (except for the direct term).
    
    c = np.zeros(n_vars)
    
    for i in range(n_types):
        v1, v2 = types[i]
        # Direct term: pi_i * (v1 * x_{i,0} + v2 * x_{i,1})
        c[i * 2 + 0] += probs[i] * v1
        c[i * 2 + 1] += probs[i] * v2
        
        # Integral terms:
        # For good 0: sum_{m=1}^{v1} x_{(v1-m, v2), 0}
        # For each m in 1..v1:
        #   k_idx corresponds to type (v1-m, v2)
        #   coeff for x_{k,0} is -pi_i
        for m in range(1, v1 + 1):
            k_v1 = v1 - m
            k_v2 = v2
            k_idx = k_v1 * 4 + k_v2
            c[k_idx * 2 + 0] -= probs[i]
            
        # For good 1: sum_{m=1}^{v2} x_{(v1, v2-m), 1}
        for m in range(1, v2 + 1):
            k_v1 = v1
            k_v2 = v2 - m
            k_idx = k_v1 * 4 + k_v2
            c[k_idx * 2 + 1] -= probs[i]

    # Constraints:
    # 1. Monotonicity: x_{i,g} >= x_{j,g} if v_i >= v_j
    A_ub_rows = []
    b_ub_vals = []
    
    for g in range(n_goods):
        for i in range(n_types):
            for j in range(n_types):
                vi = types[i]
                vj = types[j]
                # Check if vi >= vj component-wise
                if vi[0] >= vj[0] and vi[1] >= vj[1]:
                    if i != j: # If i==j, constraint is 0>=0, trivial
                        # x_{i,g} - x_{j,g} >= 0  =>  -x_{i,g} + x_{j,g} <= 0
                        row = np.zeros(n_vars)
                        row[i * 2 + g] = -1.0
                        row[j * 2 + g] = 1.0
                        A_ub_rows.append(row)
                        b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds
    bounds = [(0, 1)] * n_vars
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
    else:
        x_opt = res.x.reshape(n_types, n_goods)
        x_opt = np.clip(x_opt, 0, 1)
        
        # Compute payments using the identity
        payments = np.zeros(n_types)
        for i in range(n_types):
            v1, v2 = types[i]
            p_val = v1 * x_opt[i, 0] + v2 * x_opt[i, 1]
            
            # Subtract integral for good 0
            for m in range(1, v1 + 1):
                k_idx = (v1 - m) * 4 + v2
                p_val -= x_opt[k_idx, 0]
                
            # Subtract integral for good 1
            for m in range(1, v2 + 1):
                k_idx = v1 * 4 + (v2 - m)
                p_val -= x_opt[k_idx, 1]
                
            payments[i] = p_val
        
        # Clip payments to be non-negative (should be true, but safeguard)
        payments = np.clip(payments, 0, None)
        
        allocations = x_opt

    return {
        'allocations': allocations,
        'payments': payments
    }
