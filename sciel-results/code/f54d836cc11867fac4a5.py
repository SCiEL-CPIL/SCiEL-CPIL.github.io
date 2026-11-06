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
    
    # Parse distribution probabilities
    # distribution_spec should give probabilities for each type
    if isinstance(distribution_spec, dict):
        # Try to extract probabilities from dict
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'weights' in distribution_spec:
            probs = np.array(distribution_spec['weights'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            # Try common keys
            for key in distribution_spec:
                if isinstance(distribution_spec[key], (list, tuple, np.ndarray)):
                    probs = np.array(distribution_spec[key], dtype=float)
                    break
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Ensure probabilities sum to 1
    if probs.shape[0] != n_types:
        # Try to reshape or handle differently
        if probs.size == n_types:
            probs = probs.reshape(n_types)
        else:
            probs = np.ones(n_types) / n_types
    
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n_types) / n_types
    
    # Value matrix: V[t, g] = value of good g for type t
    V = np.array([[t[0], t[1]] for t in types], dtype=float)  # (16, 2)
    
    # Variable indexing:
    # x[g, t, t'] for g in {0,1}, t in {0..15}, t' in {0..15}
    # p[t'] for t' in {0..15}
    # Total variables: 2*16*16 + 16 = 512 + 16 = 528
    
    def x_idx(g, t, tp):
        return g * n_types * n_types + t * n_types + tp
    
    def p_idx(tp):
        return 2 * n_types * n_types + tp
    
    n_vars = 2 * n_types * n_types + n_types
    
    # Objective: maximize sum_{t'} probs[t'] * p[t']
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for tp in range(n_types):
        c[p_idx(tp)] = -probs[tp]
    
    # Constraints:
    # 1. IC: For all true type t and reported type t' (t != t'):
    #    u(t,t) >= u(t,t')
    #    sum_g V[t,g]*x[g,t,t] - p(t) >= sum_g V[t,g]*x[g,t,t'] - p(t')
    #    sum_g V[t,g]*(x[g,t,t] - x[g,t,t']) - p(t) + p(t') >= 0
    #    -sum_g V[t,g]*x[g,t,t] + sum_g V[t,g]*x[g,t,t'] + p(t) - p(t') <= 0
    
    A_ub = []
    b_ub = []
    
    # IC constraints: for each t, t' with t != t'
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            for g in range(2):
                row[x_idx(g, t, t)] -= V[t, g]
                row[x_idx(g, t, tp)] += V[t, g]
            row[p_idx(t)] += 1.0
            row[p_idx(tp)] -= 1.0
            A_ub.append(row)
            b_ub.append(0.0)
    
    # IR constraints: u(t,t) >= 0
    # sum_g V[t,g]*x[g,t,t] - p(t) >= 0
    # -sum_g V[t,g]*x[g,t,t] + p(t) <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        for g in range(2):
            row[x_idx(g, t, t)] -= V[t, g]
        row[p_idx(t)] += 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= x <= 1, p >= 0
    bounds = []
    for g in range(2):
        for t in range(n_types):
            for tp in range(n_types):
                bounds.append((0, 1))
    for tp in range(n_types):
        bounds.append((0, None))
    
    # Check deadline
    if time.time() > deadline - 5.0:
        # Return a simple mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if time.time() > deadline - 2.0:
            if result.success:
                x_opt = result.x
                allocations = np.zeros((n_types, 2))
                payments = np.zeros(n_types)
                for g in range(2):
                    for t in range(n_types):
                        allocations[t, g] = x_opt[x_idx(g, t, t)]
                for tp in range(n_types):
                    payments[tp] = x_opt[p_idx(tp)]
                return {'allocations': allocations, 'payments': payments}
            else:
                allocations = np.zeros((n_types, 2))
                payments = np.zeros(n_types)
                return {'allocations': allocations, 'payments': payments}
        
        if result.success:
            x_opt = result.x
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            for g in range(2):
                for t in range(n_types):
                    allocations[t, g] = x_opt[x_idx(g, t, t)]
            for tp in range(n_types):
                payments[tp] = x_opt[p_idx(tp)]
            
            # Clip small numerical errors
            allocations = np.clip(allocations, 0, 1)
            payments = np.maximum(payments, 0)
            
            return {'allocations': allocations, 'payments': payments}
        else:
            # Fallback: simple mechanism
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
    
    except Exception as e:
        # Fallback on any error
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
