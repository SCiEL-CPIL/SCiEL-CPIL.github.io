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
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'weights' in distribution_spec:
            probs = np.array(distribution_spec['weights'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
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
        if probs.size == n_types:
            probs = probs.reshape(n_types)
        else:
            probs = np.ones(n_types) / n_types
    
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n_types) / n_types
    
    # Value matrix: V[t, g] = value of good g for type t
    V = np.array([[t[0], t[1]] for t in types], dtype=float)  # (16, 2)
    
    # Corrected Variable Indexing:
    # Allocation variables x[g, t_rep] depend ONLY on the report t_rep.
    # x[g, t_rep] is the probability of allocating good g when report t_rep is made.
    # p[t_rep] is the payment for report t_rep.
    
    def x_idx(g, t_rep):
        return g * n_types + t_rep
    
    def p_idx(t_rep):
        return 2 * n_types + t_rep
    
    n_vars = 2 * n_types + n_types  # 32 + 16 = 48
    
    # Objective: maximize sum_{t'} probs[t'] * p[t']
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for t_rep in range(n_types):
        c[p_idx(t_rep)] = -probs[t_rep]
    
    A_ub = []
    b_ub = []
    
    # 1. IC Constraints:
    # For all true type t and reported type t_rep (t != t_rep):
    # u(t, t) >= u(t, t_rep)
    # sum_g V[t,g]*x[g,t] - p(t) >= sum_g V[t,g]*x[g,t_rep] - p(t_rep)
    # -sum_g V[t,g]*x[g,t] + sum_g V[t,g]*x[g,t_rep] + p(t) - p(t_rep) <= 0
    
    for t in range(n_types):
        for t_rep in range(n_types):
            if t == t_rep:
                continue
            row = np.zeros(n_vars)
            for g in range(2):
                row[x_idx(g, t)] -= V[t, g]
                row[x_idx(g, t_rep)] += V[t, g]
            row[p_idx(t)] += 1.0
            row[p_idx(t_rep)] -= 1.0
            A_ub.append(row)
            b_ub.append(0.0)
    
    # 2. IR Constraints:
    # u(t, t) >= 0
    # sum_g V[t,g]*x[g,t] - p(t) >= 0
    # -sum_g V[t,g]*x[g,t] + p(t) <= 0
    
    for t in range(n_types):
        row = np.zeros(n_vars)
        for g in range(2):
            row[x_idx(g, t)] -= V[t, g]
        row[p_idx(t)] += 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= x <= 1, p >= 0
    bounds = []
    for g in range(2):
        for t_rep in range(n_types):
            bounds.append((0, 1))
    for t_rep in range(n_types):
        bounds.append((0, None))
    
    # Check deadline
    if time.time() > deadline - 5.0:
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    # Solve LP with tighter tolerances for higher precision
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                        options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                'primal_feasibility_tolerance': 1e-12})
        
        if result.success:
            x_opt = result.x
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            
            for t in range(n_types):
                for g in range(2):
                    allocations[t, g] = x_opt[x_idx(g, t)]
                payments[t] = x_opt[p_idx(t)]
            
            # Clip small numerical errors
            allocations = np.clip(allocations, 0, 1)
            payments = np.maximum(payments, 0)
            
            return {'allocations': allocations, 'payments': payments}
        else:
            # Fallback: simple mechanism (e.g., no allocation, no payment)
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
    
    except Exception as e:
        # Fallback on any error
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
