# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods,
    values in {0,1,2,3}, with 16 types.
    
    Returns:
        allocations: (16, 2) array of allocation probabilities
        payments: (16,) array of expected payments
    """
    t_start = time.time()
    deadline = t_start + 80.0
    
    # Parse the 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    # Type index: idx = v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    
    # Extract probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    probs = np.array(probs, dtype=float).flatten()
    if len(probs) != n_types:
        probs = np.ones(n_types) / n_types
    probs = np.clip(probs, 0, None)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_types) / n_types
    
    values = np.array([[v1, v2] for v1, v2 in types], dtype=float)
    
    n_x = n_types * 2
    n_p = n_types
    n_vars = n_x + n_p
    
    # Objective: maximize expected revenue
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_x + t] = -probs[t]
    
    # Build constraints
    n_constraints = n_x + n_p + n_types * n_types
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    row = 0
    # x[t,g] <= 1
    for t in range(n_types):
        for g in range(2):
            var_idx = t * 2 + g
            A_ub[row, var_idx] = 1.0
            b_ub[row] = 1.0
            row += 1
    
    # IR: p[t] - sum_g x[t,g]*v[t,g] <= 0
    for t in range(n_types):
        for g in range(2):
            var_idx = t * 2 + g
            A_ub[row, var_idx] = -values[t, g]
        A_ub[row, n_x + t] = 1.0
        b_ub[row] = 0.0
        row += 1
    
    # IC: for all t, t'
    for t in range(n_types):
        for t_prime in range(n_types):
            for g in range(2):
                A_ub[row, t_prime * 2 + g] = values[t, g]
                A_ub[row, t * 2 + g] = -values[t, g]
            A_ub[row, n_x + t] = 1.0
            A_ub[row, n_x + t_prime] = -1.0
            b_ub[row] = 0.0
            row += 1
    
    bounds = [(0, 1)] * n_x + [(0, None)] * n_p
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-9,
                              'primal_feasibility_tolerance': 1e-9})
    
    if result.status != 0:
        allocations = np.ones((n_types, 2)) / 2
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_flat = result.x[:n_x].reshape(n_types, 2)
    p = result.x[n_x:]
    
    # Clip with small margin to ensure strict feasibility
    eps = 1e-9
    x_flat = np.clip(x_flat, eps, 1.0 - eps)
    p = np.clip(p, eps, None)
    
    return {'allocations': x_flat, 'payments': p}
# EVOLVE-BLOCK-END
