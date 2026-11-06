"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction that maximizes expected revenue.
    
    Args:
        distribution_spec: dict with 'probs' - a (16,) array of probabilities for each type
        support: the set of types, each type is (v1, v2) with vi in {0,1,2,3}
        constraints: dict with constraints (may be empty or have specific params)
    
    Returns:
        dict with 'allocations': (16, 2) array, 'payments': (16,) array
    """
    # Types: all combinations of (v1, v2) where vi in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Get probabilities
    if isinstance(distribution_spec, dict) and 'probs' in distribution_spec:
        probs = np.array(distribution_spec['probs'], dtype=float)
    else:
        # Default: uniform
        probs = np.ones(n_types) / n_types
    
    # Ensure probs sum to 1
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
    
    # Variable ordering:
    # 0..15: p1[type] for each type
    # 16..31: p2[type] for each type
    # 32..47: q[type] for each type
    n_vars = 3 * n_types
    
    def idx_p1(t): return t
    def idx_p2(t): return n_types + t
    def idx_q(t): return 2 * n_types + t
    
    # Objective: maximize sum(probs[t] * q[t]) => minimize -sum(probs[t] * q[t])
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[idx_q(t)] = -probs[t]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints: q[t] <= v1[t]*p1[t] + v2[t]*p2[t]
    # => q[t] - v1[t]*p1[t] - v2[t]*p2[t] <= 0
    for t in range(n_types):
        v1, v2 = types[t]
        row = np.zeros(n_vars)
        row[idx_q(t)] = 1.0
        row[idx_p1(t)] = -v1
        row[idx_p2(t)] = -v2
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC constraints: for each type t and misreport t':
    # v1[t]*p1[t'] + v2[t]*p2[t'] - q[t'] <= v1[t]*p1[t] + v2[t]*p2[t] - q[t]
    # => v1[t]*p1[t'] + v2[t]*p2[t'] - q[t'] - v1[t]*p1[t] - v2[t]*p2[t] + q[t] <= 0
    for t in range(n_types):
        v1, v2 = types[t]
        for t_prime in range(n_types):
            if t_prime == t:
                continue
            row = np.zeros(n_vars)
            # Left side: v1[t]*p1[t'] + v2[t]*p2[t'] - q[t']
            row[idx_p1(t_prime)] = v1
            row[idx_p2(t_prime)] = v2
            row[idx_q(t_prime)] = -1.0
            # Right side: -v1[t]*p1[t] - v2[t]*p2[t] + q[t]
            row[idx_p1(t)] = -v1
            row[idx_p2(t)] = -v2
            row[idx_q(t)] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds: 0 <= p1, p2 <= 1; q >= 0 (no upper bound on q, but IR constrains it)
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # p1
    for t in range(n_types):
        bounds.append((0, 1))  # p2
    for t in range(n_types):
        bounds.append((0, None))  # q
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        x = result.x
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for t in range(n_types):
            allocations[t, 0] = x[idx_p1(t)]
            allocations[t, 1] = x[idx_p2(t)]
            payments[t] = x[idx_q(t)]
        
        # Ensure numerical precision
        allocations = np.clip(allocations, 0, 1)
        payments = np.maximum(payments, 0)
        
        return {'allocations': allocations, 'payments': payments}
    else:
        # Fallback: trivial mechanism (no allocation, no payment)
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
# EVOLVE-BLOCK-END
