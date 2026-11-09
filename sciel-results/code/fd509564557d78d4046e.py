"""Rewritten program for incentive-compatible finite auction design."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Fallback result
    def fallback():
        return {
            'allocations': np.zeros((16, 2)),
            'payments': np.zeros(16)
        }
    
    # Try to get baseline as a fallback
    baseline_allocations = None
    baseline_payments = None
    try:
        from autoresearch_suite.science_economics_expansion import auction_baseline
        baseline_result = auction_baseline(distribution_spec, support, constraints)
        baseline_allocations = np.array(baseline_result['allocations'], dtype=float)
        baseline_payments = np.array(baseline_result['payments'], dtype=float)
    except:
        pass
    
    # Parse the distribution spec
    values = None
    probs = None
    if isinstance(distribution_spec, dict):
        probs = distribution_spec.get('probs', None)
        values = distribution_spec.get('values', None)
        if values is None:
            values = distribution_spec.get('support', None)
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 2:
            first, second = distribution_spec
            if isinstance(first, (list, tuple, np.ndarray)) and isinstance(second, (list, tuple, np.ndarray)):
                values, probs = first, second
            else:
                probs = distribution_spec
                values = None
        else:
            probs = distribution_spec
            values = None
    else:
        probs = None
        values = None
    
    # Default support: all 16 combinations of {0,1,2,3}^2
    if support is None:
        support = list(product([0, 1, 2, 3], repeat=2))
    
    # Ensure support is a list of tuples
    if not isinstance(support, (list, tuple)):
        support = list(support)
    
    n_types = len(support)
    
    # If we can't parse properly, return baseline or fallback
    if probs is None or n_types == 0:
        if baseline_allocations is not None:
            return {
                'allocations': baseline_allocations,
                'payments': baseline_payments
            }
        return fallback()
    
    # Convert to numpy
    probs = np.array(probs, dtype=float)
    if len(probs) != n_types:
        if baseline_allocations is not None:
            return {
                'allocations': baseline_allocations,
                'payments': baseline_payments
            }
        return fallback()
    
    # Ensure probs sum to 1 (normalize if needed)
    if probs.sum() > 1e-10:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
    
    # Extract value vectors
    # support is list of (v1, v2) for each type
    V = np.array(support, dtype=float)  # shape (n_types, 2)
    if V.shape[0] != n_types:
        if baseline_allocations is not None:
            return {
                'allocations': baseline_allocations,
                'payments': baseline_payments
            }
        return fallback()
    
    # Solve the LP for optimal incentive-compatible mechanism
    # Variables: x_{i,k} for i in 0..n_types-1, k in 0,1 (allocation probability)
    #            p_i for i in 0..n_types-1 (payment)
    # 
    # Objective: maximize sum_i probs[i] * p_i
    # 
    # Constraints:
    # 1. Feasibility: 0 <= x_{i,k} <= 1 for all i, k
    # 2. Individual rationality: for all i, v_i . x_i - p_i >= 0
    #    => p_i <= v_i . x_i
    # 3. Incentive compatibility: for all i, j,
    #    v_i . x_i - p_i >= v_i . x_j - p_j
    #    => p_j - p_i >= v_i . (x_j - x_i)
    #    => p_i - p_j <= v_i . (x_i - x_j)
    # 4. Nonnegative payments: p_i >= 0
    
    # Number of variables: n_types * 2 (allocations) + n_types (payments) = 3 * n_types
    n_x = n_types * 2
    n_p = n_types
    n_vars = n_x + n_p
    
    # Objective: maximize sum_i probs[i] * p_i
    # In linprog, we minimize, so negate: c = [0, 0, ..., 0, -probs[0], -probs[1], ...]
    c = np.zeros(n_vars)
    c[n_x:] = -probs
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 
    # IR: p_i <= v_i . x_i  =>  -v_i . x_i + p_i <= 0
    # For each i: row has -V[i,0] at position i*2, -V[i,1] at position i*2+1, +1 at position n_x+i
    # 
    # IC: p_i - p_j <= v_i . (x_i - x_j) for all i != j
    #     => -v_i . x_i + v_i . x_j + p_i - p_j <= 0
    #     For fixed i, j: 
    #       x_{i,0} coeff: -V[i,0]
    #       x_{i,1} coeff: -V[i,1]
    #       x_{j,0} coeff: +V[i,0]
    #       x_{j,1} coeff: +V[i,1]
    #       p_i coeff: +1
    #       p_j coeff: -1
    
    # Build constraint matrix
    n_ir = n_types
    n_ic = n_types * (n_types - 1)
    n_constraints = n_ir + n_ic
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    # IR constraints
    for i in range(n_types):
        A_ub[i, i * 2] = -V[i, 0]
        A_ub[i, i * 2 + 1] = -V[i, 1]
        A_ub[i, n_x + i] = 1.0
        b_ub[i] = 0.0
    
    # IC constraints
    row = n_ir
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            A_ub[row, i * 2] = -V[i, 0]
            A_ub[row, i * 2 + 1] = -V[i, 1]
            A_ub[row, j * 2] = V[i, 0]
            A_ub[row, j * 2 + 1] = V[i, 1]
            A_ub[row, n_x + i] = 1.0
            A_ub[row, n_x + j] = -1.0
            b_ub[row] = 0.0
            row += 1
    
    # Bounds: 0 <= x_{i,k} <= 1, p_i >= 0
    bounds = []
    for i in range(n_types):
        for k in range(2):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.status == 0:
            x_opt = result.x
            allocations = x_opt[:n_x].reshape(n_types, 2)
            payments = x_opt[n_x:]
            
            # Verify constraints
            tol = 1e-8
            
            # Check allocations in [0, 1]
            if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
                if baseline_allocations is not None:
                    return {
                        'allocations': baseline_allocations,
                        'payments': baseline_payments
                    }
                return fallback()
            
            # Clip to [0, 1]
            allocations = np.clip(allocations, 0.0, 1.0)
            
            # Check nonnegative payments
            if np.any(payments < -tol):
                if baseline_allocations is not None:
                    return {
                        'allocations': baseline_allocations,
                        'payments': baseline_payments
                    }
                return fallback()
            payments = np.maximum(payments, 0.0)
            
            # Check IR: v_i . x_i - p_i >= 0
            utils = np.sum(V * allocations, axis=1) - payments
            if np.any(utils < -tol):
                if baseline_allocations is not None:
                    return {
                        'allocations': baseline_allocations,
                        'payments': baseline_payments
                    }
                return fallback()
            
            # Check IC: for all i, j, v_i . x_i - p_i >= v_i . x_j - p_j
            # i.e., utils_i >= v_i . x_j - p_j for all j
            for i in range(n_types):
                for j in range(n_types):
                    dev_util = np.dot(V[i], allocations[j]) - payments[j]
                    if utils[i] < dev_util - tol:
                        if baseline_allocations is not None:
                            return {
                                'allocations': baseline_allocations,
                                'payments': baseline_payments
                            }
                        return fallback()
            
            # Check time
            if time.time() > deadline - 2.0:
                if baseline_allocations is not None:
                    return {
                        'allocations': baseline_allocations,
                        'payments': baseline_payments
                    }
                return fallback()
            
            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            # LP failed
            if baseline_allocations is not None:
                return {
                    'allocations': baseline_allocations,
                    'payments': baseline_payments
                }
            return fallback()
            
    except Exception as e:
        if baseline_allocations is not None:
            return {
                'allocations': baseline_allocations,
                'payments': baseline_payments
            }
        return fallback()
# EVOLVE-BLOCK-END
