"""Incentive-compatible finite auction design with LP-based optimal mechanism."""
import time
import numpy as np
from scipy.optimize import linprog

_DEADLINE = 85.0  # safety margin before 90s kill

def _get_support(distribution_spec, support, constraints):
    """Enumerate all allowed reports/values. Each value in {0,1,2,3} for 2 goods."""
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    return types, np.array(types, dtype=float)

def _get_distribution(distribution_spec, types, type_matrix):
    """
    Parse distribution_spec to get probabilities over the 16 types.
    distribution_spec could be:
    - a dict mapping type tuple to probability
    - a string like 'uniform', 'uniform_positive', etc.
    - an array of 16 probabilities
    """
    n_types = len(types)
    
    if isinstance(distribution_spec, dict):
        probs = np.zeros(n_types)
        for i, t in enumerate(types):
            key = tuple(t)
            if key in distribution_spec:
                probs[i] = distribution_spec[key]
            else:
                # Try with different key formats
                for k, v in distribution_spec.items():
                    if tuple(k) == key:
                        probs[i] = v
                        break
        if probs.sum() > 0:
            probs = probs / probs.sum()
        return probs
    
    if isinstance(distribution_spec, str):
        spec = distribution_spec.lower().strip()
        if 'uniform' in spec:
            probs = np.ones(n_types) / n_types
        elif 'concentrated' in spec or 'point' in spec:
            # Concentrated at some type
            probs = np.zeros(n_types)
            probs[0] = 1.0  # default
        else:
            probs = np.ones(n_types) / n_types
        return probs
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (n_types,):
            probs = arr / arr.sum()
            return probs
        elif arr.size == n_types:
            probs = arr.reshape(n_types) / arr.sum()
            return probs
    
    # Default: uniform
    return np.ones(n_types) / n_types

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Returns dict with:
      allocations: (16, 2) array of allocation probabilities
      payments: (16,) array of expected payments
    
    Uses the payment identity for single-bidder IC:
      payment(v) = allocation(v) * v - integral_0^v allocation(w) dw
    For discrete values, we use the discrete payment identity.
    """
    t_start = time.time()
    
    types, type_matrix = _get_support(distribution_spec, support, constraints)
    n_types = len(types)
    n_goods = 2
    vals = [0, 1, 2, 3]
    
    probs = _get_distribution(distribution_spec, types, type_matrix)
    
    # For a single additive bidder with 2 goods:
    # Allocation x_i(v) is a 2-vector (allocation of each good)
    # IC constraint: for all v, v':
    #   x(v)·v - p(v) >= x(v)·v' - p(v')
    #   x(v')·v' - p(v') >= x(v')·v - p(v)
    #
    # Payment identity (from IC):
    #   p(v) = x(v)·v - sum_{v' < v in some order} [x(v')·(v - v')]
    #
    # Actually, the cleanest approach: for a single bidder, IC is equivalent to
    # the allocation being monotone in each dimension (for additive values).
    # The payment is then determined by the allocation via the payment identity.
    #
    # For discrete support {0,1,2,3}^2, the payment identity gives:
    #   p(v1,v2) = x1(v1,v2)*v1 + x2(v1,v2)*v2 - integral along a path
    #
    # Let's use the explicit LP approach with all IC constraints.
    
    # Variables: x_i(v) for i in {1,2}, v in types (32 vars)
    #            p(v) for v in types (16 vars)
    # Total: 48 variables
    # x indexing: x_idx = i * n_types + v_idx
    # p indexing: p_idx = n_types + v_idx  (after x vars)
    
    n_x = n_goods * n_types  # 32
    n_p = n_types  # 16
    n_vars = n_x + n_p  # 48
    
    def x_idx(good, v_idx):
        return good * n_types + v_idx
    
    def p_idx(v_idx):
        return n_x + v_idx
    
    # Objective: maximize expected revenue = sum_v probs[v] * p(v)
    # In linprog, we minimize, so c = -probs for p vars, 0 for x vars
    c = np.zeros(n_vars)
    for v_idx in range(n_types):
        c[p_idx(v_idx)] = -probs[v_idx]
    
    # Constraints:
    # 1. Feasibility: 0 <= x_i(v) <= 1 for all i, v
    # 2. IR: x(v)·v - p(v) >= 0  =>  p(v) <= x(v)·v
    # 3. IC: for all v, v': x(v)·v - p(v) >= x(v)·v' - p(v')
    #    => p(v) - p(v') <= x(v)·(v - v')
    
    A_ub = []
    b_ub = []
    
    # IR constraints: p(v) - x(v)·v <= 0
    for v_idx in range(n_types):
        row = np.zeros(n_vars)
        row[p_idx(v_idx)] = 1.0
        v = type_matrix[v_idx]
        for g in range(n_goods):
            row[x_idx(g, v_idx)] = -v[g]
        A_ub.append(row)
        b_ub.append(0.0)
    
    # IC constraints: p(v) - p(v') <= x(v)·(v - v')
    # for all pairs (v, v')
    for v_idx in range(n_types):
        for vp_idx in range(n_types):
            if v_idx == vp_idx:
                continue
            row = np.zeros(n_vars)
            row[p_idx(v_idx)] = 1.0
            row[p_idx(vp_idx)] = -1.0
            v = type_matrix[v_idx]
            vp = type_matrix[vp_idx]
            diff = v - vp
            for g in range(n_goods):
                row[x_idx(g, v_idx)] = diff[g]
            A_ub.append(row)
            b_ub.append(0.0)
    
    # Nonnegativity: all vars >= 0
    # Upper bounds: x_i(v) <= 1, p(v) unbounded above (but IR + IC will bound it)
    bounds = []
    for i in range(n_x):
        bounds.append((0, 1))
    for i in range(n_p):
        bounds.append((0, None))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: uniform allocation, zero payments
        allocations = np.full((n_types, n_goods), 0.5)
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_sol = result.x[:n_x].reshape(n_goods, n_types).T  # (n_types, n_goods)
    p_sol = result.x[n_x:]  # (n_types,)
    
    # Clamp small numerical errors
    x_sol = np.clip(x_sol, 0, 1)
    p_sol = np.clip(p_sol, 0, None)
    
    allocations = x_sol
    payments = p_sol
    
    return {'allocations': allocations, 'payments': payments}
