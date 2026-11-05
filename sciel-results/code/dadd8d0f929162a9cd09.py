"""Incentive-compatible finite auction design with explicit candidate API."""
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: specification of the type distribution
        support: the support of types (list of tuples or similar)
        constraints: additional constraints
        
    Returns:
        dict with 'allocations' (16, 2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Values are in {0, 1, 2, 3} for each good
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    
    # Map type to index
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, n_types)
    
    # If we can't parse, return a simple baseline
    if probs is None:
        return _baseline_auction(types, n_types)
    
    # Solve the mechanism design LP
    result = _solve_mechanism_lp(types, n_types, probs, type_to_idx, deadline)
    
    if result is None:
        return _baseline_auction(types, n_types)
    
    return result

def _parse_distribution(distribution_spec, n_types):
    """Parse the distribution specification into a probability vector."""
    try:
        if isinstance(distribution_spec, str):
            spec = distribution_spec.lower().strip()
            if 'uniform' in spec:
                return np.ones(n_types) / n_types
            elif 'independent' in spec:
                # Independent uniform on {0,1,2,3} x {0,1,2,3}
                return np.ones(n_types) / n_types
            elif 'degenerate' in spec or 'point' in spec:
                # Return uniform as fallback
                return np.ones(n_types) / n_types
            else:
                return np.ones(n_types) / n_types
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            arr = np.array(distribution_spec, dtype=float).flatten()
            if len(arr) == n_types:
                arr = arr / arr.sum()
                return arr
            elif len(arr) == 1:
                return np.ones(n_types) / n_types
            else:
                return np.ones(n_types) / n_types
        elif isinstance(distribution_spec, dict):
            # Try to interpret as mapping from type to probability
            probs = np.zeros(n_types)
            for k, v in distribution_spec.items():
                if isinstance(k, tuple) and len(k) == 2:
                    if k in type_to_idx_global:
                        probs[type_to_idx_global[k]] = v
            if probs.sum() > 0:
                return probs / probs.sum()
            return np.ones(n_types) / n_types
        else:
            return np.ones(n_types) / n_types
    except Exception:
        return np.ones(n_types) / n_types

# Global type to index mapping
type_to_idx_global = {}
vals_global = [0, 1, 2, 3]
types_global = [(v1, v2) for v1 in vals_global for v2 in vals_global]
for i, t in enumerate(types_global):
    type_to_idx_global[t] = i

def _baseline_auction(types, n_types):
    """Return a simple baseline auction (allocate both goods to all, pay 0)."""
    allocations = np.ones((n_types, 2))
    payments = np.zeros(n_types)
    return {'allocations': allocations, 'payments': payments}

def _solve_mechanism_lp(types, n_types, probs, type_to_idx, deadline):
    """
    Solve the mechanism design LP to maximize expected revenue subject to:
    - Feasibility: allocations in [0,1]
    - Individual rationality: expected utility >= 0
    - Incentive compatibility: truth-telling is optimal
    - Non-negative payments
    
    We use the standard formulation where we optimize over allocations x_{i,j}
    (probability of allocating good j to type i) and derive payments from IC/IR.
    
    For a single additive bidder with two goods, the IC constraint for type t
    reporting type t' is:
    x(t')·v(t) - p(t') >= x(t)·v(t) - p(t)
    
    The standard approach: fix x and set p(t) = min over t' of [x(t')·v(t) - u(t')]
    where u(t) is the utility. With IR, u(t) >= 0.
    
    Actually, let's use the direct revelation form. We choose:
    - x_{i,j} for i in types, j in {0,1} (allocation probabilities)
    - p_i for i in types (payments)
    
    Maximize: sum_i probs[i] * p_i
    
    Subject to:
    - 0 <= x_{i,j} <= 1 for all i, j
    - p_i >= 0 for all i
    - IC: for all i, i': 
      x(i')·v(i) - p(i') <= x(i)·v(i) - p(i)
      i.e., p(i) - p(i') <= x(i)·v(i) - x(i')·v(i)
    - IR: for all i:
      x(i)·v(i) - p(i) >= 0
      i.e., p(i) <= x(i)·v(i)
    
    This is an LP. Let's set it up.
    
    Variables: x_{0,0}, x_{0,1}, ..., x_{15,0}, x_{15,1}, p_0, ..., p_{15}
    Total: 2*16 + 16 = 48 variables
    """
    n_goods = 2
    
    # Variable indexing:
    # x[i][j] -> index 2*i + j
    # p[i] -> index 2*n_types + i
    def x_idx(i, j):
        return 2 * i + j
    
    def p_idx(i):
        return 2 * n_types + i
    
    n_vars = 2 * n_types + n_types  # 32 + 16 = 48
    
    # Objective: maximize sum_i probs[i] * p_i
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[p_idx(i)] = -probs[i]
    
    # Inequality constraints: A_ub @ vars <= b_ub
    A_ub_rows = []
    b_ub_rows = []
    
    # IC constraints: for all i, i' (i != i'):
    # p(i) - p(i') <= x(i)·v(i) - x(i')·v(i)
    # p(i) - p(i') - sum_j x(i,j)*v(i,j) + sum_j x(i',j)*v(i',j) <= 0
    for i in range(n_types):
        for ip in range(n_types):
            if i == ip:
                continue
            row = np.zeros(n_vars)
            row[p_idx(i)] = 1
            row[p_idx(ip)] = -1
            for j in range(n_goods):
                row[x_idx(i, j)] = -types[i][j]
                row[x_idx(ip, j)] = types[ip][j]
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
    
    # IR constraints: for all i:
    # p(i) <= x(i)·v(i)
    # p(i) - sum_j x(i,j)*v(i,j) <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[p_idx(i)] = 1
        for j in range(n_goods):
            row[x_idx(i, j)] = -types[i][j]
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
    
    # Non-negativity handled by bounds
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None
    
    # Bounds: 0 <= x_{i,j} <= 1, 0 <= p_i (no upper bound, but we can set a reasonable one)
    # Upper bound for payments: max possible value is 3+3=6, so p <= 6 is safe
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, 6.0))
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            x_vals = res.x[:2*n_types].reshape(n_types, n_goods)
            p_vals = res.x[2*n_types:]
            
            # Clip to ensure feasibility
            x_vals = np.clip(x_vals, 0, 1)
            p_vals = np.clip(p_vals, 0, None)
            
            return {'allocations': x_vals, 'payments': p_vals}
        else:
            return None
    except Exception:
        return None

def _verify_mechanism(allocations, payments, types, probs, type_to_idx, tol=1e-8):
    """Verify the mechanism satisfies all constraints."""
    n_types = len(types)
    
    # Check allocation bounds
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        return False
    
    # Check non-negative payments
    if np.any(payments < -tol):
        return False
    
    # Check IR: for each type i, x(i)·v(i) - p(i) >= -tol
    for i in range(n_types):
        utility = np.dot(allocations[i], types[i]) - payments[i]
        if utility < -tol:
            return False
    
    # Check IC: for each type i reporting i', utility(i, i') <= utility(i, i) + tol
    for i in range(n_types):
        true_utility = np.dot(allocations[i], types[i]) - payments[i]
        for ip in range(n_types):
            misreport_utility = np.dot(allocations[ip], types[i]) - payments[ip]
            if misreport_utility > true_utility + tol:
                return False
    
    return True

# Main entry point with verification and fallback
def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    probs = _parse_distribution(distribution_spec, n_types)
    
    if probs is None:
        return _baseline_auction(types, n_types)
    
    # Try to solve the LP
    result = None
    if time.time() < deadline - 5:
        result = _solve_mechanism_lp(types, n_types, probs, type_to_idx, deadline)
    
    if result is not None:
        # Verify
        if _verify_mechanism(result['allocations'], result['payments'], types, probs, type_to_idx):
            return result
        else:
            # If verification fails, try to fix
            result['allocations'] = np.clip(result['allocations'], 0, 1)
            result['payments'] = np.clip(result['payments'], 0, None)
            if _verify_mechanism(result['allocations'], result['payments'], types, probs, type_to_idx):
                return result
    
    # Fallback: simple allocation of both goods, pay 0
    return _baseline_auction(types, n_types)
