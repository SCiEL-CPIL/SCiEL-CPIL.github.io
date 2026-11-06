"""Incentive-compatible finite auction design - combined utility-based LP with decoupled fallback."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: Specification of the type distribution
        support: Support of types (values in {0,1,2,3} for 2 goods)
        constraints: Additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s limit
    
    # Types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = list(product(range(4), range(4)))
    n_types = len(types)
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, constraints)
    
    # First try the fast decoupled single-item approach
    result = _solve_decoupled(types, probs)
    
    if result is not None:
        return result
    
    # If that failed, try the full LP
    if time.time() < deadline - 10.0:
        result = _solve_auction_utility_lp(types, probs, deadline)
    
    if result is not None:
        return result
    
    # If still failed, return baseline
    return _baseline_result(types, probs)

def _parse_distribution(distribution_spec, support, constraints):
    """Parse distribution spec into probability vector over 16 types."""
    n_types = 16
    
    if isinstance(distribution_spec, dict):
        dist_type = distribution_spec.get('type', 'uniform')
        if dist_type == 'uniform':
            return np.ones(n_types) / n_types
        elif dist_type == 'product':
            marginals = distribution_spec.get('marginals', None)
            if marginals is not None and len(marginals) == 2:
                p1, p2 = marginals
                probs = np.zeros(n_types)
                for i, (v1, v2) in enumerate(product(range(4), range(4))):
                    probs[i] = p1[v1] * p2[v2]
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = np.ones(n_types) / n_types
                return probs
            else:
                return np.ones(n_types) / n_types
        elif dist_type == 'explicit':
            explicit_probs = distribution_spec.get('probs', None)
            if explicit_probs is not None and len(explicit_probs) == n_types:
                probs = np.array(explicit_probs, dtype=float)
                probs = np.maximum(probs, 0)
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = np.ones(n_types) / n_types
                return probs
            else:
                return np.ones(n_types) / n_types
    
    # Default: uniform
    return np.ones(n_types) / n_types

def _solve_decoupled(types, probs):
    """
    Solve the auction by decoupling into two independent single-item auctions.
    This is fast and exact for additive values with independent goods.
    """
    n_types = len(types)
    values = [0, 1, 2, 3]
    n_vals = len(values)
    
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    for good in range(2):
        # Get marginal distribution of value for this good
        marg_probs = np.zeros(n_vals)
        for t in range(n_types):
            v = types[t][good]
            marg_probs[v] += probs[t]
        
        # Solve single-item auction for this good
        x, p = _solve_single_item(marg_probs, values)
        
        if x is None or p is None:
            return None
        
        # Assign to all types with this good's value
        for t in range(n_types):
            v = types[t][good]
            allocations[t, good] = x[v]
            payments[t] += p[v]
    
    # Verify the combined mechanism
    if _verify_mechanism(types, probs, allocations, payments, eps=1e-8):
        return {
            'allocations': allocations,
            'payments': payments
        }
    else:
        return None

def _solve_single_item(marg_probs, values):
    """
    Solve the optimal single-item, single-bidder auction with discrete values.
    """
    n_vals = len(values)
    
    # Variables: x[0..3] (allocation probabilities), u[0..3] (utilities)
    # 8 variables total
    
    # Maximize: sum_v marg_probs[v] * (v*x[v] - u[v])
    # linprog minimizes, so negate
    c = np.zeros(2 * n_vals)
    for v in range(n_vals):
        c[v] = -marg_probs[v] * values[v]       # coefficient for x[v]
        c[n_vals + v] = marg_probs[v]           # coefficient for u[v]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # x non-decreasing: x[v] >= x[v-1] for v=1,2,3
    for v in range(1, n_vals):
        row = np.zeros(2 * n_vals)
        row[v] = -1.0
        row[v-1] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC: u[v] - u[v-1] >= x[v] for v=1,2,3
    for v in range(1, n_vals):
        row = np.zeros(2 * n_vals)
        row[n_vals + v] = -1.0
        row[n_vals + v-1] = 1.0
        row[v] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IR: u[0] >= 0
    row = np.zeros(2 * n_vals)
    row[n_vals + 0] = -1.0
    A_ub_rows.append(row)
    b_ub_vals.append(0.0)
    
    # p[v] >= 0: v*x[v] - u[v] >= 0
    for v in range(n_vals):
        row = np.zeros(2 * n_vals)
        row[v] = -values[v]
        row[n_vals + v] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    bounds = []
    for v in range(n_vals):
        bounds.append((0, 1))  # x[v]
    for v in range(n_vals):
        bounds.append((0, None))  # u[v]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'primal_feasibility_tolerance': 1e-12,
                             'dual_feasibility_tolerance': 1e-12, 'ipm_optimality_tolerance': 1e-12})
        
        if not res.success:
            return None, None
        
        x = res.x[:n_vals]
        u = res.x[n_vals:]
        p = np.array([values[v] * x[v] - u[v] for v in range(n_vals)])
        
        # Clean up numerical noise
        x = np.clip(x, 0.0, 1.0)
        p = np.maximum(p, 0.0)
        
        return x, p
        
    except Exception:
        return None, None

def _solve_auction_utility_lp(types, probs, deadline):
    """
    Solve the LP using utility-based formulation.
    
    Variables: u(t) = ex-post utility for type t
    Objective: maximize sum_t probs[t] * (v(t)·x(t) - u(t))
    
    For discrete types with values in {0,1,2,3}^2, we use:
    - Monotonicity: u(t) >= u(t') + (v(t) - v(t'))·x(t') for t >= t'
    - This implies x_j(t) is non-decreasing in v_j
    
    We optimize over u and derive x from the KKT conditions.
    """
    n_types = len(types)
    n_goods = 2
    
    # Variable indexing:
    # x_1(t): index t
    # x_2(t): index n_types + t  
    # u(t): index 2 * n_types + t
    
    n_vars = 3 * n_types
    
    # Objective: maximize sum_t probs[t] * (v(t)·x(t) - u(t))
    # = sum_t probs[t] * (v1(t)*x1(t) + v2(t)*x2(t) - u(t))
    # In linprog, minimize the negative
    c = np.zeros(n_vars)
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        c[t] = -probs[t] * v_t[0]           # x_1(t)
        c[n_types + t] = -probs[t] * v_t[1] # x_2(t)
        c[2 * n_types + t] = probs[t]       # u(t)
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints: u(t) >= u(t') + v(t)·(x(t) - x(t'))
    # Rearranged: u(t) - u(t') - v(t)·x(t) + v(t)·x(t') >= 0
    # For linprog (<= form): -u(t) + u(t') + v(t)·x(t) - v(t)·x(t') <= 0
    
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            
            row = np.zeros(n_vars)
            
            # -v(t)·x(t) + v(t)·x(t')
            row[t] += -v_t[0]
            row[n_types + t] += -v_t[1]
            row[t_prime] += v_t[0]
            row[n_types + t_prime] += v_t[1]
            
            # u(t') - u(t)
            row[2 * n_types + t] += -1.0
            row[2 * n_types + t_prime] += 1.0
            
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR constraints: u(t) >= 0
    # For linprog (<= form): -u(t) <= 0
    
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[2 * n_types + t] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # Non-negativity of payments: p(t) = v(t)·x(t) - u(t) >= 0
    # v(t)·x(t) - u(t) >= 0
    # For linprog (<= form): -v(t)·x(t) + u(t) <= 0
    
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        row = np.zeros(n_vars)
        row[t] += -v_t[0]
        row[n_types + t] += -v_t[1]
        row[2 * n_types + t] += 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # x_1(t)
    for t in range(n_types):
        bounds.append((0, 1))  # x_2(t)
    for t in range(n_types):
        bounds.append((0, None))  # u(t)
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                        options={'presolve': True, 'primal_feasibility_tolerance': 1e-12,
                                'dual_feasibility_tolerance': 1e-12, 'ipm_optimality_tolerance': 1e-12})
        
        if not result.success:
            return None
        
        x = result.x
        
        # Extract allocations and utilities
        allocations = np.zeros((n_types, n_goods))
        utilities = np.zeros(n_types)
        
        for t in range(n_types):
            allocations[t, 0] = x[t]
            allocations[t, 1] = x[n_types + t]
            utilities[t] = x[2 * n_types + t]
        
        # Compute payments: p(t) = v(t)·x(t) - u(t)
        payments = np.zeros(n_types)
        for t in range(n_types):
            v_t = np.array(types[t], dtype=float)
            payments[t] = v_t @ allocations[t] - utilities[t]
        
        # Clip with small epsilon
        eps = 1e-10
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.maximum(payments, 0.0)
        
        # Verify
        if _verify_mechanism(types, probs, allocations, payments, eps=1e-8):
            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            return None
            
    except Exception:
        return None

def _verify_mechanism(types, probs, allocations, payments, eps=1e-8):
    """Verify that the mechanism satisfies all constraints."""
    n_types = len(types)
    
    # Check feasibility
    if np.any(allocations < -eps) or np.any(allocations > 1 + eps):
        return False
    if np.any(payments < -eps):
        return False
    
    # Check IC
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        util_t = v_t @ allocations[t] - payments[t]
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            util_misreport = v_t @ allocations[t_prime] - payments[t_prime]
            if util_misreport > util_t + eps:
                return False
    
    # Check IR
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        util_t = v_t @ allocations[t] - payments[t]
        if util_t < -eps:
            return False
    
    return True

def _baseline_result(types, probs):
    """Return a simple baseline mechanism that is always feasible."""
    n_types = len(types)
    # Allocate nothing, charge nothing - always IR and IC
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
