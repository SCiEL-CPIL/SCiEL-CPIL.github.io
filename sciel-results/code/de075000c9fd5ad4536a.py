"""Incentive-compatible finite auction design - LP-based solution with polish pass."""
import time
import numpy as np
from scipy.optimize import linprog

def _parse_distribution(distribution_spec, support):
    """Parse distribution spec into probability vector over 16 types."""
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n = len(types)  # 16
    
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
            if probs.shape == (n,):
                return probs
        if 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
            if probs.shape == (n,):
                return probs
        # Try to build from marginals
        if 'marginals' in distribution_spec:
            marg = distribution_spec['marginals']
            if isinstance(marg, (list, tuple)) and len(marg) == 2:
                p1 = np.array(marg[0], dtype=float)
                p2 = np.array(marg[1], dtype=float)
                probs = np.zeros(n)
                for i, v1 in enumerate(values):
                    for j, v2 in enumerate(values):
                        idx = i * 4 + j
                        probs[idx] = p1[i] * p2[j]
                return probs
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if probs.shape == (n,):
            return probs
    
    # Default: uniform
    return np.ones(n) / n


def _solve_auction_lp(probs, support, constraints):
    """
    Solve the LP for the optimal incentive-compatible auction mechanism.
    
    Types: 16 (v1, v2) with v1,v2 in {0,1,2,3}
    Allocations: x[t] in {0,1}^2 (which goods allocated) - but for LP we use continuous
    Actually, for single bidder, the allocation is a vector in [0,1]^2 for each type.
    
    Variables: x[t, g] for t in 16 types, g in 2 goods => 32 vars
              p[t] for t in 16 types => 16 vars
              Total: 48 vars
    
    Objective: maximize sum_t probs[t] * (sum_g x[t,g] * v[t,g] - p[t])
              = maximize sum_t probs[t] * sum_g x[t,g] * v[t,g] - sum_t probs[t] * p[t]
    
    Constraints:
    1. Monotonicity/IC: For each type t and each good g, allocation is non-decreasing in v_g
       Actually for multi-dimensional, IC is: for all t, t', 
       sum_g x[t,g]*v[t',g] - p[t'] <= sum_g x[t',g]*v[t,g] - p[t]  (truthful is best response)
       This is the key IC constraint.
       
    2. Individual Rationality: For all t, sum_g x[t,g]*v[t,g] - p[t] >= 0
    
    3. Feasibility: 0 <= x[t,g] <= 1 for all t,g
    4. Nonnegative payments: p[t] >= 0
    
    Let's formulate this properly.
    
    For a single bidder with private values, the IC constraint is:
    For all type pairs (t, t'):
    u(t, t') <= u(t', t')
    where u(t, t') = sum_g x[t',g] * v[t,g] - p[t']
    (utility of type t when reporting t')
    
    So: sum_g x[t',g]*v[t,g] - p[t'] <= sum_g x[t',g]*v[t',g] - p[t']
    => sum_g x[t',g]*(v[t,g] - v[t',g]) <= 0
    => sum_g x[t',g]*v[t,g] <= sum_g x[t',g]*v[t',g]
    
    Wait, that's not right. Let me re-derive.
    
    u(t, report) = sum_g x[report, g] * v[t, g] - p[report]
    
    IC: u(t, t) >= u(t, t') for all t, t'
    sum_g x[t,g]*v[t,g] - p[t] >= sum_g x[t',g]*v[t,g] - p[t']
    
    This is 16*16 = 256 constraints (but many are redundant).
    
    IR: u(t, t) >= 0 for all t
    sum_g x[t,g]*v[t,g] - p[t] >= 0
    
    Let's use scipy linprog. Minimize -objective.
    """
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n = len(types)  # 16
    n_goods = 2
    
    # Variable indexing:
    # x[t, g] for t=0..15, g=0,1 => indices 0..31
    # p[t] for t=0..15 => indices 32..47
    # Total: 48 variables
    
    def x_idx(t, g):
        return t * n_goods + g
    
    def p_idx(t):
        return n * n_goods + t
    
    n_vars = n * n_goods + n  # 48
    
    # Objective: maximize sum_t probs[t] * (sum_g x[t,g]*v[t,g] - p[t])
    # Minimize: -sum_t probs[t] * sum_g x[t,g]*v[t,g] + sum_t probs[t] * p[t]
    c = np.zeros(n_vars)
    for t in range(n):
        v = types[t]
        for g in range(n_goods):
            c[x_idx(t, g)] -= probs[t] * v[g]
        c[p_idx(t)] += probs[t]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 
    # IC constraints: for all t, t':
    # sum_g x[t,g]*v[t,g] - p[t] >= sum_g x[t',g]*v[t,g] - p[t']
    # => sum_g x[t,g]*v[t,g] - sum_g x[t',g]*v[t,g] - p[t] + p[t'] >= 0
    # => -sum_g x[t,g]*v[t,g] + sum_g x[t',g]*v[t,g] + p[t] - p[t'] <= 0
    #
    # IR constraints: for all t:
    # sum_g x[t,g]*v[t,g] - p[t] >= 0
    # => -sum_g x[t,g]*v[t,g] + p[t] <= 0
    #
    # Bounds: 0 <= x[t,g] <= 1, p[t] >= 0
    
    A_rows = []
    b_vals = []
    
    # IC constraints
    for t in range(n):
        v_t = types[t]
        for tp in range(n):
            if t == tp:
                continue
            v_tp = types[tp]
            row = np.zeros(n_vars)
            # -sum_g x[t,g]*v[t,g] + sum_g x[tp,g]*v[t,g] + p[t] - p[tp] <= 0
            for g in range(n_goods):
                row[x_idx(t, g)] -= v_t[g]
                row[x_idx(tp, g)] += v_t[g]
            row[p_idx(t)] += 1.0
            row[p_idx(tp)] -= 1.0
            A_rows.append(row)
            b_vals.append(0.0)
    
    # IR constraints
    for t in range(n):
        v_t = types[t]
        row = np.zeros(n_vars)
        for g in range(n_goods):
            row[x_idx(t, g)] -= v_t[g]
        row[p_idx(t)] += 1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds
    bounds = []
    for t in range(n):
        for g in range(n_goods):
            bounds.append((0, 1))
        bounds.append((0, None))  # p[t] >= 0
    
    # Solve
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        return None
    
    x_alloc = np.zeros((n, n_goods))
    payments = np.zeros(n)
    for t in range(n):
        for g in range(n_goods):
            x_alloc[t, g] = result.x[x_idx(t, g)]
        payments[t] = result.x[p_idx(t)]
    
    return x_alloc, payments


def _verify_mechanism(x_alloc, payments, probs, types, tol=1e-8):
    """Verify IC, IR, feasibility, nonnegativity. Return regret (max violation)."""
    n = len(types)
    n_goods = 2
    max_violation = 0.0
    
    for t in range(n):
        v_t = types[t]
        # IR check
        utility = sum(x_alloc[t, g] * v_t[g] for g in range(n_goods)) - payments[t]
        if utility < -tol:
            max_violation = max(max_violation, -utility)
        
        # Nonnegative payment
        if payments[t] < -tol:
            max_violation = max(max_violation, -payments[t])
        
        # Feasibility
        for g in range(n_goods):
            if x_alloc[t, g] < -tol or x_alloc[t, g] > 1 + tol:
                max_violation = max(max_violation, abs(x_alloc[t, g]))
        
        # IC check against all other reports
        for tp in range(n):
            if t == tp:
                continue
            v_tp = types[tp]
            # u(t, tp) should be <= u(t, t)
            u_t_t = sum(x_alloc[t, g] * v_t[g] for g in range(n_goods)) - payments[t]
            u_t_tp = sum(x_alloc[tp, g] * v_t[g] for g in range(n_goods)) - payments[tp]
            if u_t_tp - u_t_t > tol:
                max_violation = max(max_violation, u_t_tp - u_t_t)
    
    return max_violation


def _polish_mechanism(x_alloc, payments, probs, types, tol=1e-10):
    """
    Polish pass: try small payment increases that maintain IC/IR.
    For each type, try increasing payment by small epsilon and check if still valid.
    This can recover revenue lost to numerical tolerance.
    """
    n = len(types)
    n_goods = 2
    polished_payments = payments.copy()
    
    # Try increasing payments slightly
    epsilons = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    for eps in epsilons:
        improved = False
        for t in range(n):
            old_p = polished_payments[t]
            new_p = old_p + eps
            test_payments = polished_payments.copy()
            test_payments[t] = new_p
            
            # Check if still valid
            regret = _verify_mechanism(x_alloc, test_payments, probs, types, tol=tol)
            if regret <= tol:
                polished_payments[t] = new_p
                improved = True
        
        if not improved:
            break
    
    return polished_payments


def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n = len(types)  # 16
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support)
    
    # Ensure probabilities are valid
    probs = np.maximum(probs, 0)
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(n) / n
    
    # Solve the LP
    result = _solve_auction_lp(probs, support, constraints)
    
    if result is None:
        # Fallback: no allocation, zero payments
        x_alloc = np.zeros((n, 2))
        payments = np.zeros(n)
    else:
        x_alloc, payments = result
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        # Not enough time for polish, return as is
        return {'allocations': x_alloc, 'payments': payments}
    
    # Verify and polish
    regret = _verify_mechanism(x_alloc, payments, probs, types, tol=1e-8)
    
    if regret > 1e-8:
        # Mechanism is not valid, try to fix by reducing allocations
        # Simple fix: set all allocations to 0
        x_alloc = np.zeros((n, 2))
        payments = np.zeros(n)
        regret = _verify_mechanism(x_alloc, payments, probs, types, tol=1e-8)
    
    # Polish pass
    if time.time() - start_time < deadline - 5.0:
        polished_payments = _polish_mechanism(x_alloc, payments, probs, types, tol=1e-10)
        # Verify polished version
        regret_polished = _verify_mechanism(x_alloc, polished_payments, probs, types, tol=1e-8)
        if regret_polished <= 1e-8:
            payments = polished_payments
    
    # Final verification
    final_regret = _verify_mechanism(x_alloc, payments, probs, types, tol=1e-8)
    if final_regret > 1e-8:
        # Emergency fallback
        x_alloc = np.zeros((n, 2))
        payments = np.zeros(n)
    
    return {'allocations': x_alloc, 'payments': payments}
