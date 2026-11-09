"""Incentive-compatible finite auction design: 1 bidder, 2 goods, values in {0,1,2,3}."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    DEADLINE = 85.0

    vals = [0, 1, 2, 3]
    types = list(itertools.product(vals, vals))
    type_index = {t: i for i, t in enumerate(types)}
    n_types = 16
    n_goods = 2

    # Parse distribution
    if isinstance(distribution_spec, dict):
        mu = np.zeros(n_types)
        for t, p in distribution_spec.items():
            if t in type_index:
                mu[type_index[t]] = p
        if mu.sum() == 0:
            mu = np.ones(n_types) / n_types
        else:
            mu = mu / mu.sum()
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        mu = np.array(distribution_spec, dtype=float).flatten()
        if len(mu) == n_types:
            if mu.sum() == 0:
                mu = np.ones(n_types) / n_types
            else:
                mu = mu / mu.sum()
        else:
            mu = np.ones(n_types) / n_types
    else:
        mu = np.ones(n_types) / n_types

    v = np.array([list(r) for r in types], dtype=float)

    # Variables:
    # a[t, g] for t in 0..15, g in 0..1 (32 variables)
    # p[t] for t in 0..15 (16 variables)
    # Total variables: 48
    
    n_vars = 48
    a_idx = lambda t, g: t * 2 + g
    p_idx = lambda t: 32 + t

    # Objective: Maximize sum(mu[t] * p[t])
    # linprog minimizes, so we use c = -mu * p[t]
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[p_idx(t)] = -mu[t]

    # Constraints:
    # 1. Monotonicity: a[t, g] <= a[t', g] if v_g[t'] >= v_g[t]
    # 2. 0 <= a[t, g] <= 1
    # 3. IC: p[t] - p[t'] >= (v[t] - v[t']) . a[t']
    #    => -a[t',0]*v[t,0] - a[t',1]*v[t,1] + p[t] - p[t'] >= (v[t,0]-v[t',0])*a[t',0] + (v[t,1]-v[t',1])*a[t',1]
    #    => p[t] - p[t'] - sum_g (v[t,g] - v[t',g]) * a[t',g] >= 0
    #    => -p[t] + p[t'] + sum_g (v[t,g] - v[t',g]) * a[t',g] <= 0
    # 4. IR: p[t] <= v[t] . a[t]
    #    => -p[t] + sum_g v[t,g] * a[t,g] >= 0
    #    => p[t] - sum_g v[t,g] * a[t,g] <= 0
    # 5. Non-negativity: p[t] >= 0 (handled by bounds)
    # 6. a[t,g] >= 0 (handled by bounds)

    A_ub = []
    b_ub = []

    # Monotonicity constraints
    for t in range(n_types):
        v_t = types[t]
        for g in range(n_goods):
            for t_prime in range(n_types):
                v_tp = types[t_prime]
                if v_tp[g] > v_t[g]:
                    # a[t, g] - a[t_prime, g] <= 0
                    row = np.zeros(n_vars)
                    row[a_idx(t, g)] = 1.0
                    row[a_idx(t_prime, g)] = -1.0
                    A_ub.append(row)
                    b_ub.append(0.0)

    # IC constraints
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            row[p_idx(t)] = -1.0
            row[p_idx(t_prime)] = 1.0
            for g in range(n_goods):
                coeff = v[t, g] - v[t_prime, g]
                row[a_idx(t_prime, g)] += coeff
            A_ub.append(row)
            b_ub.append(0.0)

    # IR constraints
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[p_idx(t)] = 1.0
        for g in range(n_goods):
            row[a_idx(t, g)] -= v[t, g]
        A_ub.append(row)
        b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds
    bounds = []
    for i in range(32): # a[t,g] in [0,1]
        bounds.append((0, 1))
    for i in range(16): # p[t] in [0, inf)
        bounds.append((0, None))

    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            # Extract allocations and payments
            alloc = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            
            for t in range(n_types):
                for g in range(n_goods):
                    alloc[t, g] = res.x[a_idx(t, g)]
                payments[t] = res.x[p_idx(t)]
            
            # Verify and fix small numerical errors
            # Check IR
            for t in range(n_types):
                util = v[t] @ alloc[t] - payments[t]
                if util < -1e-8:
                    # Violation found, adjust payment down? No, IR violation means payment too high.
                    # But LP should have satisfied it. If numerical error, clamp.
                    payments[t] = v[t] @ alloc[t]
            
            # Check IC
            ic_violated = False
            for t in range(n_types):
                for tp in range(n_types):
                    lhs = v[t] @ alloc[t] - payments[t]
                    rhs = v[t] @ alloc[tp] - payments[tp]
                    if lhs - rhs < -1e-8:
                        ic_violated = True
                        break
                if ic_violated:
                    break
            
            if not ic_violated:
                return {
                    'allocations': alloc,
                    'payments': payments
                }
        else:
            # Fallback if LP fails
            pass
            
    except Exception:
        pass

    # Fallback: Simple threshold mechanism (same as seed but robust)
    best_revenue = -1.0
    best_alloc = None
    best_pay = None

    for thresh1 in range(5):
        for thresh2 in range(5):
            alloc = np.zeros((n_types, n_goods))
            for t_idx, (v1, v2) in enumerate(types):
                alloc[t_idx, 0] = 1.0 if v1 >= thresh1 else 0.0
                alloc[t_idx, 1] = 1.0 if v2 >= thresh2 else 0.0

            payments = np.zeros(n_types)
            for t_idx, (v1, v2) in enumerate(types):
                a1_val = alloc[t_idx, 0]
                a2_val = alloc[t_idx, 1]
                sum_a1 = 0.0
                for i in range(1, v1 + 1):
                    t_i = (i, v2)
                    if t_i in type_index:
                        sum_a1 += alloc[type_index[t_i], 0]
                sum_a2 = 0.0
                for j in range(1, v2 + 1):
                    t_j = (v1, j)
                    if t_j in type_index:
                        sum_a2 += alloc[type_index[t_j], 1]
                payments[t_idx] = v1 * a1_val + v2 * a2_val - sum_a1 - sum_a2

            ir_ok = True
            for t_idx in range(n_types):
                util = v[t_idx] @ alloc[t_idx] - payments[t_idx]
                if util < -1e-8:
                    ir_ok = False
                    break
            if not ir_ok:
                continue

            ic_ok = True
            for t_idx in range(n_types):
                for tp_idx in range(n_types):
                    lhs = v[t_idx] @ alloc[t_idx] - payments[t_idx]
                    rhs = v[t_idx] @ alloc[tp_idx] - payments[tp_idx]
                    if lhs - rhs < -1e-8:
                        ic_ok = False
                        break
                if not ic_ok:
                    break
            if not ic_ok:
                continue

            exp_rev = np.sum(mu * payments)
            if exp_rev > best_revenue:
                best_revenue = exp_rev
                best_alloc = alloc.copy()
                best_pay = payments.copy()

    if best_alloc is None:
        best_alloc = np.zeros((n_types, n_goods))
        best_pay = np.zeros(n_types)

    return {
        'allocations': best_alloc,
        'payments': best_pay
    }
# EVOLVE-BLOCK-END
