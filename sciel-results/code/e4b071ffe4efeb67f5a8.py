# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def _get_probs(distribution_spec, n_types):
    """Extract type probabilities from distribution_spec (robust to several formats)."""
    if distribution_spec is None:
        return np.full(n_types, 1.0 / n_types)
    if isinstance(distribution_spec, dict):
        for key in ('probs', 'probabilities', 'p', 'weights', 'dist'):
            if key in distribution_spec:
                arr = np.asarray(distribution_spec[key], dtype=float).ravel()
                if arr.size == n_types:
                    s = arr.sum()
                    if s > 0:
                        return arr / s
                    return arr
        # maybe nested per-goods
        v1 = distribution_spec.get('v1', distribution_spec.get('goods0', None))
        v2 = distribution_spec.get('v2', distribution_spec.get('goods1', None))
        if v1 is not None and v2 is not None:
            p1 = np.asarray(v1, dtype=float).ravel()
            p2 = np.asarray(v2, dtype=float).ravel()
            if p1.size == 4 and p2.size == 4:
                grid = np.outer(p1, p2)
                return grid.ravel()
        # fallback: any array of length 16
        for v in distribution_spec.values():
            arr = np.asarray(v, dtype=float).ravel()
            if arr.size == n_types:
                s = arr.sum()
                if s > 0:
                    return arr / s
    arr = np.asarray(distribution_spec, dtype=float).ravel()
    if arr.size == n_types:
        s = arr.sum()
        if s > 0:
            return arr / s
    return np.full(n_types, 1.0 / n_types)


def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s

    # Build type list: all (v1,v2) in {0,1,2,3}^2, in a fixed canonical order.
    vals = [0, 1, 2, 3]
    types = [(a, b) for a in vals for b in vals]
    n = len(types)  # 16
    assert n == 16

    # If support is provided and matches, use it to align ordering; otherwise canonical.
    if support is not None:
        sup = list(support)
        if len(sup) == n:
            # map to canonical value grid
            types = [tuple(int(x) for x in t) for t in sup]

    probs = _get_probs(distribution_spec, n)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if s <= 0:
        probs = np.full(n, 1.0 / n)
    else:
        probs = probs / s

    V = np.array([[v1, v2] for (v1, v2) in types], dtype=float)  # (16,2)

    # Utility-based LP formulation:
    # Variables: a_{t,1}, a_{t,2} for t=0..15 (32 vars), then u_t for t=0..15 (16 vars)
    # Total: 48 variables
    #
    # Objective: maximize expected revenue = sum_t pi_t * (v_t . a_t - u_t)
    #   => minimize -sum_t pi_t * (v_t . a_t - u_t) = sum_t pi_t * (u_t - v_t . a_t)
    #
    # IC: for all t, t': u_t >= v_t . a_t' - v_t . a_t + u_t'
    #   => v_t . a_t' - v_t . a_t - u_t + u_t' <= 0
    #
    # IR: u_t >= 0  =>  -u_t <= 0
    #
    # Bounds: a in [0,1], u in [0, inf)

    nA = 2 * n  # 32 allocation vars
    nU = n      # 16 utility vars
    nvar = nA + nU  # 48

    def a1_idx(t): return t
    def a2_idx(t): return n + t
    def u_idx(t): return nA + t

    # Objective: minimize sum_t pi_t * (u_t - v_t . a_t)
    c = np.zeros(nvar)
    for t in range(n):
        c[a1_idx(t)] = -probs[t] * V[t, 0]
        c[a2_idx(t)] = -probs[t] * V[t, 1]
        c[u_idx(t)] = probs[t]

    # Inequality constraints A_ub x <= b_ub
    A_rows = []
    b_rows = []

    # IC: for all t (true), t' (report):
    # v(t).a(t') - v(t).a(t) - u_t + u_t' <= 0
    for t in range(n):
        for tp in range(n):
            if t == tp:
                continue
            row = np.zeros(nvar)
            # v(t).a(t')
            row[a1_idx(tp)] += V[t, 0]
            row[a2_idx(tp)] += V[t, 1]
            # - v(t).a(t)
            row[a1_idx(t)] -= V[t, 0]
            row[a2_idx(t)] -= V[t, 1]
            # - u_t
            row[u_idx(t)] -= 1.0
            # + u_t'
            row[u_idx(tp)] += 1.0
            A_rows.append(row)
            b_rows.append(0.0)

    # IR: -u_t <= 0  =>  u_t >= 0
    for t in range(n):
        row = np.zeros(nvar)
        row[u_idx(t)] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)

    A_ub = np.array(A_rows, dtype=float)
    b_ub = np.array(b_rows, dtype=float)

    # Bounds: a in [0,1], u in [0, inf)
    bounds = []
    for i in range(nA):
        bounds.append((0.0, 1.0))
    for i in range(nU):
        bounds.append((0.0, None))

    # Solve with HiGHS using tighter tolerances
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                  options={'presolve': True, 'mip_rel_gap': 1e-12, 'time_limit': DEADLINE - (time.time() - t0) - 5.0})

    if not res.success:
        # Fallback: safe mechanism
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}

    x = res.x
    allocations = np.zeros((n, 2))
    utilities = np.zeros(n)
    for t in range(n):
        allocations[t, 0] = np.clip(x[a1_idx(t)], 0.0, 1.0)
        allocations[t, 1] = np.clip(x[a2_idx(t)], 0.0, 1.0)
        utilities[t] = np.clip(x[u_idx(t)], 0.0, None)

    # Recover payments from utilities: p_t = v_t . a_t - u_t
    payments = np.zeros(n)
    for t in range(n):
        payments[t] = V[t] @ allocations[t] - utilities[t]

    # Ensure payments are non-negative (clip if tiny negative due to numerical error)
    payments = np.maximum(payments, 0.0)

    # Polish: verify constraints; if violated, do a high-precision re-solve
    tol = 1e-8
    ok = True
    for t in range(n):
        util_t = V[t] @ allocations[t] - payments[t]
        if util_t < -tol:
            ok = False
            break
        for tp in range(n):
            util_tp_as_t = V[t] @ allocations[tp] - payments[tp]
            if util_t < util_tp_as_t - tol:
                ok = False
                break
        if not ok:
            break

    if not ok:
        # High-precision polish: re-solve with tighter tolerances
        res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                       options={'presolve': False, 'primal_feasibility_tolerance': 1e-12,
                                'dual_feasibility_tolerance': 1e-12,
                                'time_limit': DEADLINE - (time.time() - t0) - 2.0})
        if res2.success:
            x2 = res2.x
            for t in range(n):
                allocations[t, 0] = np.clip(x2[a1_idx(t)], 0.0, 1.0)
                allocations[t, 1] = np.clip(x2[a2_idx(t)], 0.0, 1.0)
                utilities[t] = np.clip(x2[u_idx(t)], 0.0, None)
            for t in range(n):
                payments[t] = max(0.0, V[t] @ allocations[t] - utilities[t])
        else:
            # Final fallback: constant allocation, zero payments
            allocations = np.ones((n, 2))
            payments = np.zeros(n)
            return {'allocations': allocations, 'payments': payments}

    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
