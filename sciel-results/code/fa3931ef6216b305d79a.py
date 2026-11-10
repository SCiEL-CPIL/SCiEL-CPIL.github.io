# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 600.0  # seconds; keep safety margin

def _build_data():
    n = 10
    demands = [
        (0,5,3),(1,7,2),(2,9,4),(3,6,1),(0,4,5),(2,8,2),
        (5,9,3),(1,4,2),(6,0,4),(3,8,1),(7,2,3),(4,9,2),
        (8,1,5),(0,6,1),(5,2,2),
    ]
    D = len(demands)
    # A[d, e] = 1 if clockwise route of demand d uses edge e
    A = np.zeros((D, n))
    cw_len = np.zeros(D)
    for d,(s,t,w) in enumerate(demands):
        if s == t:
            cw_len[d] = 0
        else:
            L = (t - s) % n
            cw_len[d] = L
            for e in range(s, t) if t > s else list(range(s,n)) + list(range(0,t)):
                A[d, e] = 1.0
    return n, demands, D, A, cw_len

def _loads(routing, n, A, cw_len, weights):
    # routing: 0=cw, 1=ccw
    x = (1.0 - routing) * A + routing * (1.0 - A)  # D x n
    return x.T @ weights  # n

def _max_load(routing, n, A, cw_len, weights):
    return float(np.max(_loads(routing, n, A, cw_len, weights)))

def _solve_lp_simplex(n, D, A, cw_len, weights):
    # Minimize t s.t.
    #   sum_d w_d * y_d * A[d,e] + sum_d w_d * (1-y_d) * (1-A[d,e]) <= t   for all e
    #   0 <= y_d <= 1
    # where y_d=1 means ccw.
    # Rewrite: sum_d w_d * (A[d,e] - (1-A[d,e])) * y_d + const_e <= t
    # const_e = sum_d w_d * (1-A[d,e])
    # coeff_e_d = w_d * (2*A[d,e] - 1)
    #
    # Variables: y_1..y_D, t  (D+1 vars)
    # Constraints: for each edge e: sum_d coeff_e_d * y_d - t <= const_e
    #              plus bounds handled by standard form with slacks.
    #
    # Standard form for simplex (min): minimize c^T x s.t. M x <= b, x >= 0
    # c = [0]*D + [1]
    # M has n rows (edge constraints): [coeff_e, -1] <= const_e
    # Plus we need y_d <= 1: y_d <= 1  ->  [e_d] <= 1
    #
    # Total constraints: n + D
    # Total vars: D + 1
    c = np.zeros(D + 1)
    c[-1] = 1.0
    M = np.zeros((n + D, D + 1))
    b = np.zeros(n + D)
    for e in range(n):
        coeff = weights * (2.0 * A[:, e] - 1.0)  # D
        M[e, :D] = coeff
        M[e, D] = -1.0
        b[e] = float(np.sum(weights * (1.0 - A[:, e])))
    for d in range(D):
        M[n + d, d] = 1.0
        b[n + d] = 1.0

    # Solve via scipy if available, else fallback
    try:
        from scipy.optimize import linprog
        res = linprog(c, A_ub=M, b_ub=b, method='highs', bounds=[(0,1)]*D + [(0,None)])
        if res.success:
            y = res.x[:D]
            t_val = res.x[D]
            return y, t_val
    except Exception:
        pass
    # Fallback: all cw
    return np.zeros(D), None

def _round_lp(y, n, D, A, cw_len, weights):
    """Round LP solution and local-search polish."""
    routing = np.where(y >= 0.5, 1.0, 0.0)
    best_ml = _max_load(routing, n, A, cw_len, weights)
    best_routing = routing.copy()

    # Local search: try flipping each demand, keep if improves
    improved = True
    while improved:
        improved = False
        for d in range(D):
            trial = best_routing.copy()
            trial[d] = 1.0 - trial[d]
            ml = _max_load(trial, n, A, cw_len, weights)
            if ml < best_ml - 1e-9:
                best_ml = ml
                best_routing = trial
                improved = True
    return best_routing, best_ml

def _pairwise_local_search(routing, n, D, A, cw_len, weights, max_iter=200):
    """Flip pairs of demands if it reduces max load."""
    best_routing = routing.copy()
    best_ml = _max_load(best_routing, n, A, cw_len, weights)
    for it in range(max_iter):
        improved = False
        for d1 in range(D):
            for d2 in range(d1+1, D):
                trial = best_routing.copy()
                trial[d1] = 1.0 - trial[d1]
                trial[d2] = 1.0 - trial[d2]
                ml = _max_load(trial, n, A, cw_len, weights)
                if ml < best_ml - 1e-9:
                    best_ml = ml
                    best_routing = trial
                    improved = True
        if not improved:
            break
    return best_routing, best_ml

def ring_routing() -> np.ndarray:
    t0 = time.time()
    n, demands, D, A, cw_len = _build_data()
    weights = np.array([w for (_,_,w) in demands], dtype=float)

    # Solve LP
    y, t_lp = _solve_lp_simplex(n, D, A, cw_len, weights)

    # Round and polish
    routing, ml = _round_lp(y, n, D, A, cw_len, weights)

    # Pairwise local search
    routing, ml = _pairwise_local_search(routing, n, D, A, cw_len, weights, max_iter=50)

    # If we have time, try more restarts from perturbed LP solutions
    elapsed = time.time() - t0
    if elapsed < _DEADLINE - 10:
        rng = np.random.default_rng(42)
        for restart in range(20):
            if time.time() - t0 > _DEADLINE - 10:
                break
            # Perturb LP solution
            y_pert = y + rng.normal(0, 0.3, D)
            y_pert = np.clip(y_pert, 0, 1)
            r2, ml2 = _round_lp(y_pert, n, D, A, cw_len, weights)
            r2, ml2 = _pairwise_local_search(r2, n, D, A, cw_len, weights, max_iter=30)
            if ml2 < ml:
                routing = r2
                ml = ml2

    # Final: ensure valid output
    routing = np.where(routing >= 0.5, 1.0, 0.0)
    return routing

# EVOLVE-BLOCK-END
