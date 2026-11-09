# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem on a ring of n=10 nodes with 15 demands.
    Returns an array of shape (15,) where entry >= 0.5 routes CCW, else CW.
    Minimizes the maximum edge load.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 10.0  # seconds

    n = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]
    D = len(demands)

    # Precompute edges used by each demand in CW and CCW direction
    def cw_edges(s, t):
        if s == t:
            return list(range(n))
        edges = []
        e = s
        while e != t:
            edges.append(e)
            e = (e + 1) % n
        return edges

    def ccw_edges(s, t):
        if s == t:
            return list(range(n))
        edges = []
        e = t
        while e != s:
            edges.append(e)
            e = (e + 1) % n
        return edges

    cw_edge_sets = []
    ccw_edge_sets = []
    for s, t, w in demands:
        cw_edge_sets.append(set(cw_edges(s, t)))
        ccw_edge_sets.append(set(ccw_edges(s, t)))

    # Build coefficient matrix A (n x D) and base vector b (n,)
    # load_e = b_e + sum_d A[e,d] * x_d, where x_d = 1 if CW, 0 if CCW
    # A[e,d] = +w_d if d uses edge e in CW, -w_d if d uses edge e in CCW, 0 otherwise
    # b_e = sum of w_d for demands d where e is in ccw_edges(d)
    A = np.zeros((n, D))
    b = np.zeros(n)
    for e in range(n):
        for d in range(D):
            s, t, w = demands[d]
            if e in cw_edge_sets[d]:
                A[e, d] = w
            elif e in ccw_edge_sets[d]:
                b[e] += w
                A[e, d] = -w

    def max_load_from_binary(x):
        """Compute max edge load from binary x (1=CW, 0=CCW)."""
        loads = b + A @ x
        return np.max(loads)

    def routing_from_binary(x):
        """Convert binary x to routing array (0.0=CW, 1.0=CCW)."""
        return np.where(x == 1, 0.0, 1.0)

    # Exhaustive search over all 2^15 = 32768 routings
    best_max_load = float('inf')
    best_x = np.zeros(D, dtype=int)

    total_routings = 1 << D
    for mask in range(total_routings):
        if time.time() > deadline - safety_margin:
            break
        x = np.array([(mask >> d) & 1 for d in range(D)], dtype=float)
        loads = b + A @ x
        ml = np.max(loads)
        if ml < best_max_load:
            best_max_load = ml
            best_x = x.copy()

    # Smooth optimization approach: minimize log-sum-exp of edge loads
    # with continuous variables, then round and do local search
    alpha_values = [10, 20, 50, 100]
    for alpha in alpha_values:
        if time.time() > deadline - safety_margin:
            break

        def objective(x_cont):
            """Smooth approximation of max load via log-sum-exp."""
            loads = b + A @ x_cont
            # Clip for numerical stability
            loads_clipped = np.clip(loads, -100, 100)
            max_l = np.max(loads_clipped)
            if max_l > 100:
                return max_l
            sum_exp = np.sum(np.exp(alpha * (loads_clipped - max_l)))
            return max_l + np.log(sum_exp) / alpha

        def gradient(x_cont):
            """Gradient of the log-sum-exp objective."""
            loads = b + A @ x_cont
            loads_clipped = np.clip(loads, -100, 100)
            max_l = np.max(loads_clipped)
            if max_l > 100:
                # Degenerate case
                loads_bool = loads_clipped == max_l
                return A.T @ loads_bool
            exp_terms = np.exp(alpha * (loads_clipped - max_l))
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            return A.T @ weights

        # Try multiple initial points
        initial_points = [
            np.zeros(D),
            np.ones(D),
            np.random.seed(42)
        ]
        np.random.seed(42)
        random_inits = [np.random.rand(D) for _ in range(5)]
        initial_points = [np.zeros(D), np.ones(D)] + random_inits

        for x0 in initial_points:
            if time.time() > deadline - safety_margin:
                break
            try:
                result = minimize(
                    objective,
                    x0,
                    jac=gradient,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * D,
                    options={'maxiter': 1000, 'ftol': 1e-12}
                )
                x_cont = result.x
                # Round to binary
                x_rounded = (x_cont >= 0.5).astype(float)
                ml = max_load_from_binary(x_rounded)
                if ml < best_max_load:
                    best_max_load = ml
                    best_x = x_rounded.copy()
            except Exception:
                pass

    # Local search: 1-flip and 2-flip
    if time.time() > deadline - safety_margin:
        pass
    else:
        improved = True
        while improved:
            if time.time() > deadline - safety_margin:
                break
            improved = False
            # 1-flip
            for d in range(D):
                if time.time() > deadline - safety_margin:
                    break
                x_new = best_x.copy()
                x_new[d] = 1 - x_new[d]
                ml = max_load_from_binary(x_new)
                if ml < best_max_load:
                    best_max_load = ml
                    best_x = x_new
                    improved = True
            # 2-flip
            for d1 in range(D):
                if time.time() > deadline - safety_margin:
                    break
                for d2 in range(d1 + 1, D):
                    if time.time() > deadline - safety_margin:
                        break
                    x_new = best_x.copy()
                    x_new[d1] = 1 - x_new[d1]
                    x_new[d2] = 1 - x_new[d2]
                    ml = max_load_from_binary(x_new)
                    if ml < best_max_load:
                        best_max_load = ml
                        best_x = x_new
                        improved = True

    # Convert best_x to routing array
    best_routing = routing_from_binary(best_x)

    return best_routing

# EVOLVE-BLOCK-END
