# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
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

    # Precompute edge incidence for each demand for both directions
    # clockwise_mask[d] : boolean array of length n, True if edge e is on clockwise path
    # ccw_mask[d] : boolean array of length n, True if edge e is on counter-clockwise path
    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw = np.zeros(n, dtype=bool)
        ccw = np.zeros(n, dtype=bool)
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
        if s <= t:
            for e in range(s, t):
                cw[e] = True
        else:
            for e in range(s, n):
                cw[e] = True
            for e in range(0, t):
                cw[e] = True
        # Counter-clockwise: all other edges
        ccw = ~cw
        cw_masks.append(cw)
        ccw_masks.append(ccw)

    cw_masks = np.array(cw_masks, dtype=float)  # shape (D, n)
    ccw_masks = np.array(ccw_masks, dtype=float)  # shape (D, n)
    weights = np.array([w for s, t, w in demands], dtype=float)  # shape (D,)

    # A[e, d] = w[d] * (ccw[d, e] - cw[d, e])
    # b[e] = sum_d w[d] * cw[d, e]
    # Load[e] = b[e] + sum_d A[e, d] * x[d]
    A = (ccw_masks.T - cw_masks.T) * weights  # shape (n, D)
    b = weights @ cw_masks  # shape (n,)

    def make_objective(alpha):
        def objective(x):
            loads = b + A @ x
            m = np.max(loads)
            if m == 0:
                return 0.0
            lse = m + np.log(np.sum(np.exp(alpha * (loads - m)))) / alpha
            return lse
        def gradient(x):
            loads = b + A @ x
            m = np.max(loads)
            if m == 0:
                return np.zeros(D)
            exp_terms = np.exp(alpha * (loads - m))
            softmax = exp_terms / np.sum(exp_terms)
            grad = A.T @ softmax
            return grad
        return objective, gradient

    # Try multiple alpha values and initial guesses
    best_x = None
    best_val = np.inf

    np.random.seed(42)
    alphas = [20.0, 30.0, 50.0, 80.0]
    initial_guesses = [
        np.zeros(D),
        np.ones(D),
        np.random.rand(D),
        np.random.rand(D),
        np.random.rand(D),
        np.random.rand(D),
        np.random.rand(D),
    ]

    for alpha in alphas:
        obj, grad = make_objective(alpha)
        for x0 in initial_guesses:
            try:
                res = minimize(
                    obj,
                    x0,
                    jac=grad,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * D,
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                # Evaluate actual max load for comparison
                loads = b + A @ res.x
                actual_max = np.max(loads)
                if actual_max < best_val:
                    best_val = actual_max
                    best_x = res.x.copy()
            except Exception:
                continue

    if best_x is None:
        return np.zeros(D)

    # Round to binary
    routing = (best_x >= 0.5).astype(float)

    # Local search to refine the binary solution
    def max_load(routing_vec):
        loads = b + A @ routing_vec
        return np.max(loads)

    current_load = max_load(routing)
    improved = True
    while improved:
        improved = False
        # 1-flip moves
        for d in range(D):
            new_routing = routing.copy()
            new_routing[d] = 1.0 - new_routing[d]
            new_load = max_load(new_routing)
            if new_load < current_load - 1e-10:
                routing = new_routing
                current_load = new_load
                improved = True
        # 2-flip moves
        if not improved:
            for d1 in range(D):
                if improved:
                    break
                for d2 in range(d1 + 1, D):
                    new_routing = routing.copy()
                    new_routing[d1] = 1.0 - new_routing[d1]
                    new_routing[d2] = 1.0 - new_routing[d2]
                    new_load = max_load(new_routing)
                    if new_load < current_load - 1e-10:
                        routing = new_routing
                        current_load = new_load
                        improved = True
                        break
        # 3-flip moves
        if not improved:
            for d1 in range(D):
                if improved:
                    break
                for d2 in range(d1 + 1, D):
                    if improved:
                        break
                    for d3 in range(d2 + 1, D):
                        new_routing = routing.copy()
                        new_routing[d1] = 1.0 - new_routing[d1]
                        new_routing[d2] = 1.0 - new_routing[d2]
                        new_routing[d3] = 1.0 - new_routing[d3]
                        new_load = max_load(new_routing)
                        if new_load < current_load - 1e-10:
                            routing = new_routing
                            current_load = new_load
                            improved = True
                            break

    return routing


# EVOLVE-BLOCK-END
