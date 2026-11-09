# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
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
    weights = np.array([d[2] for d in demands], dtype=np.float64)

    # Precompute clockwise edge-usage matrix
    cw = np.zeros((D, n), dtype=np.float64)
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            for e in range(s, t):
                cw[i, e] = 1.0
        else:
            for e in range(s, n):
                cw[i, e] = 1.0
            for e in range(0, t):
                cw[i, e] = 1.0

    ccw = 1.0 - cw
    delta = ccw - cw
    w_delta = delta * weights[:, np.newaxis]  # (D, n)

    # ILP formulation:
    # Variables: x[0..D-1] binary (1 = CCW, 0 = CW), t continuous >= 0
    # Minimize t
    # Subject to: for each edge e: sum_d w_d * (cw[d,e] + x_d * delta[d,e]) <= t
    #            which is: base_load[e] + sum_d w_delta[d,e] * x_d <= t
    #            => sum_d w_delta[d,e] * x_d - t <= -base_load[e]
    #            i.e., for each e: [w_delta[:,e], -1] @ [x, t] <= -base_load[e]
    
    base_load = cw.T @ weights  # (n,)

    # Objective: minimize t (variable index D)
    c = np.zeros(D + 1)
    c[D] = 1.0

    # Constraints: A @ vars <= b
    # For each edge e: sum_d w_delta[d,e] * x_d - t <= -base_load[e]
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    for e in range(n):
        A_ub[e, :D] = w_delta[:, e]
        A_ub[e, D] = -1.0
        b_ub[e] = -base_load[e]

    # Bounds: x_d in [0, 1] for d in 0..D-1, t in [0, inf)
    lb = np.zeros(D + 1)
    ub = np.full(D + 1, np.inf)
    ub[:D] = 1.0
    bounds = Bounds(lb, ub)

    # Integrality: x_0..x_{D-1} are binary (1), t is continuous (0)
    integrality = np.zeros(D + 1, dtype=np.int8)
    integrality[:D] = 1

    # Use linear constraint form
    A_eq = None
    b_eq = None

    constraint = LinearConstraint(A_ub, -np.inf, b_ub)

    start_time = time.time()
    result = milp(
        c=c,
        constraints=[constraint],
        integrality=integrality,
        bounds=bounds,
    )

    elapsed = time.time() - start_time

    if result.success and result.x is not None:
        x = result.x[:D]
        # Round to get binary decisions (should already be 0 or 1)
        routing = np.round(x).astype(np.float64)
        return routing
    else:
        # Fallback: all clockwise
        return np.zeros(D)

# EVOLVE-BLOCK-END
