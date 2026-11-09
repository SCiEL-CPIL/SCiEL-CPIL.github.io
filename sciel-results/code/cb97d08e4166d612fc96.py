# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

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
    start_time = time.time()
    deadline = 630.0 - 30.0  # safety margin

    n = 10
    D = 15

    # Demands: (source, target, weight)
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

    # Precompute edge sets for each demand in both directions
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
    # Counter-clockwise: all other edges

    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = []
        e = s
        while e != t:
            cw.append(e)
            e = (e + 1) % n
        cw_edges.append(cw)
        # CCW edges are all edges not in CW
        all_edges = list(range(n))
        ccw = [e for e in all_edges if e not in cw]
        ccw_edges.append(ccw)

    # Build incidence matrix: A[i][e] = 1 if demand i uses edge e in CW direction
    # If we choose CW, add weight to CW edges; if CCW, add weight to CCW edges.
    # Let x_i = 0 for CW, x_i = 1 for CCW.
    # Load on edge e = sum_i w_i * (A_cw[i][e] * (1 - x_i) + A_ccw[i][e] * x_i)
    #               = sum_i w_i * A_cw[i][e] + sum_i w_i * (A_ccw[i][e] - A_cw[i][e]) * x_i

    A_cw = np.zeros((D, n))
    A_ccw = np.zeros((D, n))
    for i in range(D):
        for e in cw_edges[i]:
            A_cw[i, e] = 1.0
        for e in ccw_edges[i]:
            A_ccw[i, e] = 1.0

    weights = np.array([d[2] for d in demands], dtype=float)

    # Base load if all CW
    base_load = (A_cw * weights[:, None]).sum(axis=0)
    # Delta load per unit x_i
    delta = (A_ccw - A_cw) * weights[:, None]  # shape (D, n)

    # Try ILP with scipy
    best_routing = np.zeros(D)
    best_max_load = np.inf

    def compute_max_load(routing):
        """Compute max edge load given routing (0=CW, 1=CCW)."""
        loads = base_load + (delta * routing[:, None]).sum(axis=0)
        return loads.max()

    # Initial: all CW
    init_load = compute_max_load(best_routing)
    best_max_load = init_load

    # Try to solve ILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        # Minimize t subject to:
        #   base_load_e + sum_i delta[i,e] * x_i <= t  for all e
        #   x_i in {0, 1}
        #
        # Variables: [x_0, ..., x_14, t] -> 16 variables
        # Objective: minimize t -> c = [0,...,0, 1]

        c = np.zeros(D + 1)
        c[D] = 1.0  # minimize t

        # Constraints: base_load_e + sum_i delta[i,e] * x_i - t <= 0 for each edge e
        # i.e., sum_i delta[i,e] * x_i - t <= -base_load_e

        A_ub = np.zeros((n, D + 1))
        b_ub = np.zeros(n)
        for e in range(n):
            A_ub[e, :D] = delta[:, e]
            A_ub[e, D] = -1.0
            b_ub[e] = -base_load[e]

        # x_i in {0,1}, t >= 0
        integrality = np.concatenate([np.ones(D), np.zeros(1)])  # x_i integer, t continuous
        bounds = Bounds(
            lb=np.concatenate([np.zeros(D), np.array([0.0])]),
            ub=np.concatenate([np.ones(D), np.array([np.inf])]),
        )

        constraints = [
            LinearConstraint(A_ub, -np.inf, b_ub),
        ]

        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
        )

        if result.success and result.x is not None:
            x_opt = result.x[:D].round().astype(int)
            # Convert to routing: 0=CW, 1=CCW
            routing_ilp = x_opt.astype(float)
            load_ilp = compute_max_load(routing_ilp)
            if load_ilp < best_max_load:
                best_max_load = load_ilp
                best_routing = routing_ilp

    except Exception as e:
        pass  # Fall back to heuristic

    # Heuristic: local search if ILP didn't give good result or as backup
    if time.time() < deadline:
        # Greedy + local search
        def local_search(routing, max_iters=5000):
            routing = routing.copy()
            current_max = compute_max_load(routing)
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                improved = False
                # Try flipping each demand
                for i in range(D):
                    old = routing[i]
                    routing[i] = 1.0 - old
                    new_max = compute_max_load(routing)
                    if new_max < current_max - 1e-12:
                        current_max = new_max
                        improved = True
                    else:
                        routing[i] = old
                if not improved:
                    break
            return routing, current_max

        # Start from best found so far
        routing_ls, load_ls = local_search(best_routing)
        if load_ls < best_max_load:
            best_max_load = load_ls
            best_routing = routing_ls

        # Try multiple random restarts
        np.random.seed(42)
        for restart in range(20):
            if time.time() > deadline:
                break
            routing_rand = np.random.randint(0, 2, D).astype(float)
            routing_ls, load_ls = local_search(routing_rand)
            if load_ls < best_max_load:
                best_max_load = load_ls
                best_routing = routing_ls

    return best_routing


# EVOLVE-BLOCK-END
