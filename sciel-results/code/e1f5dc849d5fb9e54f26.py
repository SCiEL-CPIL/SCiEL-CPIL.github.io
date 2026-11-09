# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        value < 0.5 -> route the demand clockwise
        value >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill

    # Problem definition
    n = 10  # number of nodes
    D = 15  # number of demands

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
        (5, 2, 2)
    ]

    # Precompute clockwise and counter-clockwise edge usage for each demand
    # For demand (s, t, w):
    # Clockwise uses edges s, s+1, ..., t-1 (mod n)
    # Counter-clockwise uses all other edges

    def get_clockwise_edges(s, t, n):
        """Get list of edges on clockwise path from s to t."""
        edges = []
        if s == t:
            return edges
        e = s
        while e != t:
            edges.append(e)
            e = (e + 1) % n
        return edges

    # A[d, e] = 1 if demand d uses edge e when routed clockwise
    A_cw = np.zeros((D, n), dtype=float)
    # A_ccw[d, e] = 1 if demand d uses edge e when routed counter-clockwise
    A_ccw = np.zeros((D, n), dtype=float)

    for d, (s, t, w) in enumerate(demands):
        cw_edges = get_clockwise_edges(s, t, n)
        for e in range(n):
            if e in cw_edges:
                A_cw[d, e] = 1.0
            else:
                A_ccw[d, e] = 1.0

    # Weights vector
    w = np.array([demands[d][2] for d in range(D)], dtype=float)

    # LP formulation:
    # Variables: x[0..D-1] in [0,1] (routing direction), t (max load)
    # Minimize: t
    # Subject to: for each edge e:
    #   sum_d (w[d] * A_cw[d,e] * (1-x[d]) + w[d] * A_ccw[d,e] * x[d]) <= t
    #   which is: sum_d w[d] * A_cw[d,e] - sum_d w[d] * (A_cw[d,e] - A_ccw[d,e]) * x[d] <= t
    #   Let B[d,e] = A_cw[d,e] - A_ccw[d,e]
    #   Then: sum_d w[d]*A_cw[d,e] - sum_d w[d]*B[d,e]*x[d] - t <= 0

    B = A_cw - A_ccw  # B[d,e] = 1 if cw uses e, -1 if ccw uses e, 0 if neither (not possible here)

    # For each edge e:
    # base_load[e] = sum_d w[d] * A_cw[d,e]
    # constraint: base_load[e] - sum_d w[d]*B[d,e]*x[d] - t <= 0
    # i.e.: sum_d w[d]*B[d,e]*x[d] + t >= base_load[e]
    # Rearranged for linprog (which handles <= constraints):
    # -sum_d w[d]*B[d,e]*x[d] - t <= -base_load[e]

    base_load = np.array([np.sum(w * A_cw[:, e]) for e in range(n)], dtype=float)

    # Variables: [x_0, ..., x_14, t]
    # Objective: minimize t -> c = [0, ..., 0, 1]
    c = np.zeros(D + 1)
    c[D] = 1.0  # coefficient of t

    # Constraints: for each edge e:
    # -sum_d w[d]*B[d,e]*x[d] - t <= -base_load[e]
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)

    for e in range(n):
        for d in range(D):
            A_ub[e, d] = -w[d] * B[d, e]
        A_ub[e, D] = -1.0
        b_ub[e] = -base_load[e]

    # Bounds: x[d] in [0,1], t in [0, inf)
    bounds = [(0, 1)] * D + [(0, None)]

    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        x_lp = result.x[:D]
    else:
        # Fallback: all clockwise
        x_lp = np.zeros(D)

    # Round to binary: try all 2^15 combinations near the LP solution
    # First, round directly
    x_binary = np.where(x_lp >= 0.5, 1.0, 0.0)

    def compute_max_load(x_bin):
        """Compute maximum edge load given binary routing vector."""
        loads = np.zeros(n)
        for d in range(D):
            if x_bin[d] < 0.5:
                # clockwise
                loads += w[d] * A_cw[d, :]
            else:
                # counter-clockwise
                loads += w[d] * A_ccw[d, :]
        return np.max(loads)

    best_load = compute_max_load(x_binary)
    best_x = x_binary.copy()

    # Local search: flip one bit at a time
    improved = True
    while improved:
        improved = False
        for d in range(D):
            if time.time() > deadline - 30:
                break
            x_trial = best_x.copy()
            x_trial[d] = 1.0 - x_trial[d]
            trial_load = compute_max_load(x_trial)
            if trial_load < best_load - 1e-9:
                best_load = trial_load
                best_x = x_trial
                improved = True
        if time.time() > deadline - 30:
            break

    # Try random restarts if time permits
    if time.time() < deadline - 60:
        rng = np.random.default_rng(42)
        for _ in range(1000):
            if time.time() > deadline - 30:
                break
            x_rand = rng.random(D)
            x_rand_bin = np.where(x_rand >= 0.5, 1.0, 0.0)
            rand_load = compute_max_load(x_rand_bin)
            if rand_load < best_load - 1e-9:
                best_load = rand_load
                best_x = x_rand_bin.copy()
                # Local search from this point
                improved2 = True
                while improved2 and time.time() < deadline - 30:
                    improved2 = False
                    for d in range(D):
                        if time.time() > deadline - 30:
                            break
                        x_trial = best_x.copy()
                        x_trial[d] = 1.0 - x_trial[d]
                        trial_load = compute_max_load(x_trial)
                        if trial_load < best_load - 1e-9:
                            best_load = trial_load
                            best_x = x_trial
                            improved2 = True

    return best_x

# EVOLVE-BLOCK-END
