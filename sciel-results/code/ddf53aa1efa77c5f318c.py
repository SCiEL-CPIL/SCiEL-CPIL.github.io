# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


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
    n_nodes = 10
    n_edges = 10
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

    # Precompute edge masks for each demand
    # mask[d][e] = 1 if edge e is on the clockwise path from s to t
    # Edge e connects node e to (e+1) % 10
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod 10)
    # Counter-clockwise uses all other edges

    cw_masks = np.zeros((D, n_edges), dtype=np.int8)
    ccw_masks = np.zeros((D, n_edges), dtype=np.int8)

    for d_idx, (s, t, w) in enumerate(demands):
        if s <= t:
            # Clockwise: edges s, s+1, ..., t-1
            for e in range(s, t):
                cw_masks[d_idx, e] = 1
            for e in range(t, n_nodes):
                ccw_masks[d_idx, e] = 1
            for e in range(0, s):
                ccw_masks[d_idx, e] = 1
        else:
            # s > t: clockwise wraps around
            # Clockwise: edges s, s+1, ..., 9, 0, ..., t-1
            for e in range(s, n_nodes):
                cw_masks[d_idx, e] = 1
            for e in range(0, t):
                cw_masks[d_idx, e] = 1
            for e in range(t, s):
                ccw_masks[d_idx, e] = 1

    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    def compute_max_load(routing):
        """Compute the maximum edge load given a routing vector."""
        # routing: array of 0/1, 0=cw, 1=ccw
        # load[e] = sum over demands d of weight[d] * mask[d, e]
        # mask = cw_mask if routing[d]==0 else ccw_mask
        mask = np.where(routing[:, None] == 0, cw_masks, ccw_masks)
        loads = mask.T @ weights
        return np.max(loads)

    def evaluate(routing):
        """Return (max_load, routing) for a routing vector."""
        ml = compute_max_load(routing)
        return ml, routing

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, leave margin

    # Start with all clockwise
    best_routing = np.zeros(D, dtype=np.float64)
    best_score = compute_max_load(best_routing)

    # Local search: flip one demand at a time
    def local_search(routing, max_iterations=10000):
        nonlocal best_routing, best_score
        routing = routing.copy()
        current_score = compute_max_load(routing)
        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline - 10:
                break

            # Try flipping each demand
            best_flip = -1
            best_new_score = current_score

            for d in range(D):
                new_routing = routing.copy()
                new_routing[d] = 1.0 - new_routing[d]
                new_score = compute_max_load(new_routing)
                if new_score < best_new_score - 1e-12:
                    best_new_score = new_score
                    best_flip = d

            if best_flip >= 0:
                routing[best_flip] = 1.0 - routing[best_flip]
                current_score = best_new_score
                improved = True
                if current_score < best_score - 1e-12:
                    best_score = current_score
                    best_routing = routing.copy()

        return routing, current_score

    # Run local search from initial all-cw
    local_search(np.zeros(D, dtype=np.float64))

    # Also try all-ccw
    local_search(np.ones(D, dtype=np.float64))

    # Simulated annealing / random restarts
    def simulated_annealing(routing_init, max_iter=50000, T_start=50.0, T_end=1.0):
        nonlocal best_routing, best_score
        routing = routing_init.copy()
        current_score = compute_max_load(routing)
        improved = True
        iterations = 0

        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            if time.time() > deadline - 10:
                break

            T = T_start * (T_end / T_start) ** (iterations / max_iter)

            # Pick a random demand to flip
            d = random.randint(0, D - 1)
            new_routing = routing.copy()
            new_routing[d] = 1.0 - new_routing[d]
            new_score = compute_max_load(new_routing)

            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                routing = new_routing
                current_score = new_score
                if current_score < best_score - 1e-12:
                    best_score = current_score
                    best_routing = routing.copy()
                improved = True

        return routing, current_score

    # Multiple random restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 30:
            break
        init = np.random.randint(0, 2, D).astype(np.float64)
        sa_routing, sa_score = simulated_annealing(init, max_iter=20000, T_start=20.0, T_end=0.5)
        # Polish with local search
        local_search(sa_routing, max_iterations=5000)

    # Final polish with more aggressive local search
    local_search(best_routing, max_iterations=50000)

    # Ensure output is valid: entries should be 0 or 1 (or >= 0.5 for ccw)
    # The task says: entry >= 0.5 routes counter-clockwise, any other value clockwise
    # So we return 0.0 for clockwise and 1.0 for counter-clockwise
    result = np.where(best_routing >= 0.5, 1.0, 0.0).astype(np.float64)

    return result


# EVOLVE-BLOCK-END
