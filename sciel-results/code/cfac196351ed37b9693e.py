# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
        0 -> clockwise, 1 (or >= 0.5) -> counter-clockwise
    """
    t_start = time.time()
    DEADLINE = 620.0  # safety margin before 630s kill

    # Problem constants
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute edge masks for CW and CCW routing for each demand
    # CW from s to t: edges s, s+1, ..., t-1 (mod 10)
    # CCW: all other edges
    cw_edges = np.zeros((D, n), dtype=np.int8)
    ccw_edges = np.zeros((D, n), dtype=np.int8)
    for i, (s, t, w) in enumerate(demands):
        # CW path: s -> s+1 -> ... -> t
        path = []
        e = s
        while e != t:
            path.append(e)
            e = (e + 1) % n
        for e in path:
            cw_edges[i, e] = 1
        # CCW is complement
        ccw_edges[i] = 1 - cw_edges[i]

    # Weight vector
    weights = np.array([w for (_, _, w) in demands], dtype=np.float64)

    def compute_loads(routing):
        """Compute edge loads given a binary routing vector (0=CW, 1=CCW)."""
        # loads = sum over demands of weight * edge_mask
        # For each demand: if routing[i]==0 use cw_edges[i], else ccw_edges[i]
        # Vectorized: loads = weights @ (routing @ ccw_edges + (1-routing) @ cw_edges)
        # Simpler: start with all CW, then flip CCW ones
        # Actually: loads[e] = sum_i w_i * (routing[i]*ccw_edges[i,e] + (1-routing[i])*cw_edges[i,e])
        # = sum_i w_i * cw_edges[i,e] + sum_i w_i * routing[i] * (ccw_edges[i,e] - cw_edges[i,e])
        # Precompute: base_loads = weights @ cw_edges
        # diff = ccw_edges - cw_edges  (values: -1, 0, +1)
        # loads = base_loads + (weights * routing) @ diff
        return base_loads + (weights * routing) @ diff

    # Precompute base (all CW) and difference
    base_loads = weights @ cw_edges  # shape (n,)
    diff = ccw_edges - cw_edges      # shape (D, n), values in {-1, 0, 1}

    def max_load(routing):
        loads = compute_loads(routing)
        return np.max(loads)

    # Greedy initialization: route each demand to minimize resulting max load
    routing = np.zeros(D, dtype=np.float64)
    for i in range(D):
        # Try both directions
        loads_cw = compute_loads(routing)  # routing[i] is still 0
        loads_ccw = base_loads + (weights * routing) @ diff
        # Actually need to set routing[i] temporarily
        routing[i] = 0
        ml_cw = np.max(base_loads + (weights * routing) @ diff)
        routing[i] = 1
        ml_ccw = np.max(base_loads + (weights * routing) @ diff)
        routing[i] = 1 if ml_ccw < ml_cw else 0

    best_routing = routing.copy()
    best_ml = max_load(best_routing)

    # Simulated annealing
    T_start = 5.0
    T_end = 0.01
    n_iter = 50000
    cool_rate = (T_end / T_start) ** (1.0 / n_iter)

    current = best_routing.copy()
    current_ml = best_ml

    for it in range(n_iter):
        if time.time() > DEADLINE - 30.0:
            break

        T = T_start * (cool_rate ** it)

        # Pick a random demand to flip
        i = np.random.randint(D)
        current[i] = 1.0 - current[i]
        new_ml = max_load(current)
        delta = new_ml - current_ml

        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_ml = new_ml
            if new_ml < best_ml:
                best_ml = new_ml
                best_routing = current.copy()
        else:
            current[i] = 1.0 - current[i]  # revert

    # Local search polish: exhaustive single-flip neighborhood
    improved = True
    while improved and time.time() < DEADLINE - 10.0:
        improved = False
        for i in range(D):
            if time.time() > DEADLINE - 10.0:
                break
            old_val = best_routing[i]
            best_routing[i] = 1.0 - old_val
            ml = max_load(best_routing)
            if ml < best_ml:
                best_ml = ml
                improved = True
            else:
                best_routing[i] = old_val  # revert

    # Double-flip neighborhood search
    for i in range(D):
        if time.time() > DEADLINE - 5.0:
            break
        for j in range(i + 1, D):
            if time.time() > DEADLINE - 5.0:
                break
            old_i, old_j = best_routing[i], best_routing[j]
            best_routing[i] = 1.0 - old_i
            best_routing[j] = 1.0 - old_j
            ml = max_load(best_routing)
            if ml < best_ml:
                best_ml = ml
                improved = True
            else:
                best_routing[i] = old_i
                best_routing[j] = old_j

    # Final polish pass
    improved = True
    while improved and time.time() < DEADLINE - 2.0:
        improved = False
        for i in range(D):
            old_val = best_routing[i]
            best_routing[i] = 1.0 - old_val
            ml = max_load(best_routing)
            if ml < best_ml:
                best_ml = ml
                improved = True
            else:
                best_routing[i] = old_val

    return best_routing

# EVOLVE-BLOCK-END
