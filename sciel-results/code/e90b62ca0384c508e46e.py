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
        0 -> clockwise, >= 0.5 -> counter-clockwise
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute: for each demand, which edges are used if CW, which if CCW
    # CW from s to t: edges s, s+1, ..., t-1 (mod 10)
    # CCW: all other edges
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = set()
        e = s
        while e != t:
            cw.add(e)
            e = (e + 1) % n
        all_edges = set(range(n))
        ccw = all_edges - cw
        cw_edges.append(cw)
        ccw_edges.append(ccw)

    # Precompute weight per demand
    weights = np.array([w for _, _, w in demands])

    # Brute force: try all 2^15 = 32768 combinations
    # For efficiency, use incremental computation
    best_max_load = float('inf')
    best_routing = None

    # Use a more efficient approach: iterate through all 2^D combinations
    # For each combination, compute edge loads
    for mask in range(1 << D):
        if time.time() > deadline:
            break

        # Compute edge loads for this assignment
        edge_loads = np.zeros(n)
        for d in range(D):
            w = weights[d]
            if (mask >> d) & 1:
                # CCW
                for e in ccw_edges[d]:
                    edge_loads[e] += w
            else:
                # CW
                for e in cw_edges[d]:
                    edge_loads[e] += w

        max_load = edge_loads.max()
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = mask

            # Early termination: if we found a very good solution
            # Theoretical lower bound: total weight / n
            total_weight = weights.sum()
            lower_bound = total_weight / n
            if best_max_load == lower_bound:
                break

    # Convert best_routing mask to output array
    # 0 -> clockwise, 1 -> counter-clockwise
    # Output: >= 0.5 means CCW
    result = np.zeros(D)
    if best_routing is not None:
        for d in range(D):
            if (best_routing >> d) & 1:
                result[d] = 1.0
            else:
                result[d] = 0.0
    else:
        # Fallback: all clockwise
        result = np.zeros(D)

    return result

# EVOLVE-BLOCK-END
