# EVOLVE-BLOCK-START
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem using Branch and Bound with graph-based load calculation.
    This approach explores the routing space (partitions of demands) and prunes branches
    where the partial load already exceeds the best known maximum load.
    """
    n_nodes = 10
    n_edges = 10
    D = 15

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

    # Precompute edge lists for CW and CCW for each demand
    # cw_edges[d] is a list of edge indices
    # ccw_edges[d] is a list of edge indices
    cw_edges = [[] for _ in range(D)]
    ccw_edges = [[] for _ in range(D)]
    weights = [0] * D

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        # CW: s -> s+1 -> ... -> t
        cur = s
        while cur != t:
            cw_edges[d].append(cur)
            cur = (cur + 1) % n_nodes
        
        # CCW: All other edges
        cw_set = set(cw_edges[d])
        for e in range(n_edges):
            if e not in cw_set:
                ccw_edges[d].append(e)

    # Initial upper bound: Route everything CW
    initial_loads = [0.0] * n_edges
    for d in range(D):
        for e in cw_edges[d]:
            initial_loads[e] += weights[d]
    best_max_load = max(initial_loads)
    best_routing = np.zeros(D) # All CW

    # Branch and Bound
    # state: index of demand being decided, current loads
    def solve(idx, loads, routing):
        nonlocal best_max_load, best_routing
        
        if idx == D:
            current_max = max(loads)
            if current_max < best_max_load:
                best_max_load = current_max
                best_routing = routing.copy()
            return

        # Pruning: If any current load already exceeds best_max_load, 
        # adding more demands (positive weights) will only increase loads.
        # So we can prune this branch.
        if max(loads) >= best_max_load:
            return

        # Option 1: Route demand idx CW
        # Apply weights
        for e in cw_edges[idx]:
            loads[e] += weights[idx]
        
        routing[idx] = 0.0
        solve(idx + 1, loads, routing)
        
        # Undo
        for e in cw_edges[idx]:
            loads[e] -= weights[idx]

        # Option 2: Route demand idx CCW
        for e in ccw_edges[idx]:
            loads[e] += weights[idx]
        
        routing[idx] = 1.0
        solve(idx + 1, loads, routing)
        
        # Undo
        for e in ccw_edges[idx]:
            loads[e] -= weights[idx]

    solve(0, [0.0] * n_edges, np.zeros(D))

    return best_routing

# EVOLVE-BLOCK-END
