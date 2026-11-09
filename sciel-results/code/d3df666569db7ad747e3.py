# EVOLVE-BLOCK-START
import numpy as np
import time

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

    # Precompute clockwise edges for each demand
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod n)
    cw_edges = []
    for s, t, w in demands:
        edges = []
        e = s
        while e != t:
            edges.append(e)
            e = (e + 1) % n
        cw_edges.append(edges)

    # Counter-clockwise edges are all edges not in cw_edges
    all_edges = set(range(n))
    ccw_edges = []
    for edges in cw_edges:
        ccw_set = set(edges)
        ccw_edges.append(sorted(all_edges - ccw_set))

    # Build incidence matrix: A[d, e] = 1 if edge e is used when demand d is routed clockwise
    A = np.zeros((D, n), dtype=np.int32)
    for d in range(D):
        for e in cw_edges[d]:
            A[d, e] = 1

    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    def compute_loads(routing):
        """
        routing: array of length D, 0 for CW, 1 for CCW
        Returns loads array of length n
        """
        # Load = sum over demands of w_d * (A[d,:] if CW else (1-A[d,:]))
        # = sum_d w_d * (1 - 2*routing[d]) * A[d,:] + sum_d w_d * routing[d] * (1 - A[d,:])
        # Simpler: for each demand, if CW (0), add w to A[d,:]; if CCW (1), add w to (1-A[d,:])
        loads = np.zeros(n, dtype=np.float64)
        for d in range(D):
            if routing[d] < 0.5:  # CW
                loads[cw_edges[d]] += weights[d]
            else:  # CCW
                loads[ccw_edges[d]] += weights[d]
        return loads

    def max_load(routing):
        return np.max(compute_loads(routing))

    # Local search: single and double flips
    best_routing = np.zeros(D, dtype=np.float64)
    best_score = max_load(best_routing)

    deadline = time.time() + 600.0
    safety_margin = 30.0

    def time_remaining():
        return deadline - time.time() - safety_margin

    # Single-flip local search
    improved = True
    while improved and time_remaining() > 0:
        improved = False
        for d in range(D):
            trial = best_routing.copy()
            trial[d] = 1.0 - trial[d]
            score = max_load(trial)
            if score < best_score - 1e-12:
                best_routing = trial
                best_score = score
                improved = True

    # Double-flip local search
    improved = True
    while improved and time_remaining() > 0:
        improved = False
        for d1 in range(D):
            for d2 in range(d1 + 1, D):
                trial = best_routing.copy()
                trial[d1] = 1.0 - trial[d1]
                trial[d2] = 1.0 - trial[d2]
                score = max_load(trial)
                if score < best_score - 1e-12:
                    best_routing = trial
                    best_score = score
                    improved = True
                    break
            if improved:
                break

    # Multi-start with random restarts
    best_routing_overall = best_routing.copy()
    best_score_overall = best_score

    np.random.seed(42)
    max_restarts = 200
    for r in range(max_restarts):
        if time_remaining() <= 0:
            break
        routing = np.random.rand(D)  # random 0/1
        routing = (routing >= 0.5).astype(np.float64)

        # Local search from this start
        improved = True
        while improved and time_remaining() > 0:
            improved = False
            for d in range(D):
                trial = routing.copy()
                trial[d] = 1.0 - trial[d]
                score = max_load(trial)
                if score < best_score - 1e-12:
                    routing = trial
                    best_score = score
                    improved = True

        improved = True
        while improved and time_remaining() > 0:
            improved = False
            for d1 in range(D):
                for d2 in range(d1 + 1, D):
                    trial = routing.copy()
                    trial[d1] = 1.0 - trial[d1]
                    trial[d2] = 1.0 - trial[d2]
                    score = max_load(trial)
                    if score < best_score - 1e-12:
                        routing = trial
                        best_score = score
                        improved = True
                        break
                if improved:
                    break

        if best_score < best_score_overall - 1e-12:
            best_routing_overall = routing.copy()
            best_score_overall = best_score

    # Final polish: try all single and double flips from the best
    routing = best_routing_overall.copy()
    best_score = best_score_overall

    for _ in range(3):  # a few polish passes
        improved = True
        while improved and time_remaining() > 0:
            improved = False
            for d in range(D):
                trial = routing.copy()
                trial[d] = 1.0 - trial[d]
                score = max_load(trial)
                if score < best_score - 1e-12:
                    routing = trial
                    best_score = score
                    improved = True

        improved = True
        while improved and time_remaining() > 0:
            improved = False
            for d1 in range(D):
                for d2 in range(d1 + 1, D):
                    trial = routing.copy()
                    trial[d1] = 1.0 - trial[d1]
                    trial[d2] = 1.0 - trial[d2]
                    score = max_load(trial)
                    if score < best_score - 1e-12:
                        routing = trial
                        best_score = score
                        improved = True
                        break
                if improved:
                    break

    return routing

# EVOLVE-BLOCK-END
