# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute CW edge masks for each demand
    # CW from s to t uses edges s, s+1, ..., t-1 (mod N)
    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw_mask = 0
        if s == t:
            # Full ring
            cw_mask = (1 << N) - 1
        else:
            e = s
            while e != t:
                cw_mask |= (1 << e)
                e = (e + 1) % N
        ccw_mask = ((1 << N) - 1) ^ cw_mask
        cw_masks.append(cw_mask)
        ccw_masks.append(ccw_masks)

    # Weighted masks: each edge's contribution is weight * (edge in mask)
    # For fast evaluation, precompute per-demand contribution vectors
    cw_contrib = np.zeros((D, N), dtype=np.float64)
    ccw_contrib = np.zeros((D, N), dtype=np.float64)
    for i, (s, t, w) in enumerate(demands):
        for e in range(N):
            if (cw_masks[i] >> e) & 1:
                cw_contrib[i, e] = w
            if (ccw_masks[i] >> e) & 1:
                ccw_contrib[i, e] = w

    # Fast max-load evaluation using precomputed contributions
    def eval_max_load(routing_bits):
        # routing_bits: list of 0/1, 0=CW, 1=CCW
        loads = np.zeros(N, dtype=np.float64)
        for i in range(D):
            if routing_bits[i]:
                loads += ccw_contrib[i]
            else:
                loads += cw_contrib[i]
        return loads.max()

    # Initial solution: all CW
    best_bits = [0] * D
    best_load = eval_max_load(best_bits)

    # Greedy: for each demand, pick direction that minimizes max load
    # Do multiple passes
    for _pass in range(20):
        improved = False
        for i in range(D):
            # Try flipping
            old_val = best_bits[i]
            best_bits[i] = 1 - old_val
            new_load = eval_max_load(best_bits)
            if new_load < best_load:
                best_load = new_load
                improved = True
            else:
                best_bits[i] = old_val
        if not improved:
            break

    # Local search with random restarts and time budget
    start_time = time.time()
    deadline = 620.0  # safety margin before 630s

    # Simulated annealing
    import random
    random.seed(42)

    current_bits = best_bits[:]
    current_load = best_load
    T = 1.0
    T_min = 0.01
    cooling = 0.999

    # Also try random restarts with greedy
    while time.time() < deadline:
        # Simulated annealing step
        if T > T_min:
            i = random.randint(0, D - 1)
            current_bits[i] = 1 - current_bits[i]
            new_load = eval_max_load(current_bits)
            delta = new_load - current_load
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_load = new_load
                if current_load < best_load:
                    best_load = current_load
                    best_bits = current_bits[:]
            else:
                current_bits[i] = 1 - current_bits[i]
            T *= cooling
        else:
            # Random restart with greedy
            random_bits = [random.randint(0, 1) for _ in range(D)]
            rand_load = eval_max_load(random_bits)
            # Greedy improvement from random start
            for _p in range(10):
                imp = False
                for j in range(D):
                    old = random_bits[j]
                    random_bits[j] = 1 - old
                    nl = eval_max_load(random_bits)
                    if nl < rand_load:
                        rand_load = nl
                        imp = True
                    else:
                        random_bits[j] = old
                if not imp:
                    break
            if rand_load < best_load:
                best_load = rand_load
                best_bits = random_bits[:]
                # Reset SA
                current_bits = best_bits[:]
                current_load = best_load
                T = 1.0

    # Final greedy polish
    for _pass in range(30):
        improved = False
        for i in range(D):
            old_val = best_bits[i]
            best_bits[i] = 1 - old_val
            new_load = eval_max_load(best_bits)
            if new_load < best_load:
                best_load = new_load
                improved = True
            else:
                best_bits[i] = old_val
        if not improved:
            break

    # Return: >= 0.5 means CCW, else CW
    result = np.array([1.0 if b else 0.0 for b in best_bits])
    return result


# EVOLVE-BLOCK-END
