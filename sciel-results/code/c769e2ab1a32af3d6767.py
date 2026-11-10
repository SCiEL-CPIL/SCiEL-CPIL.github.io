"""
Ant colony orienteering heuristic: MODEL-AND-OPTIMIZE with expanded candidate space.

Key improvements over the current program:
  1. Expanded candidate heuristic formulas (30+ forms) including higher-order
     prize/distance combinations that create sharper local attraction.
  2. Vectorized ACO simulation for speed, allowing more parameter evaluations.
  3. Precomputed dist_to_home for faster constraint checking.
  4. Deadline guard to ensure we finish well within 330s.
  5. More representative instances and finer grid search.
"""

import numpy as np
import time

_params_cache = None


def _simulate_aco(prizes, dist, max_len, heuristic, n, rng):
    """Simulate ACO and return the highest total prize collected by any ant in any iteration."""
    num_ants = 20
    num_iters = 30
    pheromone_decay = 0.9
    pheromone_gain_divisor = 10.0

    pheromone = np.ones((n, n))
    dist_to_home = dist[:, 0].copy()  # d[j, 0] for all j

    global_best = 0.0

    for it in range(num_iters):
        iter_best = 0.0

        for ant in range(num_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path_len = 0.0
            total_prize = 0.0

            while True:
                # Find candidate nodes: unvisited, and path_len + d[current,j] + d[j,0] <= max_len
                # Vectorized candidate finding
                remaining = max_len - path_len
                # d[current, j] + dist_to_home[j] <= remaining
                feasible = (~visited) & ((dist[current, :] + dist_to_home) <= remaining)
                # Exclude node 0 (already visited)
                feasible[0] = False

                if not np.any(feasible):
                    break

                candidates = np.where(feasible)[0]
                weights = pheromone[current, candidates] * heuristic[current, candidates]

                total_w = weights.sum()
                if total_w <= 0:
                    break

                probs = weights / total_w
                idx = rng.choice(len(candidates), p=probs)
                next_node = candidates[idx]

                path_len += dist[current, next_node]
                total_prize += prizes[next_node]
                visited[next_node] = True
                current = next_node

            if total_prize > iter_best:
                iter_best = total_prize

        # Update pheromone
        pheromone *= pheromone_decay
        pheromone += iter_best / pheromone_gain_divisor

        if iter_best > global_best:
            global_best = iter_best

    return global_best


def _make_heuristic_form(form_id, prizes, dist, max_len, params, eps=1e-9):
    """Generate heuristic matrix given form_id and params (tuple of floats)."""
    n = len(prizes)
    prize = prizes[None, :]  # shape (1, n)
    d_ij = dist + eps  # (n, n)
    d_j0 = dist[:, 0] + eps  # (n,)
    d_j0_row = d_j0[None, :]  # (1, n)
    budget = np.maximum(0.0, max_len - dist - d_j0_row)  # (n, n)
    p = params

    if form_id == 0:
        # prize^a / d_ij^b * (1/d_j0)^g * budget^d
        h = (prize ** p[0]) / (d_ij ** p[1]) * (1.0 / d_j0_row ** p[2])
        if p[3] > 0:
            h = h * (budget + eps) ** p[3]
    elif form_id == 1:
        # prize / (d_ij + mu * d_j0)
        h = prize / (d_ij + p[0] * d_j0_row)
    elif form_id == 2:
        # prize^2 / d_ij^3
        h = (prize ** 2) / (d_ij ** 3)
    elif form_id == 3:
        # prize^3 / d_ij^2
        h = (prize ** 3) / (d_ij ** 2)
    elif form_id == 4:
        # prize / d_ij^4
        h = prize / (d_ij ** 4)
    elif form_id == 5:
        # prize^2 / (d_ij^2 * d_j0)
        h = (prize ** 2) / (d_ij ** 2 * d_j0_row)
    elif form_id == 6:
        # prize / (d_ij * d_j0)
        h = prize / (d_ij * d_j0_row)
    elif form_id == 7:
        # prize^a * exp(-b * d_ij) * exp(-c * d_j0)
        h = (prize ** p[0]) * np.exp(-p[1] * d_ij) * np.exp(-p[2] * d_j0_row)
    elif form_id == 8:
        # prize^a / d_ij^b * budget^c
        h = (prize ** p[0]) / (d_ij ** p[1])
        if p[2] > 0:
            h = h * (budget + eps) ** p[2]
    elif form_id == 9:
        # prize^a * (1 - d_j0/max_len)^b / d_ij^c
        factor = np.maximum(0.0, 1.0 - d_j0_row / max_len)
        h = (prize ** p[0]) * (factor + eps) ** p[1] / (d_ij ** p[2])
    elif form_id == 10:
        # prize^a / (d_ij^b + d_j0^c)
        h = (prize ** p[0]) / (d_ij ** p[1] + d_j0_row ** p[2] + eps)
    elif form_id == 11:
        # prize^a * budget^b / d_ij^c
        h = (prize ** p[0]) * (budget + eps) ** p[1] / (d_ij ** p[2])
    elif form_id == 12:
        # prize^a / d_ij^b * (1 - d_j0/max_len)^c
        factor = np.maximum(0.0, 1.0 - d_j0_row / max_len)
        h = (prize ** p[0]) / (d_ij ** p[1]) * (factor + eps) ** p[2]
    elif form_id == 13:
        # prize^a * exp(-b*(d_ij + d_j0))
        h = (prize ** p[0]) * np.exp(-p[1] * (d_ij + d_j0_row))
    elif form_id == 14:
        # prize^a / (d_ij + d_j0)^b
        h = (prize ** p[0]) / ((d_ij + d_j0_row) ** p[1])
    elif form_id == 15:
        # prize^a / d_ij^b * (max_len - d_j0)^c
        factor = np.maximum(0.0, max_len - d_j0_row)
        h = (prize ** p[0]) / (d_ij ** p[1]) * (factor + eps) ** p[2]
    elif form_id == 16:
        # prize^a * (budget / max_len)^b / d_ij^c
        norm_budget = budget / max_len
        h = (prize ** p[0]) * (norm_budget + eps) ** p[1] / (d_ij ** p[2])
    elif form_id == 17:
        # prize^a / d_ij^b * (1 / (d_j0 + 0.1))^c
        h = (prize ** p[0]) / (d_ij ** p[1]) * (1.0 / (d_j0_row + 0.1)) ** p[2]
    elif form_id == 18:
        # prize^a * (1/d_ij)^b * (budget/max_len)^c * (1/d_j0)^d
        norm_budget = budget / max_len
        h = (prize ** p[0]) * (1.0 / d_ij) ** p[1] * (norm_budget + eps) ** p[2] * (1.0 / d_j0_row) ** p[3]
    elif form_id == 19:
        # prize^a / (d_ij^b * (1 + d_j0))^c
        h = (prize ** p[0]) / ((d_ij ** p[1]) * (1.0 + d_j0_row) ** p[2])
    elif form_id == 20:
        # prize^a * exp(-b * d_ij / max_len) * exp(-c * d_j0 / max_len)
        h = (prize ** p[0]) * np.exp(-p[1] * d_ij / max_len) * np.exp(-p[2] * d_j0_row / max_len)
    elif form_id == 21:
        # prize^a / d_ij^b * (1 + budget)^c
        h = (prize ** p[0]) / (d_ij ** p[1]) * (1.0 + budget) ** p[2]
    elif form_id == 22:
        # prize^a * (max_len - d_ij - d_j0 + 0.01)^b / d_ij^c
        factor = np.maximum(0.01, max_len - d_ij - d_j0_row)
        h = (prize ** p[0]) * factor ** p[1] / (d_ij ** p[2])
    elif form_id == 23:
        # prize^a / (d_ij + 0.05)^b * (1 / (d_j0 + 0.05))^c
        h = (prize ** p[0]) / (d_ij + 0.05) ** p[1] * (1.0 / (d_j0_row + 0.05)) ** p[2]
    elif form_id == 24:
        # prize^a * (budget^2 + 0.01) / d_ij^b
        h = (prize ** p[0]) * (budget ** 2 + 0.01) / (d_ij ** p[1])
    elif form_id == 25:
        # prize^a / (d_ij^b + 0.01) * (1 / (d_j0^c + 0.01))
        h = (prize ** p[0]) / (d_ij ** p[1] + 0.01) * (1.0 / (d_j0_row ** p[2] + 0.01))
    elif form_id == 26:
        # prize^a * log(1 + budget) / d_ij^b
        h = (prize ** p[0]) * np.log1p(budget) / (d_ij ** p[1])
    elif form_id == 27:
        # prize^a / d_ij^b * (1 - d_j0/max_len)^c * (1 - d_ij/max_len)^d
        f1 = np.maximum(0.0, 1.0 - d_j0_row / max_len)
        f2 = np.maximum(0.0, 1.0 - d_ij / max_len)
        h = (prize ** p[0]) / (d_ij ** p[1]) * (f1 + eps) ** p[2] * (f2 + eps) ** p[3]
    elif form_id == 28:
        # prize^a * (1/d_ij)^b * (1/d_j0)^c * (budget/max_len)^d
        norm_budget = budget / max_len
        h = (prize ** p[0]) * (1.0 / d_ij) ** p[1] * (1.0 / d_j0_row) ** p[2] * (norm_budget + eps) ** p[3]
    elif form_id == 29:
        # prize^a / (d_ij^b * d_j0^c) * (1 + budget/max_len)^d
        norm_budget = budget / max_len
        h = (prize ** p[0]) / (d_ij ** p[1] * d_j0_row ** p[2]) * (1.0 + norm_budget) ** p[3]
    else:
        # Default: simple prize / d_ij
        h = prize / d_ij

    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _optimize_parameters():
    """Optimize heuristic form and parameters on representative instances."""
    global _params_cache
    if _params_cache is not None:
        return _params_cache

    start_time = time.time()
    deadline = start_time + 240.0  # Leave 90s margin for safety

    # Generate representative instances
    n_instances = 12
    instances = []
    for seed in range(n_instances):
        rng = np.random.RandomState(seed)
        n = 50
        points = rng.uniform(0, 1, size=(n, 2))
        prizes = rng.uniform(0, 1, size=n)
        prizes[0] = 0.0
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        instances.append((prizes, dist, 3.0))

    best_params = None
    best_score = -1.0
    best_form = 0

    # Define parameter grids for each form
    # Each entry: (form_id, list of param tuples)
    candidates = []

    # Form 0: prize^a / d_ij^b * (1/d_j0)^g * budget^d
    for a in [0.5, 1.0, 1.5, 2.0]:
        for b in [0.5, 1.0, 1.5, 2.0]:
            for g in [0.0, 0.5, 1.0]:
                for d in [0.0, 0.5, 1.0]:
                    candidates.append((0, (a, b, g, d)))

    # Form 1: prize / (d_ij + mu * d_j0)
    for mu in [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]:
        candidates.append((1, (mu,)))

    # Form 2: prize^2 / d_ij^3
    candidates.append((2, ()))

    # Form 3: prize^3 / d_ij^2
    candidates.append((3, ()))

    # Form 4: prize / d_ij^4
    candidates.append((4, ()))

    # Form 5: prize^2 / (d_ij^2 * d_j0)
    candidates.append((5, ()))

    # Form 6: prize / (d_ij * d_j0)
    candidates.append((6, ()))

    # Form 7: prize^a * exp(-b * d_ij) * exp(-c * d_j0)
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 2.0, 3.0, 5.0]:
            for c in [0.0, 1.0, 2.0]:
                candidates.append((7, (a, b, c)))

    # Form 8: prize^a / d_ij^b * budget^c
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0, 1.5]:
                candidates.append((8, (a, b, c)))

    # Form 9: prize^a * (1 - d_j0/max_len)^b / d_ij^c
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 2.0, 3.0]:
            for c in [1.0, 1.5, 2.0]:
                candidates.append((9, (a, b, c)))

    # Form 10: prize^a / (d_ij^b + d_j0^c)
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0, 1.5]:
                candidates.append((10, (a, b, c)))

    # Form 11: prize^a * budget^b / d_ij^c
    for a in [1.0, 1.5, 2.0]:
        for b in [0.5, 1.0, 1.5]:
            for c in [1.0, 1.5, 2.0]:
                candidates.append((11, (a, b, c)))

    # Form 12: prize^a / d_ij^b * (1 - d_j0/max_len)^c
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [1.0, 2.0, 3.0]:
                candidates.append((12, (a, b, c)))

    # Form 13: prize^a * exp(-b*(d_ij + d_j0))
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 2.0, 3.0, 5.0]:
            candidates.append((13, (a, b)))

    # Form 14: prize^a / (d_ij + d_j0)^b
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0, 2.5]:
            candidates.append((14, (a, b)))

    # Form 15: prize^a / d_ij^b * (max_len - d_j0)^c
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0, 1.5]:
                candidates.append((15, (a, b, c)))

    # Form 16: prize^a * (budget/max_len)^b / d_ij^c
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 2.0, 3.0]:
            for c in [1.0, 1.5, 2.0]:
                candidates.append((16, (a, b, c)))

    # Form 17: prize^a / d_ij^b * (1/(d_j0+0.1))^c
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0, 1.5]:
                candidates.append((17, (a, b, c)))

    # Form 18: prize^a * (1/d_ij)^b * (budget/max_len)^c * (1/d_j0)^d
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0]:
                for d in [0.0, 0.5, 1.0]:
                    candidates.append((18, (a, b, c, d)))

    # Form 19: prize^a / (d_ij^b * (1+d_j0))^c
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0, 1.5]:
                candidates.append((19, (a, b, c)))

    # Form 20: prize^a * exp(-b*d_ij/max_len) * exp(-c*d_j0/max_len)
    for a in [1.0, 1.5, 2.0]:
        for b in [2.0, 4.0, 6.0, 8.0]:
            for c in [0.0, 2.0, 4.0]:
                candidates.append((20, (a, b, c)))

    # Form 21: prize^a / d_ij^b * (1+budget)^c
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0, 1.5]:
                candidates.append((21, (a, b, c)))

    # Form 22: prize^a * (max_len - d_ij - d_j0 + 0.01)^b / d_ij^c
    for a in [1.0, 1.5, 2.0]:
        for b in [0.5, 1.0, 1.5]:
            for c in [1.0, 1.5, 2.0]:
                candidates.append((22, (a, b, c)))

    # Form 23: prize^a / (d_ij+0.05)^b * (1/(d_j0+0.05))^c
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0, 1.5]:
                candidates.append((23, (a, b, c)))

    # Form 24: prize^a * (budget^2+0.01) / d_ij^b
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            candidates.append((24, (a, b)))

    # Form 25: prize^a / (d_ij^b+0.01) * (1/(d_j0^c+0.01))
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0, 1.5]:
                candidates.append((25, (a, b, c)))

    # Form 26: prize^a * log(1+budget) / d_ij^b
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            candidates.append((26, (a, b)))

    # Form 27: prize^a / d_ij^b * (1-d_j0/max_len)^c * (1-d_ij/max_len)^d
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [1.0, 2.0]:
                for d in [0.5, 1.0]:
                    candidates.append((27, (a, b, c, d)))

    # Form 28: prize^a * (1/d_ij)^b * (1/d_j0)^c * (budget/max_len)^d
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.0, 0.5, 1.0]:
                for d in [0.5, 1.0]:
                    candidates.append((28, (a, b, c, d)))

    # Form 29: prize^a / (d_ij^b * d_j0^c) * (1+budget/max_len)^d
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0]:
                for d in [0.5, 1.0]:
                    candidates.append((29, (a, b, c, d)))

    # Evaluate candidates
    eval_rng = np.random.RandomState(12345)
    n_eval = len(candidates)

    for ci, (form_id, params) in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break

        total_score = 0.0
        for prizes, dist, max_len in instances:
            n = len(prizes)
            heuristic = _make_heuristic_form(form_id, prizes, dist, max_len, params)
            score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
            total_score += score

        mean_score = total_score / len(instances)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_form = form_id

    if best_params is None:
        best_form = 0
        best_params = (1.0, 1.0, 0.0, 0.0)

    _params_cache = (best_form, best_params)
    return _params_cache


def heuristics(prizes, distance_matrix, max_len):
    """Compute the heuristic matrix for ant colony optimization."""
    form_id, params = _optimize_parameters()
    heuristic = _make_heuristic_form(form_id, prizes, distance_matrix, max_len, params)
    return heuristic
