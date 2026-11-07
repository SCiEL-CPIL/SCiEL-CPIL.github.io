import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)

    if time.time() > deadline:
        return np.ones((n, n)) * 1e-9

    # Precompute
    dist_to_origin = distance_matrix[:, 0]
    total_cost = distance_matrix + dist_to_origin[None, :]  # (n, n)
    max_cost = np.max(total_cost)
    if max_cost <= 0:
        max_cost = 1.0

    # Greedy evaluator: simulates one ant using the heuristic as a deterministic guide
    def get_prize(eta):
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        curr = 0
        total_len = 0.0
        total_prize = 0.0

        for _ in range(n):
            best_val = -1.0
            best_j = -1
            for j in range(1, n):
                if visited[j]:
                    continue
                new_len = total_len + distance_matrix[curr, j] + distance_matrix[j, 0]
                if new_len <= max_len:
                    val = eta[curr, j]
                    if val > best_val:
                        best_val = val
                        best_j = j
            if best_j == -1:
                break
            total_len += distance_matrix[curr, best_j]
            total_prize += prizes[best_j]
            visited[best_j] = True
            curr = best_j
        return total_prize

    eps = 1e-9

    # Precompute prize powers and distance powers for grid
    # Heuristic: prize[j]^a * neighbor_density[j]^nd / (d[i,j]^b * d[j,0]^g) * slack^s
    # neighbor_density[j] = sum of prizes of feasible neighbors of j (precomputed)

    # Compute neighbor density: for each node j, sum of prizes of nodes k
    # such that d[j,k] + d[k,0] <= max_len (i.e., k is reachable from j and back)
    neighbor_density = np.zeros(n)
    for j in range(n):
        for k in range(n):
            if k == j:
                continue
            if distance_matrix[j, k] + dist_to_origin[k] <= max_len:
                neighbor_density[j] += prizes[k]
    neighbor_density = neighbor_density + eps  # avoid log(0) or pow(0)

    def compute_heuristic(a, b, g, s, nd):
        prize_term = (prizes + eps) ** a
        nd_term = neighbor_density ** nd
        dist_ij_term = (distance_matrix + eps) ** b
        dist_j0_term = (dist_to_origin + eps) ** g
        eta = (prize_term * nd_term)[None, :] / (dist_ij_term * dist_j0_term[None, :])

        if s > 1e-6:
            slack_factor = 1.0 - s * (total_cost / max_cost)
            slack_factor = np.clip(slack_factor, 0.1, None)
            eta = eta * slack_factor

        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    # Phase 1: Coarse grid search
    alphas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    betas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    gammas = [0.0, 0.5, 1.0, 1.5, 2.0]
    slacks = [0.0, 0.3, 0.6, 0.9]
    nds = [0.0, 0.5, 1.0, 1.5]

    best_prize = -1.0
    best_params = (3.0, 1.0, 1.0, 0.0, 0.5)

    for a in alphas:
        prize_term = (prizes + eps) ** a
        for b in betas:
            dist_ij_term = (distance_matrix + eps) ** b
            for g in gammas:
                dist_j0_term = (dist_to_origin + eps) ** g
                for s in slacks:
                    for nd in nds:
                        if time.time() > deadline - 10:
                            break
                        nd_term = neighbor_density ** nd
                        eta = (prize_term * nd_term)[None, :] / (dist_ij_term * dist_j0_term[None, :])

                        if s > 1e-6:
                            slack_factor = 1.0 - s * (total_cost / max_cost)
                            slack_factor = np.clip(slack_factor, 0.1, None)
                            eta = eta * slack_factor

                        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)

                        p = get_prize(eta)
                        if p > best_prize:
                            best_prize = p
                            best_params = (a, b, g, s, nd)

                        if time.time() > deadline - 10:
                            break
                    if time.time() > deadline - 10:
                        break
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break

    # Phase 2: Local refinement around best params
    if time.time() < deadline - 15:
        a0, b0, g0, s0, nd0 = best_params
        steps = [0.3, 0.1, 0.05, 0.02, 0.01]

        for step in steps:
            improved = False
            offsets = [-step, -step/2, 0, step/2, step]
            for da in offsets:
                for db in offsets:
                    for dg in offsets:
                        for ds in offsets:
                            for dnd in offsets:
                                if da == 0 and db == 0 and dg == 0 and ds == 0 and dnd == 0:
                                    continue
                                a = a0 + da
                                b = b0 + db
                                g = g0 + dg
                                s = s0 + ds
                                nd = nd0 + dnd

                                if a < 0.1 or b < 0.0 or g < 0.0 or s < 0.0 or nd < 0.0:
                                    continue
                                if a > 15 or b > 5 or g > 5 or s > 2.0 or nd > 3.0:
                                    continue

                                if time.time() > deadline - 5:
                                    break

                                p = get_prize(compute_heuristic(a, b, g, s, nd))
                                if p > best_prize:
                                    best_prize = p
                                    best_params = (a, b, g, s, nd)
                                    a0, b0, g0, s0, nd0 = a, b, g, s, nd
                                    improved = True

                                if time.time() > deadline - 5:
                                    break
                            if time.time() > deadline - 5:
                                break
                        if time.time() > deadline - 5:
                            break
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break

            if not improved:
                if step == steps[-1]:
                    break

    a, b, g, s, nd = best_params
    heuristic = compute_heuristic(a, b, g, s, nd)
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    return heuristic
