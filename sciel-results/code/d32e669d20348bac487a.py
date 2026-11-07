import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin before the 300s limit

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)

    if time.time() > deadline:
        return np.ones((n, n)) * 1e-9

    dist_to_origin = distance_matrix[:, 0]
    eps = 1e-9

    # Precompute the "remaining budget" after visiting j from i:
    # budget[i, j] = max_len - (d[i, j] + d[j, 0])
    # If this is <= 0, the edge is infeasible (can't even go i->j->0)
    remaining = max_len - (distance_matrix + dist_to_origin[None, :])
    feasible = remaining > 0

    def get_prize(eta):
        """Greedy evaluation: follow the heuristic to maximize prize."""
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
                # Constraint: total_len + d[curr, j] + d[j, 0] <= max_len
                new_len = total_len + distance_matrix[curr, j] + dist_to_origin[j]
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

    # Heuristic family:
    # eta[i, j] = prize[j]^a / (d[i,j]^b * d[j,0]^g) * slack_factor
    # where slack_factor rewards edges that leave more room for future nodes
    #
    # Alternative: use a "detour-aware" ratio
    # eta[i, j] = prize[j]^a / (d[i,j] + d[j,0])^b
    #
    # Let's try a combined approach with a slack bonus

    def compute_heuristic(a, b, g, slack_w):
        """
        eta[i,j] = (prize[j] + eps)^a / ((d[i,j] + eps)^b * (d[j,0] + eps)^g)
                   * (1 + slack_w * max(0, remaining[i,j] / max_len))
        """
        prize_term = (prizes + eps) ** a
        dist_ij_term = (distance_matrix + eps) ** b
        dist_j0_term = (dist_to_origin + eps) ** g

        base = prize_term[None, :] / (dist_ij_term * dist_j0_term[None, :])

        # Slack factor: reward edges that leave more remaining budget
        slack = np.maximum(remaining, 0.0) / max_len
        slack_factor = 1.0 + slack_w * slack

        eta = base * slack_factor[None, :]
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def evaluate_params(a, b, g, slack_w):
        eta = compute_heuristic(a, b, g, slack_w)
        return get_prize(eta)

    # Phase 1: Coarse grid search
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    betas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    gammas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
    slack_ws = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    best_prize = -1.0
    best_params = (3.0, 1.5, 1.0, 1.0)

    for a in alphas:
        prize_term = (prizes + eps) ** a
        for b in betas:
            dist_ij_term = (distance_matrix + eps) ** b
            for g in gammas:
                dist_j0_term = (dist_to_origin + eps) ** g
                for sw in slack_ws:
                    if time.time() > deadline - 30:
                        break

                    eta = prize_term[None, :] / (dist_ij_term * dist_j0_term[None, :])
                    if sw > 0:
                        slack = np.maximum(remaining, 0.0) / max_len
                        eta = eta * (1.0 + sw * slack)[None, :]
                    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)

                    p = get_prize(eta)
                    if p > best_prize:
                        best_prize = p
                        best_params = (a, b, g, sw)

                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break

    # Phase 2: Local search around best parameters
    if time.time() < deadline - 20:
        a0, b0, g0, sw0 = best_params
        step_sizes = [0.25, 0.1, 0.05, 0.02, 0.01]

        for step in step_sizes:
            improved = False
            for da in [-step, 0, step]:
                for db in [-step, 0, step]:
                    for dg in [-step, 0, step]:
                        for dsw in [-step, 0, step]:
                            if da == 0 and db == 0 and dg == 0 and dsw == 0:
                                continue
                            a = a0 + da
                            b = b0 + db
                            g = g0 + dg
                            sw = sw0 + dsw

                            if a < 0.1 or b < 0.1 or g < 0.1 or sw < 0:
                                continue
                            if a > 12 or b > 6 or g > 6 or sw > 5:
                                continue

                            if time.time() > deadline - 10:
                                break

                            p = evaluate_params(a, b, g, sw)
                            if p > best_prize:
                                best_prize = p
                                best_params = (a, b, g, sw)
                                a0, b0, g0, sw0 = a, b, g, sw
                                improved = True

                        if time.time() > deadline - 10:
                            break
                    if time.time() > deadline - 10:
                        break
                if time.time() > deadline - 10:
                    break

            if not improved and step < 0.05:
                break

    a, b, g, sw = best_params
    heuristic = compute_heuristic(a, b, g, sw)

    # Ensure shape and validity
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    return heuristic
