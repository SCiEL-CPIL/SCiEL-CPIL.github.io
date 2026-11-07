"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 280.0  # Safety margin (evaluator kills at 330s)

    N = len(prizes)

    # Precompute d_j0 = distance from each node to node 0
    d_j0 = distance_matrix[:, 0].copy()
    # Precompute round-trip cost d[i,j] + d[j,0]
    d_total = distance_matrix + d_j0[None, :]

    # Precompute detour costs: d[i,j] + d[j,0] - d[i,0]
    d_i0 = distance_matrix[:, 0].copy()
    detour = distance_matrix + d_j0[None, :] - d_i0[:, None]
    detour = np.maximum(detour, 0.0)

    # Vectorized ACO simulation
    def run_aco(eta, seed):
        rng = np.random.RandomState(seed)
        tau = np.ones((N, N))
        best_prize_global = 0.0

        for it in range(30):
            best_prize_iter = 0.0
            for ant in range(20):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = np.zeros(N, dtype=bool)
                visited[0] = True
                while True:
                    remaining = max_len - path_len
                    eligible_mask = (~visited) & (d_total[current, :] <= remaining + 1e-12)
                    eligible = np.where(eligible_mask)[0]
                    if len(eligible) == 0:
                        break
                    weights = tau[current, eligible] * eta[current, eligible]
                    total_w = weights.sum()
                    if total_w <= 0:
                        break
                    probs = weights / total_w
                    r = rng.rand()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible):
                        idx = len(eligible) - 1
                    next_node = eligible[idx]
                    path_len += distance_matrix[current, next_node]
                    prize_total += prizes[next_node]
                    visited[next_node] = True
                    current = next_node
                if prize_total > best_prize_iter:
                    best_prize_iter = prize_total
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            tau = tau * 0.9
            tau += best_prize_iter / 10.0

        return best_prize_global

    # Heuristic generators
    def gen_heuristic_A(lam):
        # Prize_j / (d_ij + lam * d_j0)
        denom = distance_matrix + lam * d_j0[None, :]
        denom = np.maximum(denom, 1e-9)
        eta = prizes[None, :] / denom
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_B(lam):
        # Prize_j / (d_ij * (1 + lam * d_j0))
        denom = distance_matrix * (1.0 + lam * d_j0[None, :])
        denom = np.maximum(denom, 1e-9)
        eta = prizes[None, :] / denom
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_C(lam):
        # Prize_j^lam / (d_ij + d_j0)
        denom = distance_matrix + d_j0[None, :]
        denom = np.maximum(denom, 1e-9)
        prize_pow = np.power(np.maximum(prizes, 1e-9), lam)[None, :]
        eta = prize_pow / denom
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_D(lam):
        # Prize_j / d_ij^lam
        denom = np.power(np.maximum(distance_matrix, 1e-9), lam)
        denom = np.maximum(denom, 1e-9)
        eta = prizes[None, :] / denom
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_E(lam):
        # Prize_j / (d_ij * d_j0^lam)
        denom = distance_matrix * np.power(np.maximum(d_j0, 1e-9)[None, :], lam)
        denom = np.maximum(denom, 1e-9)
        eta = prizes[None, :] / denom
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_F(a, b):
        # Prize_j^a / d_ij^b
        prize_pow = np.power(np.maximum(prizes, 1e-9), a)[None, :]
        denom = np.power(np.maximum(distance_matrix, 1e-9), b)
        denom = np.maximum(denom, 1e-9)
        eta = prize_pow / denom
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_G2(a, b):
        # Prize_j^a / (d_ij^b * d_j0^b)
        prize_pow = np.power(np.maximum(prizes, 1e-9), a)[None, :]
        denom = np.power(np.maximum(distance_matrix, 1e-9), b) * np.power(np.maximum(d_j0, 1e-9)[None, :], b)
        denom = np.maximum(denom, 1e-9)
        eta = prize_pow / denom
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_H(a, b, c):
        # Prize_j^a / (d_ij^b + c*d_j0)
        prize_pow = np.power(np.maximum(prizes, 1e-9), a)[None, :]
        denom = np.power(np.maximum(distance_matrix, 1e-9), b) + c * d_j0[None, :]
        denom = np.maximum(denom, 1e-9)
        eta = prize_pow / denom
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_I(a, b):
        # Prize_j^a / (d_ij + b*d_j0)
        prize_pow = np.power(np.maximum(prizes, 1e-9), a)[None, :]
        denom = distance_matrix + b * d_j0[None, :]
        denom = np.maximum(denom, 1e-9)
        eta = prize_pow / denom
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_J(a, b, c):
        # Prize_j^a / (d_ij^b * d_j0^c)
        prize_pow = np.power(np.maximum(prizes, 1e-9), a)[None, :]
        dij_pow = np.power(np.maximum(distance_matrix, 1e-9), b)
        dj0_pow = np.power(np.maximum(d_j0, 1e-9)[None, :], c)
        eta = prize_pow / (dij_pow * dj0_pow)
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_K(a, b, c):
        # Prize_j^a / (d_ij^b * d_j0^c)  (same form as J but kept for clarity)
        prize_pow = np.power(np.maximum(prizes, 1e-9), a)[None, :]
        dij_pow = np.power(np.maximum(distance_matrix, 1e-9), b)
        dj0_pow = np.power(np.maximum(d_j0, 1e-9)[None, :], c)
        eta = prize_pow / (dij_pow * dj0_pow)
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_L(a, b):
        # Prize_j^a / (d_ij^b + d_j0^b)
        prize_pow = np.power(np.maximum(prizes, 1e-9), a)[None, :]
        denom = np.power(np.maximum(distance_matrix, 1e-9), b) + np.power(np.maximum(d_j0, 1e-9)[None, :], b)
        denom = np.maximum(denom, 1e-9)
        eta = prize_pow / denom
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_M(lam):
        # Detour Penalty: Prize_j / (d_ij * (1 + lam * detour_ij))
        denom = distance_matrix * (1.0 + lam * detour)
        denom = np.maximum(denom, 1e-9)
        eta = prizes[None, :] / denom
        return np.where(eta > 0, eta, 1e-9)

    def gen_heuristic_N(a, b, c):
        # Prize_j^a / (d_ij^b + c * detour)
        prize_pow = np.power(np.maximum(prizes, 1e-9), a)[None, :]
        denom = np.power(np.maximum(distance_matrix, 1e-9), b) + c * detour
        denom = np.maximum(denom, 1e-9)
        eta = prize_pow / denom
        return np.where(eta > 0, eta, 1e-9)

    def get_eta(c_type, p1, p2, p3):
        if c_type == 'A':
            return gen_heuristic_A(p1)
        elif c_type == 'B':
            return gen_heuristic_B(p1)
        elif c_type == 'C':
            return gen_heuristic_C(p1)
        elif c_type == 'D':
            return gen_heuristic_D(p1)
        elif c_type == 'E':
            return gen_heuristic_E(p1)
        elif c_type == 'F':
            return gen_heuristic_F(p1, p2)
        elif c_type == 'G2':
            return gen_heuristic_G2(p1, p2)
        elif c_type == 'H':
            return gen_heuristic_H(p1, p2, p3)
        elif c_type == 'I':
            return gen_heuristic_I(p1, p2)
        elif c_type == 'J':
            return gen_heuristic_J(p1, p2, p3)
        elif c_type == 'K':
            return gen_heuristic_K(p1, p2, p3)
        elif c_type == 'L':
            return gen_heuristic_L(p1, p2)
        elif c_type == 'M':
            return gen_heuristic_M(p1)
        elif c_type == 'N':
            return gen_heuristic_N(p1, p2, p3)
        else:
            return gen_heuristic_A(0.0)

    # Build candidate list: (type, p1, p2, p3)
    candidates = []

    # Type A: Prize_j / (d_ij + lam*d_j0)
    for lam in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
        candidates.append(('A', lam, 0, 0))

    # Type B: Prize_j / (d_ij * (1 + lam*d_j0))
    for lam in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        candidates.append(('B', lam, 0, 0))

    # Type C: Prize_j^lam / (d_ij + d_j0)
    for lam in [0.5, 1.0, 1.5, 2.0, 3.0]:
        candidates.append(('C', lam, 0, 0))

    # Type D: Prize_j / d_ij^lam
    for lam in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        candidates.append(('D', lam, 0, 0))

    # Type E: Prize_j / (d_ij * d_j0^lam)
    for lam in [0.5, 1.0, 1.5, 2.0, 3.0]:
        candidates.append(('E', lam, 0, 0))

    # Type F: Prize_j^a / d_ij^b
    for a in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for b in [0.5, 1.0, 1.5, 2.0, 3.0]:
            candidates.append(('F', a, b, 0))

    # Type G2: Prize_j^a / (d_ij^b * d_j0^b)
    for a in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for b in [0.5, 1.0, 1.5, 2.0, 3.0]:
            candidates.append(('G2', a, b, 0))

    # Type H: Prize_j^a / (d_ij^b + c*d_j0)
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0, 2.0, 5.0]:
                candidates.append(('H', a, b, c))

    # Type I: Prize_j^a / (d_ij + b*d_j0)
    for a in [0.5, 1.0, 1.5, 2.0]:
        for b in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            candidates.append(('I', a, b, 0))

    # Type J: Prize_j^a / (d_ij^b * d_j0^c)
    for a in [1.0, 1.5, 2.0]:
        for b in [0.5, 1.0, 1.5]:
            for c in [0.5, 1.0, 1.5]:
                candidates.append(('J', a, b, c))

    # Type K: Prize_j^a / (d_ij^b * d_j0^c)
    for a in [0.5, 1.0, 1.5, 2.0]:
        for b in [0.5, 1.0, 1.5, 2.0]:
            for c in [0.5, 1.0, 1.5, 2.0]:
                candidates.append(('K', a, b, c))

    # Type L: Prize_j^a / (d_ij^b + d_j0^b)
    for a in [0.5, 1.0, 1.5, 2.0]:
        for b in [0.5, 1.0, 1.5, 2.0]:
            candidates.append(('L', a, b, 0))

    # Type M: Detour Penalty
    for lam in [0.5, 1.0, 2.0, 5.0, 10.0]:
        candidates.append(('M', lam, 0, 0))

    # Type N: Detour in denominator
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0, 2.0, 5.0]:
                candidates.append(('N', a, b, c))

    # Measure cost of one ACO evaluation (seed 0)
    t0 = time.time()
    run_aco(gen_heuristic_A(0.0), 0)
    cost_per_eval = time.time() - t0
    if cost_per_eval < 0.001:
        cost_per_eval = 0.001

    # Phase 1: Quick screen with seed 0
    quick_results = []
    for c_type, p1, p2, p3 in candidates:
        if time.time() - start_time > time_limit - 60:
            break
        eta = get_eta(c_type, p1, p2, p3)
        s = run_aco(eta, 0)
        quick_results.append((s, c_type, p1, p2, p3, eta))

    quick_results.sort(key=lambda x: x[0], reverse=True)

    # Phase 2: Evaluate top candidates with multiple seeds
    # Dynamically choose top_k based on remaining time
    remaining_time = time_limit - (time.time() - start_time)
    # Each candidate in Phase 2 costs ~4 evaluations (seeds 0,1,2,3)
    evals_per_candidate = 4
    max_candidates_phase2 = max(1, int((remaining_time - 30) / (cost_per_eval * evals_per_candidate)))
    top_k = min(max_candidates_phase2, len(quick_results))

    best_score = -1
    best_eta = quick_results[0][4] if quick_results else gen_heuristic_A(0.0)
    best_params = (quick_results[0][1], quick_results[0][2], quick_results[0][3], quick_results[0][4]) if quick_results else ('A', 0.0, 0, 0)

    for s_quick, c_type, p1, p2, p3, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 30:
            break
        scores = [s_quick]  # seed 0 already computed
        for seed in [1, 2, 3]:
            if time.time() - start_time > time_limit - 25:
                break
            s = run_aco(eta, seed)
            scores.append(s)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_eta = eta
            best_params = (c_type, p1, p2, p3)

    # Phase 3: Local refinement around best candidate
    if time.time() - start_time < time_limit - 40:
        best_type = best_params[0]
        best_p1 = best_params[1]
        best_p2 = best_params[2]
        best_p3 = best_params[3]

        param_ranges = {
            'A': [(0, 50), (0, 0), (0, 0)],
            'B': [(0, 20), (0, 0), (0, 0)],
            'C': [(0.25, 5), (0, 0), (0, 0)],
            'D': [(0.25, 5), (0, 0), (0, 0)],
            'E': [(0.25, 5), (0, 0), (0, 0)],
            'F': [(0.25, 5), (0.25, 5), (0, 0)],
            'G2': [(0.25, 5), (0.25, 5), (0, 0)],
            'H': [(0.5, 3), (0.5, 3), (0.25, 10)],
            'I': [(0.5, 3), (0.25, 10), (0, 0)],
            'J': [(0.5, 3), (0.25, 3), (0.25, 3)],
            'K': [(0.5, 3), (0.25, 3), (0.25, 3)],
            'L': [(0.5, 3), (0.25, 3), (0, 0)],
            'M': [(0.1, 20), (0, 0), (0, 0)],
            'N': [(0.5, 3), (0.5, 3), (0.25, 10)],
        }

        ranges = param_ranges.get(best_type, [(0, 10), (0, 10), (0, 10)])

        # Finer local search
        dp_steps = [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5]
        refinement_candidates = []
        for dp1 in dp_steps:
            for dp2 in dp_steps:
                for dp3 in dp_steps:
                    np1 = best_p1 + dp1
                    np2 = best_p2 + dp2
                    np3 = best_p3 + dp3
                    if (np1 >= ranges[0][0] and np1 <= ranges[0][1] and
                        np2 >= ranges[1][0] and np2 <= ranges[1][1] and
                        np3 >= ranges[2][0] and np3 <= ranges[2][1]):
                        if (dp1 != 0 or dp2 != 0 or dp3 != 0):
                            refinement_candidates.append((best_type, np1, np2, np3))

        # Also try other types at best params
        other_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G2', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        for ot in other_types:
            if ot != best_type:
                if ot in ['A', 'B', 'C', 'D', 'E', 'M']:
                    refinement_candidates.append((ot, best_p1, 0, 0))
                elif ot in ['F', 'G2', 'I', 'L']:
                    refinement_candidates.append((ot, best_p1, best_p2, 0))
                elif ot in ['H', 'J', 'K', 'N']:
                    refinement_candidates.append((ot, best_p1, best_p2, best_p3))

        # Deduplicate
        seen = set()
        unique_refinement = []
        for rc in refinement_candidates:
            key = (rc[0], round(rc[1], 3), round(rc[2], 3), round(rc[3], 3))
            if key not in seen:
                seen.add(key)
                unique_refinement.append(rc)

        # Quick screen refinement candidates
        refinement_scores = []
        for c_type, p1, p2, p3 in unique_refinement:
            if time.time() - start_time > time_limit - 30:
                break
            eta = get_eta(c_type, p1, p2, p3)
            s = run_aco(eta, 0)
            refinement_scores.append((s, c_type, p1, p2, p3, eta))

        refinement_scores.sort(key=lambda x: x[0], reverse=True)

        # Evaluate top refinement candidates with more seeds
        remaining_time = time_limit - (time.time() - start_time)
        max_refine_evals = max(1, int((remaining_time - 20) / (cost_per_eval * 4)))
        top_ref = min(max_refine_evals, len(refinement_scores))

        for s_quick, c_type, p1, p2, p3, eta in refinement_scores[:top_ref]:
            if time.time() - start_time > time_limit - 20:
                break
            if s_quick <= best_score * 0.95:
                continue
            scores = [s_quick]
            for seed in [1, 2, 3]:
                if time.time() - start_time > time_limit - 15:
                    break
                s2 = run_aco(eta, seed)
                scores.append(s2)
            avg = np.mean(scores)
            if avg > best_score:
                best_score = avg
                best_eta = eta
                best_params = (c_type, p1, p2, p3)

    # Phase 4: Final polish - blend with top candidates
    if time.time() - start_time < time_limit - 15:
        top_candidates = quick_results[:3]
        for i in range(min(3, len(top_candidates))):
            for j in range(i + 1, min(3, len(top_candidates))):
                if time.time() - start_time > time_limit - 10:
                    break
                eta1 = top_candidates[i][4]
                eta2 = top_candidates[j][4]
                for blend_w in [0.3, 0.5, 0.7]:
                    if time.time() - start_time > time_limit - 8:
                        break
                    eta_blend = np.sqrt(eta1 * eta2)
                    eta_blend = np.where(eta_blend > 0, eta_blend, 1e-9)
                    s = run_aco(eta_blend, 0)
                    if s > best_score * 0.97:
                        scores = [s]
                        for seed in [1, 2]:
                            if time.time() - start_time > time_limit - 5:
                                break
                            s2 = run_aco(eta_blend, seed)
                            scores.append(s2)
                        avg = np.mean(scores)
                        if avg > best_score:
                            best_score = avg
                            best_eta = eta_blend

    # Final safety check
    if not np.all(np.isfinite(best_eta)) or best_eta.shape != distance_matrix.shape:
        best_eta = (prizes[None, :] + 1e-9) / (distance_matrix + 1e-9)
        best_eta = np.where(best_eta > 0, best_eta, 1e-9)

    return best_eta

# EVOLVE-BLOCK-END
