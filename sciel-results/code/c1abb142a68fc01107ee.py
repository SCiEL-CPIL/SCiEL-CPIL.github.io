"""Heuristic matrix for ant colony optimization of orienteering - structural redesign."""
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 315.0  # Safety margin before 330s kill

    N = len(prizes)
    d_j0 = distance_matrix[:, 0].copy()

    def run_aco_fast(eta, seed):
        """Run the fixed ACO protocol and return best prize over all iterations."""
        np.random.seed(seed)
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
                    costs = distance_matrix[current, :] + d_j0
                    eligible_mask = (~visited) & (costs <= remaining)

                    if not np.any(eligible_mask):
                        break

                    eligible_idx = np.where(eligible_mask)[0]
                    weights = tau[current, eligible_idx] * eta[current, eligible_idx]
                    total_w = weights.sum()
                    if total_w <= 0:
                        break

                    probs = weights / total_w
                    r = np.random.rand()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible_idx):
                        idx = len(eligible_idx) - 1

                    next_node = eligible_idx[idx]
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

    def make_heuristic(prizes, distance_matrix, max_len, mode='potential'):
        """
        Generate heuristic matrix using different structural approaches.
        
        Modes:
        - 'potential': eta[i,j] = prize[j] * (1 - (d_ij + d_j0)/max_len)^alpha
          This models the remaining budget as a multiplier on prize value.
        - 'marginal': eta[i,j] = prize[j] / (1 + d_ij + d_j0)^alpha
          This penalizes cost additively in the denominator.
        - 'log': eta[i,j] = log(1 + prize[j]) / (d_ij + d_j0 + eps)^alpha
          Log-transform prize to compress range.
        - 'rank': Use rank of prize/cost ratio.
        - 'composite': Combination of multiple factors.
        """
        eps = 1e-9
        p = np.maximum(prizes, eps)[None, :]
        dij = np.maximum(distance_matrix, eps)
        dj0 = np.maximum(d_j0, eps)[None, :]
        
        cost = dij + dj0
        
        if mode == 'potential':
            alpha = 1.0
            remaining_frac = np.maximum(1.0 - cost / max_len, 0.0)
            eta = p * np.power(remaining_frac, alpha)
            
        elif mode == 'marginal':
            alpha = 1.0
            eta = p / np.power(1.0 + cost, alpha)
            
        elif mode == 'log':
            alpha = 1.0
            eta = np.log1p(p) / np.power(cost + eps, alpha)
            
        elif mode == 'rank':
            # Compute prize/cost ratio for each node, then rank
            ratios = np.maximum(prizes, eps) / (d_j0 + eps)
            sorted_idx = np.argsort(ratios)
            ranks = np.zeros(N)
            for rank, idx in enumerate(sorted_idx):
                ranks[idx] = rank + 1
            eta = ranks[None, :] / (cost + eps)
            
        elif mode == 'composite':
            # Combine multiple signals
            remaining_frac = np.maximum(1.0 - cost / max_len, 0.0)
            prize_term = np.power(p, 0.8)
            cost_term = 1.0 / np.power(cost + eps, 0.5)
            slack_term = np.power(remaining_frac, 1.0)
            eta = prize_term * cost_term * slack_term
            
        else:
            # Fallback
            eta = p / (cost + eps)
        
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def make_parameterized_heuristic(prizes, distance_matrix, max_len, alpha, beta, gamma, delta):
        """
        More flexible parameterized heuristic:
        eta[i,j] = prize[j]^alpha * (1 - cost/max_len)^beta / (1 + cost)^gamma * slack^delta
        
        where slack = 1 - cost/max_len
        """
        eps = 1e-9
        p = np.maximum(prizes, eps)[None, :]
        dij = np.maximum(distance_matrix, eps)
        dj0 = np.maximum(d_j0, eps)[None, :]
        
        cost = dij + dj0
        slack = np.maximum(1.0 - cost / max_len, 0.0)
        
        prize_term = np.power(p, alpha)
        slack_term = np.power(slack, beta)
        cost_penalty = np.power(1.0 + cost, gamma)
        extra_slack = np.power(slack, delta)
        
        eta = prize_term * slack_term * extra_slack / cost_penalty
        
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    # Generate diverse candidate heuristics
    candidates = []
    
    # Fixed-mode heuristics
    for mode in ['potential', 'marginal', 'log', 'rank', 'composite']:
        candidates.append(('mode', mode, 0, 0, 0, 0))
    
    # Parameterized heuristics with various parameter settings
    param_sets = [
        (1.0, 1.0, 1.0, 0.0),
        (1.0, 1.5, 0.5, 0.0),
        (0.5, 1.0, 1.0, 0.0),
        (1.0, 0.5, 1.0, 0.5),
        (1.5, 1.0, 0.5, 0.0),
        (1.0, 2.0, 0.5, 0.0),
        (0.8, 1.2, 0.8, 0.3),
        (1.2, 0.8, 1.2, 0.2),
        (1.0, 1.0, 0.3, 0.7),
        (0.7, 1.5, 0.7, 0.5),
        (1.3, 1.0, 0.7, 0.3),
        (1.0, 1.0, 1.0, 1.0),
        (1.5, 1.5, 0.5, 0.5),
        (0.5, 0.5, 1.5, 0.5),
        (2.0, 1.0, 0.5, 0.0),
    ]
    for a, b, g, d in param_sets:
        candidates.append(('param', None, a, b, g, d))

    # Phase 1: Quick screening with seed 0
    quick_results = []
    for cand in candidates:
        if time.time() - start_time > time_limit - 45:
            break
        
        ctype = cand[0]
        if ctype == 'mode':
            mode = cand[1]
            eta = make_heuristic(prizes, distance_matrix, max_len, mode=mode)
        else:
            _, _, a, b, g, d = cand
            eta = make_parameterized_heuristic(prizes, distance_matrix, max_len, a, b, g, d)
        
        s = run_aco_fast(eta, 0)
        quick_results.append((s, cand, eta))

    if not quick_results:
        eta = make_heuristic(prizes, distance_matrix, max_len, mode='potential')
        return eta

    quick_results.sort(key=lambda x: x[0], reverse=True)

    # Phase 2: Evaluate top candidates with additional seeds
    top_k = min(8, len(quick_results))
    best_avg = -1
    best_eta = quick_results[0][2]

    for s_quick, cand, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 20:
            break
        scores = [s_quick]
        for seed in [1, 2]:
            if time.time() - start_time > time_limit - 15:
                break
            s = run_aco_fast(eta, seed)
            scores.append(s)
        avg = np.mean(scores)
        if avg > best_avg:
            best_avg = avg
            best_eta = eta

    # Phase 3: Local refinement around best parameterized candidate
    if time.time() - start_time < time_limit - 30:
        best_entry = quick_results[0]
        if best_entry[1][0] == 'param':
            _, _, ba, bb, bg, bd = best_entry[1]
            
            # Generate nearby parameter combinations
            nearby = []
            for da in [-0.2, -0.1, 0.1, 0.2]:
                for db in [-0.2, -0.1, 0.1, 0.2]:
                    for dg in [-0.2, -0.1, 0.1, 0.2]:
                        for dd in [-0.1, 0.0, 0.1]:
                            na = ba + da
                            nb = bb + db
                            ng = bg + dg
                            nd = bd + dd
                            if na > 0.1 and nb > 0.0 and ng > 0.0 and nd > -0.1:
                                nearby.append(('param', None, na, nb, ng, nd))
            
            seen = set()
            for cand in nearby:
                key = tuple(round(x, 3) for x in cand[2:])
                if key in seen:
                    continue
                seen.add(key)
                
                if time.time() - start_time > time_limit - 10:
                    break
                
                _, _, a, b, g, d = cand
                eta = make_parameterized_heuristic(prizes, distance_matrix, max_len, a, b, g, d)
                
                s = run_aco_fast(eta, 0)
                if s > best_avg * 0.95:
                    scores = [s]
                    for seed in [1, 2]:
                        if time.time() - start_time > time_limit - 8:
                            break
                        s2 = run_aco_fast(eta, seed)
                        scores.append(s2)
                    avg = np.mean(scores)
                    if avg > best_avg:
                        best_avg = avg
                        best_eta = eta

    # Final safety check
    if not np.all(np.isfinite(best_eta)) or best_eta.shape != distance_matrix.shape:
        best_eta = make_heuristic(prizes, distance_matrix, max_len, mode='potential')

    return best_eta
