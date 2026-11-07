import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 325.0  # safety margin before the 330s kill

    prizes_arr = np.asarray(prizes, dtype=np.float64)
    dist = np.asarray(distance_matrix, dtype=np.float64)
    N = len(prizes_arr)

    d_j0 = dist[:, 0].copy()
    d_ij_plus_j0 = dist + d_j0[None, :]

    # ------------------------------------------------------------------
    # Fast ACO evaluator (vectorized candidate selection)
    # ------------------------------------------------------------------
    def run_aco_fast(eta, seed=42, n_ants=20, n_iters=30):
        rng = np.random.RandomState(seed)
        tau = np.ones((N, N))
        best_global = 0.0
        for _ in range(n_iters):
            best_iter = 0.0
            for _ in range(n_ants):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = np.zeros(N, dtype=bool)
                visited[0] = True
                while True:
                    remaining = ~visited
                    threshold = max_len - path_len
                    eligible_mask = remaining & (d_ij_plus_j0[current, :] <= threshold + 1e-12)
                    eligible_idx = np.where(eligible_mask)[0]
                    if eligible_idx.size == 0:
                        break
                    weights = tau[current, eligible_idx] * eta[current, eligible_idx]
                    total_w = weights.sum()
                    if total_w <= 0:
                        break
                    probs = weights / total_w
                    r = rng.random_sample()
                    cumsum = np.cumsum(probs)
                    idx = int(np.searchsorted(cumsum, r))
                    if idx >= eligible_idx.size:
                        idx = eligible_idx.size - 1
                    nxt = eligible_idx[idx]
                    path_len += dist[current, nxt]
                    prize_total += prizes_arr[nxt]
                    visited[nxt] = True
                    current = nxt
                if prize_total > best_iter:
                    best_iter = prize_total
            if best_iter > best_global:
                best_global = best_iter
            tau = tau * 0.9
            tau += best_iter / 10.0
        return best_global

    # ------------------------------------------------------------------
    # Heuristic constructors (all clipped to >= 1e-9)
    # ------------------------------------------------------------------
    def _clip(eta):
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def make_alpha(alpha, p, r):
        denom = dist + alpha * d_j0[None, :]
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], r) / np.power(denom, p)
        return _clip(eta)

    def make_sum(p, q, r, s):
        denom = np.power(np.maximum(dist, 1e-12), q) + np.power(np.maximum(d_j0, 1e-12)[None, :], r)
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], p) / np.power(denom, s)
        return _clip(eta)

    def make_mul(p, q, r):
        denom = np.power(np.maximum(dist, 1e-12), p) * np.power(np.maximum(d_j0, 1e-12)[None, :], q)
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], r) / denom
        return _clip(eta)

    def make_log(alpha, p, r):
        denom = np.log1p(np.maximum(dist, 0)) + alpha * np.log1p(np.maximum(d_j0, 0))[None, :]
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], r) / np.power(denom, p)
        return _clip(eta)

    def make_exp(alpha, p, r):
        val = np.maximum(dist + d_j0[None, :], 0)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], r) * np.exp(-alpha * np.power(val, p))
        return _clip(eta)

    def make_norm(alpha, p, r):
        denom = np.power(np.maximum(dist, 1e-12), p) + alpha * np.power(np.maximum(d_j0, 1e-12)[None, :], p)
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], r) / denom
        return _clip(eta)

    # NEW structural family: prize * normalized future-slack (dynamic potential)
    def make_potential(r, beta):
        # slack after visiting j from i, assuming we must still return to 0
        slack = max_len - d_ij_plus_j0
        slack = np.maximum(slack, 0.0)
        slack_norm = slack / max_len
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], r) * np.power(slack_norm, beta)
        return _clip(eta)

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    candidates = []

    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for p in [0.5, 1.0, 1.5, 2.0]:
            for r in [1.0, 1.5, 2.0]:
                candidates.append(('alpha', alpha, p, r))

    for p in [1.0, 1.5, 2.0]:
        for q in [0.5, 1.0, 1.5]:
            for r in [0.5, 1.0, 1.5]:
                for s in [1.0, 1.5]:
                    candidates.append(('sum', p, q, r, s))

    for p in [0.5, 1.0, 1.5]:
        for q in [0.5, 1.0, 1.5]:
            for r in [1.0, 1.5, 2.0]:
                candidates.append(('mul', p, q, r))

    for alpha in [0.5, 1.0, 2.0]:
        for p in [1.0, 1.5]:
            for r in [1.0, 1.5, 2.0]:
                candidates.append(('log', alpha, p, r))

    for alpha in [1.0, 2.0, 5.0]:
        for p in [0.5, 1.0]:
            for r in [1.0, 1.5]:
                candidates.append(('exp', alpha, p, r))

    for alpha in [0.5, 1.0, 2.0]:
        for p in [1.0, 1.5]:
            for r in [1.0, 1.5, 2.0]:
                candidates.append(('norm', alpha, p, r))

    # Dynamic Potential (new structural direction)
    for r in [0.5, 1.0, 1.5, 2.0]:
        for beta in [0.5, 1.0, 1.5, 2.0]:
            candidates.append(('potential', r, beta))

    def get_eta(cand):
        t = cand[0]
        if t == 'alpha':
            return make_alpha(cand[1], cand[2], cand[3])
        if t == 'sum':
            return make_sum(cand[1], cand[2], cand[3], cand[4])
        if t == 'mul':
            return make_mul(cand[1], cand[2], cand[3])
        if t == 'log':
            return make_log(cand[1], cand[2], cand[3])
        if t == 'exp':
            return make_exp(cand[1], cand[2], cand[3])
        if t == 'norm':
            return make_norm(cand[1], cand[2], cand[3])
        if t == 'potential':
            return make_potential(cand[1], cand[2])
        raise ValueError("unknown candidate type")

    # ------------------------------------------------------------------
    # Coarse grid search with deadline guard
    # ------------------------------------------------------------------
    best_score = 0.0
    best_eta = None
    for cand in candidates:
        if time.time() - start_time > time_limit - 60:
            break
        eta = get_eta(cand)
        score = run_aco_fast(eta)
        if score > best_score:
            best_score = score
            best_eta = eta

    if best_eta is None:
        best_eta = make_alpha(1.0, 1.0, 1.5)
        best_score = run_aco_fast(best_eta)

    # ------------------------------------------------------------------
    # Local refinement around the best candidate (if it is a param family)
    # ------------------------------------------------------------------
    if time.time() - start_time < time_limit - 60:
        # Find the best candidate tuple to refine around
        best_cand = None
        for cand in candidates:
            if time.time() - start_time > time_limit - 60:
                break
            eta = get_eta(cand)
            score = run_aco_fast(eta)
            if score > best_score + 1e-12:
                best_score = score
                best_eta = eta
                best_cand = cand

        if best_cand is not None:
            t = best_cand[0]
            # Build a small local neighbourhood of parameter perturbations
            def _neigh(vals, step):
                out = []
                for i, v in enumerate(vals):
                    lo = max(v - step, 1e-3)
                    hi = v + step
                    for nv in (lo, hi):
                        c2 = list(vals)
                        c2[i] = nv
                        out.append(tuple(c2))
                return out

            # Reconstruct numeric params (excluding type tag)
            num = best_cand[1:]
            if t == 'alpha':
                for c in _neigh(num, 0.25):
                    if time.time() - start_time > time_limit - 30:
                        break
                    eta = make_alpha(*c)
                    score = run_aco_fast(eta)
                    if score > best_score:
                        best_score, best_eta = score, eta
            elif t == 'potential':
                for c in _neigh(num, 0.25):
                    if time.time() - start_time > time_limit - 30:
                        break
                    eta = make_potential(*c)
                    score = run_aco_fast(eta)
                    if score > best_score:
                        best_score, best_eta = score, eta
            elif t == 'mul':
                for c in _neigh(num, 0.25):
                    if time.time() - start_time > time_limit - 30:
                        break
                    eta = make_mul(*c)
                    score = run_aco_fast(eta)
                    if score > best_score:
                        best_score, best_eta = score, eta
            elif t == 'norm':
                for c in _neigh(num, 0.25):
                    if time.time() - start_time > time_limit - 30:
                        break
                    eta = make_norm(*c)
                    score = run_aco_fast(eta)
                    if score > best_score:
                        best_score, best_eta = score, eta
            elif t == 'exp':
                for c in _neigh(num, 0.5):
                    if time.time() - start_time > time_limit - 30:
                        break
                    eta = make_exp(*c)
                    score = run_aco_fast(eta)
                    if score > best_score:
                        best_score, best_eta = score, eta
            elif t == 'log':
                for c in _neigh(num, 0.25):
                    if time.time() - start_time > time_limit - 30:
                        break
                    eta = make_log(*c)
                    score = run_aco_fast(eta)
                    if score > best_score:
                        best_score, best_eta = score, eta
            elif t == 'sum':
                for c in _neigh(num, 0.25):
                    if time.time() - start_time > time_limit - 30:
                        break
                    eta = make_sum(*c)
                    score = run_aco_fast(eta)
                    if score > best_score:
                        best_score, best_eta = score, eta

    # Final safety: ensure finite, positive, correct shape
    best_eta = np.asarray(best_eta, dtype=np.float64)
    if best_eta.shape != (N, N) or not np.all(np.isfinite(best_eta)):
        best_eta = make_alpha(1.0, 1.0, 1.5)
    best_eta = np.where(np.isfinite(best_eta) & (best_eta > 0), best_eta, 1e-9)

    return best_eta
