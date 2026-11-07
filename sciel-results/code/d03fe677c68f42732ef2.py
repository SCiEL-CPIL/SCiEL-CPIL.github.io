"""Rewritten program: vectorized ACO + broader heuristic search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 315.0  # explicit deadline guard (evaluator kills at 330s)
    N = len(prizes)

    # Precompute the return-to-start distance and the eligibility matrix.
    d_j0 = distance_matrix[:, 0]
    # eligible[i, j]: can move i -> j and still return to 0 within max_len.
    # (path_len + d[i,j] + d[j,0] <= max_len  <=>  d[i,j] + d[j,0] <= max_len - path_len)
    eligible = (distance_matrix + d_j0[None, :]) <= (max_len + 1e-9)
    # No self moves.
    eligible[range(N), range(N)] = False

    # Fast, deterministic ACO simulator (vectorized).
    def run_aco(eta, seed):
        rng = np.random.RandomState(seed)
        tau = np.ones((N, N))
        best_prize_global = 0.0
        for _it in range(30):
            best_prize_iter = 0.0
            for _ant in range(20):
                cur = 0
                path_len = 0.0
                prize_total = 0.0
                visited = np.zeros(N, dtype=bool)
                visited[0] = True
                while True:
                    rem = max_len - path_len
                    elig = eligible[cur, :] & (~visited) & (distance_matrix[cur, :] + d_j0 <= rem + 1e-9)
                    if not np.any(elig):
                        break
                    w = tau[cur, :] * eta[cur, :]
                    w = np.where(elig, w, 0.0)
                    total = w.sum()
                    if total <= 0.0:
                        break
                    r = rng.rand() * total
                    # cumulative sum over eligible nodes
                    cs = np.cumsum(w)
                    idx = int(np.searchsorted(cs, r, side='right'))
                    if idx >= N:
                        idx = N - 1
                    nxt = idx
                    path_len += distance_matrix[cur, nxt]
                    prize_total += prizes[nxt]
                    visited[nxt] = True
                    cur = nxt
                if prize_total > best_prize_iter:
                    best_prize_iter = prize_total
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            tau = tau * 0.9
            tau += best_prize_iter / 10.0
        return best_prize_global

    # Heuristic families (eta[i, j] = attractiveness of moving i -> j).
    # All depend only on j (prize_j, d_j0) and the transition distance d_ij.
    def build(kind, a, b):
        d = distance_matrix
        if kind == 'A':
            denom = d + b * d_j0[None, :]
        elif kind == 'B':
            denom = d * (1.0 + b * d_j0[None, :])
        elif kind == 'C':
            denom = (d ** a) + b * d_j0[None, :]
        elif kind == 'D':
            denom = d + b * (d_j0[None, :] ** a)
        elif kind == 'E':
            # prize-weighted, penalize long remaining path
            denom = (d + b * d_j0[None, :]) ** a
        elif kind == 'F':
            # emphasize prize, mild distance penalty
            denom = d ** a + b
        elif kind == 'G':
            # prize over (distance + return) but prize to a power
            denom = d + b * d_j0[None, :]
            eta = (prizes[None, :] ** a) / denom
            eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
            return eta
        else:
            denom = d + 1e-9
        denom = np.where(denom > 1e-9, denom, 1e-9)
        eta = prizes[None, :] / denom
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    # Candidate list (kind, a, b).
    candidates = []
    for b in np.linspace(0.0, 6.0, 13):
        candidates.append(('A', 1.0, b))
    for b in np.linspace(0.0, 6.0, 13):
        candidates.append(('B', 1.0, b))
    for a in [0.5, 1.0, 1.5, 2.0]:
        for b in np.linspace(0.0, 4.0, 7):
            candidates.append(('C', a, b))
    for a in [0.5, 1.0, 2.0]:
        for b in np.linspace(0.0, 4.0, 7):
            candidates.append(('D', a, b))
    for a in [1.0, 1.5, 2.0]:
        for b in np.linspace(0.0, 6.0, 11):
            candidates.append(('E', a, b))
    for a in [0.5, 1.0, 1.5]:
        for b in np.linspace(0.0, 1.0, 5):
            candidates.append(('F', a, b))
    for a in [0.5, 1.0, 1.5, 2.0]:
        for b in np.linspace(0.0, 6.0, 11):
            candidates.append(('G', a, b))

    best_score = -1.0
    best_eta = build('A', 1.0, 0.0)  # safe default (prize / distance)
    for kind, a, b in candidates:
        if time.time() - start_time > time_limit - 15.0:
            break
        eta = build(kind, a, b)
        if not np.all(np.isfinite(eta)) or eta.shape != distance_matrix.shape:
            continue
        s = run_aco(eta, 0)
        if s > best_score:
            best_score = s
            best_eta = eta

    # Final safety: finite + correct shape.
    if not np.all(np.isfinite(best_eta)) or best_eta.shape != distance_matrix.shape:
        best_eta = (prizes[None, :] + 1e-9) / (distance_matrix + 1e-9)
        best_eta = np.where(np.isfinite(best_eta) & (best_eta > 0), best_eta, 1e-9)

    return best_eta

# EVOLVE-BLOCK-END
