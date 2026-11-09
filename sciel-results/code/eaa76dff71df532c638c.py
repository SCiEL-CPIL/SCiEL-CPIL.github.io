"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

CAP = 25.0
N_ANTS = 20
N_ITER = 30
RHO = 0.9
Q = 10.0
EPS = 1e-9


def _run_colony(prizes, weights, heuristic, seed):
    """Run the fixed ACO deterministically for one instance. Returns best prize found."""
    m, n = weights.shape
    h = np.asarray(heuristic, dtype=float)
    h = np.where(np.isfinite(h) & (h > 0), h, EPS)
    pher = np.full(n, 1.0, dtype=float)
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(N_ITER):
        iter_best = 0.0
        for _ in range(N_ANTS):
            used = np.zeros(m, dtype=float)
            total = 0.0
            avail = np.ones(n, dtype=bool)
            while True:
                for c in range(m):
                    avail &= (used + weights[c]) <= CAP + 1e-9
                if not avail.any():
                    break
                idx = np.where(avail)[0]
                p = pher[idx] * h[idx]
                s = p.sum()
                if s <= 0:
                    break
                k = rng.choice(idx, p=p / s)
                avail[k] = False
                used += weights[:, k]
                total += prizes[:, k].sum()
            if total > iter_best:
                iter_best = total
        if iter_best > best:
            best = iter_best
        pher = pher * RHO + iter_best / Q
    return best


def _score_heuristic(prizes, weights, heuristic, seeds):
    return float(np.mean([_run_colony(prizes, weights, heuristic, s) for s in seeds]))


def _candidates(prizes, weights):
    m, n = weights.shape
    P = prizes.T            # (n, m)
    W = weights.T           # (n, m)
    prize_sum = P.sum(axis=1)
    wsum = W.sum(axis=1)
    cands = []
    base = prize_sum / (wsum + EPS)
    cands.append(base)
    cands.append(prize_sum)
    cands.append(prize_sum / (wsum + 1.0))
    cands.append(prize_sum / (wsum + 0.1))
    cands.append(base ** 1.5)
    cands.append(base ** 2.0)
    cands.append(prize_sum / (wsum + EPS) / (wsum + EPS))
    # per-constraint normalized efficiency, aggregated
    for c in range(m):
        eff = P[:, c] / (W[:, c] + EPS)
        cands.append(eff.sum(axis=1))
        cands.append(eff.mean(axis=1))
        cands.append(np.min(eff, axis=1))
    # log form
    cands.append(np.log1p(prize_sum) / (wsum + EPS))
    # rank-based
    order = np.argsort(np.argsort(base))
    cands.append(1.0 + order / n)
    cands.append(np.exp(order / n))
    # combinations
    cands.append(base * (1.0 + prize_sum / (prize_sum.max() + EPS)))
    cands.append(base / (1.0 + wsum / (wsum.max() + EPS)))
    return cands


def heuristics(prizes, weights):
    start = time.time()
    DEADLINE = 300.0
    DEADLINE_GUARD = 320.0
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m, n = weights.shape
    seeds = [0, 1, 2]

    cands = _candidates(prizes, weights)

    best_h = cands[0]
    best_score = -np.inf
    for h in cands:
        if time.time() > DEADLINE:
            break
        try:
            sc = _score_heuristic(prizes, weights, h, seeds)
        except Exception:
            continue
        if sc > best_score:
            best_score = sc
            best_h = h

    # safety margin: ensure we return well before the kill
    if time.time() > DEADLINE_GUARD:
        out = np.asarray(best_h, dtype=float)
        out = np.where(np.isfinite(out) & (out > 0), out, EPS)
        return out

    out = np.asarray(best_h, dtype=float)
    out = np.where(np.isfinite(out) & (out > 0), out, EPS)
    return out
# EVOLVE-BLOCK-END
