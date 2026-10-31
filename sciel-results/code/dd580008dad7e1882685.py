"""Rewritten: direct optimization of squared max/min pairwise-distance ratio."""
import time
import numpy as np

N = 16
D = 2
_TIME_LIMIT = 150.0
_DEADLINE = time.time() + _TIME_LIMIT - 10.0  # safety margin


def _pairwise_sqdist(P):
    # P shape (N, D) -> (N, N) squared distances
    sq = np.sum(P * P, axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * (P @ P.T), 0.0)


def _score(P, tau=1.0):
    D2 = _pairwise_sqdist(P)
    iu = np.triu_indices(N, k=1)
    d2 = D2[iu]
    d2 = np.maximum(d2, 1e-20)
    d2s = d2 / tau
    smin = np.logaddexp.reduce(d2s) - np.log(len(d2s))   # soft-min
    smax = np.logaddexp.reduce(-d2s) * -1 - np.log(len(d2s))  # soft-max
    return np.exp(smax - smin)


def _score_exact(P):
    D2 = _pairwise_sqdist(P)
    iu = np.triu_indices(N, k=1)
    d2 = D2[iu]
    return d2.max() / max(d2.min(), 1e-30)


def _gradient(P, tau=1.0):
    # grad of log(softmax - softmin) w.r.t. d2, then chain to P
    D2 = _pairwise_sqdist(P)
    iu = np.triu_indices(N, k=1)
    d2 = D2[iu]
    d2 = np.maximum(d2, 1e-20)
    d2s = d2 / tau
    logZmin = np.logaddexp.reduce(d2s)
    logZmax = np.logaddexp.reduce(-d2s)
    pmin = np.exp(d2s - logZmin)
    pmax = np.exp(-d2s - logZmax)
    # d log(ratio)/d d2 = -pmax/tau + pmin/tau
    g_d2 = (pmin - pmax) / tau
    # backprop g_d2 (m,) to P: d2_ij = |P_i-P_j|^2, d d2_ij/dP_k = 2(P_k - P_j) if k==i, 2(P_k-P_i) if k==j
    grad = np.zeros_like(P)
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        g = g_d2[idx]
        grad[i] += 2.0 * g * (P[i] - P[j])
        grad[j] += 2.0 * g * (P[j] - P[i])
    return grad


def construct():
    global _DEADLINE
    t0 = time.time()

    # Initial: 4x4 grid centered
    g = np.array(np.meshgrid(np.arange(4), np.arange(4)), dtype=float).T.reshape(-1, 2)
    g = g - g.mean(axis=0)
    # scale so min distance is ~1
    best = g.copy()
    best_score = _score_exact(best)

    rng = np.random.default_rng(12345)

    def try_optimize(start, lr0, max_iters):
        nonlocal best, best_score
        P = start.copy()
        lr = lr0
        for it in range(max_iters):
            if time.time() > _DEADLINE:
                break
            g = _gradient(P, tau=1.0)
            P = P - lr * g
            P = P - P.mean(axis=0)  # center
            s = _score_exact(P)
            if s < best_score:
                best_score = s
                best = P.copy()
            # adapt lr
            if it % 50 == 0 and it > 0:
                lr *= 0.9
        return best, best_score

    # Main optimization from grid
    try_optimize(g, 0.01, 3000)

    # Try a few random restarts with different scaling
    for scale in [0.7, 1.0, 1.4]:
        if time.time() > _DEADLINE:
            break
        start = g * scale + rng.normal(0, 0.05, g.shape)
        try_optimize(start, 0.008, 1500)

    # Final polish: tighten with smaller tau and more iters
    if time.time() < _DEADLINE:
        P = best.copy()
        lr = 0.002
        for it in range(2000):
            if time.time() > _DEADLINE:
                break
            g = _gradient(P, tau=0.5)
            P = P - lr * g
            P = P - P.mean(axis=0)
            s = _score_exact(P)
            if s < best_score:
                best_score = s
                best = P.copy()

    return best
