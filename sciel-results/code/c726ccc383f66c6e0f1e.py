import numpy as np
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 120.0
SAFETY_MARGIN = 8.0


def _compute_ratio(h):
    """Scale-invariant ratio after L1 normalization: ||h||^2_2 / ||h*h||_inf."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    s = float(np.sum(h))
    if s == 0.0:
        return 0.0
    hn = h / s
    l2_sq = float(np.sum(hn ** 2))
    fft_h = np.fft.fft(hn, n=2 * N)
    autocorr = np.fft.ifft(fft_h * np.conj(fft_h)).real
    sup_norm = float(np.max(np.abs(autocorr)))
    if sup_norm == 0.0:
        return 0.0
    return l2_sq / sup_norm


def _sidon_baseline():
    used_diffs = set()
    S = []
    for x in range(N):
        ok = True
        for y in S:
            if abs(x - y) in used_diffs:
                ok = False
                break
        if ok:
            S.append(x)
            for y in S:
                used_diffs.add(abs(x - y))
    h = np.zeros(N)
    m = len(S)
    for idx in S:
        h[idx] = 1.0 / m
    return h


def _random_initial():
    return np.random.dirichlet(np.ones(N))


def _optimize_from(h0, deadline):
    def neg_obj(x):
        return -_compute_ratio(np.maximum(x, 0.0))

    options = {"maxiter": 300, "xatol": 1e-7, "fatol": 1e-10, "adaptive": True}
    try:
        res = minimize(neg_obj, h0.copy(), method="Nelder-Mead", options=options)
        h_best = np.maximum(res.x, 0.0)
        best = -res.fun
    except Exception:
        h_best = h0.copy()
        best = _compute_ratio(h0)

    # Coordinate refinement
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for i in range(N):
            if time.time() >= deadline:
                break
            for scale in (0.5, 1.5, 2.0, 0.0):
                trial = h_best.copy()
                trial[i] = h_best[i] * scale
                v = _compute_ratio(trial)
                if v > best + 1e-12:
                    best = v
                    h_best = trial
                    improved = True
    return h_best, best


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    np.random.seed(12345)

    best_h = None
    best_raw = 0.0

    def consider(h, raw):
        nonlocal best_h, best_raw
        if raw > best_raw:
            best_raw = raw
            best_h = np.asarray(h, dtype=float).copy()

    h_sidon = _sidon_baseline()
    consider(h_sidon, _compute_ratio(h_sidon))

    for r in range(25):
        if time.time() >= deadline:
            break
        if r % 3 == 0:
            h0 = _sidon_baseline().copy() + 0.1 * np.random.randn(N)
        else:
            h0 = _random_initial()
        h_opt, _ = _optimize_from(h0, deadline)
        consider(h_opt, _compute_ratio(h_opt))

    if best_h is None:
        return list(h_sidon)
    return list(np.maximum(best_h, 0.0))
