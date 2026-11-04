"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

# ---------------------------------------------------------------------------
# Fast overlap machinery (O(N log N) via FFT)
# ---------------------------------------------------------------------------
_M = 1
while _M < 2 * N:
    _M *= 2
_P = _M * 2
_W = np.fft.fft(np.zeros(_P))  # precomputed FFT of the zero vector
_IDX = np.arange(1, N)  # non-zero shift indices


def _overlaps(h):
    """Return the 2*(N-1) linear cross-correlation values of h with (1-h)
    at every non-zero shift.  C_k = sum_i h_i * (1 - h_{i+k}).
    Computed with a single FFT per call (O(N log N))."""
    g = 1.0 - h
    f = np.fft.ifft(np.fft.fft(g, _P) * _W)
    return f[_IDX].real, f[_P - _IDX].real


def _max_overlap(h):
    """Maximum linear overlap value."""
    a, b = _overlaps(h)
    return float(max(np.max(a), np.max(b)))


def _project_to_feasible(h):
    """Project h onto {0<=h_i<=1, sum(h)=N/2}.  Bisection on a uniform shift."""
    h = np.clip(h, 0.0, 1.0).copy()
    target = TARGET_SUM
    current = np.sum(h)
    if abs(current - target) < 1e-14:
        return h
    lo, hi = -1.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        s = np.sum(np.clip(h + mid, 0.0, 1.0))
        if s < target:
            lo = mid
        else:
            hi = mid
        if abs(s - target) < 1e-14:
            break
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h


def _objective_epigraph(x, beta, lam_sum):
    """Epigraph surrogate: minimise t + [lse(overlaps)-t]_+^2 + lam*(sum-N/2)^2."""
    h = np.clip(x[:N], 0.0, 1.0)
    t = x[N]
    a, b = _overlaps(h)
    ov = np.concatenate((a, b))
    max_ov = np.max(ov)
    if max_ov > 1000.0:
        max_ov = 1000.0
    shifted = np.clip(ov - max_ov, -50.0 / beta, 0.0)
    lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    penalty = 0.0
    if lse > t:
        penalty = (lse - t) ** 2
    return t + penalty + lam_sum * (np.sum(h) - TARGET_SUM) ** 2


def _pairwise_polish(h, deadline):
    """Pairwise coordinate descent that keeps sum(h) exactly constant
    (move delta: h_i += delta, h_j -= delta).  Uses the fast overlap and a
    geometric step schedule with per-coordinate best-step reuse."""
    h = h.copy()
    best = _max_overlap(h)
    best_delta = np.zeros(N)
    steps = [0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125,
             0.000390625, 0.0001953125, 0.00009765625, 0.000048828125]
    order = np.arange(N)
    while time.time() < deadline:
        np.random.shuffle(order)
        improved = False
        for i in order:
            if time.time() >= deadline:
                break
            base = h[i]
            cur_best = best
            cur_best_d = best_delta[i]
            for j in range(N):
                if j == i:
                    continue
                basej = h[j]
                for d in steps:
                    if base + d > 1.0 or basej - d < 0.0:
                        continue
                    h[i] = base + d
                    h[j] = basej - d
                    s = _max_overlap(h)
                    if s < cur_best - 1e-12:
                        cur_best = s
                        cur_best_d = d
                        improved = True
                    h[i] = base
                    h[j] = basej
                h[i] = base
                h[j] = basej
            if cur_best < best:
                # re-apply the best move for coordinate i
                for j in range(N):
                    if j == i:
                        continue
                    basej = h[j]
                    d = cur_best_d
                    if base + d <= 1.0 and basej - d >= 0.0:
                        h[i] = base + d
                        h[j] = basej - d
                        if _max_overlap(h) < best + 1e-12:
                            best = cur_best
                            best_delta[i] = d
                            break
                        h[i] = base
                        h[j] = basej
        if not improved:
            break
    return h, best


def _run_lbfgs(h0, t0, beta, lam, deadline, maxiter, ftol, gtol):
    """One L-BFGS-B run of the epigraph surrogate; returns improved h or None."""
    bounds = [(0.0, 1.0)] * N + [(0.0, 10.0)]
    x0 = np.concatenate([h0, [t0]])

    def obj(x):
        return _objective_epigraph(x, beta, lam)

    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': 1e-10, 'maxls': 40})
    except Exception:
        return None
    h_new = _project_to_feasible(res.x[:N])
    return h_new


def construct():
    start = time.time()
    deadline = start + 120.0  # hard safety margin before the 150 s kill
    rng = np.random.default_rng(42)

    best_h = None
    best_score = float('inf')

    def consider(h):
        nonlocal best_h, best_score
        s = _max_overlap(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # ---------------- structured + random initial guesses ----------------
    t = np.arange(N)
    inits = []
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]:
        hi = 0.5 + 0.3 * np.sin(2 * np.pi * k * t / N)
        inits.append(_project_to_feasible(hi))
    inits.append(_project_to_feasible(np.full(N, 0.5)))
    for _ in range(6):
        inits.append(_project_to_feasible(rng.uniform(0, 1, N)))
    # alternating / block patterns
    alt = np.zeros(N); alt[::2] = 0.8; alt[1::2] = 0.2
    inits.append(_project_to_feasible(alt))
    blk = np.tile([0.2, 0.8, 0.2, 0.8], N // 4)
    inits.append(_project_to_feasible(blk))

    for hi in inits:
        if time.time() > deadline - 30.0:
            break
        consider(hi)

    # ---------------- Phase 1: coarse-to-fine epigraph L-BFGS ----------------
    beta_schedule = [8, 25, 80, 250, 800, 2500, 8000]
    t_init = best_score + 0.1
    for beta in beta_schedule:
        if time.time() > deadline - 25.0:
            break
        h_new = _run_lbfgs(best_h, t_init, beta, 50.0, deadline, 400, 1e-12)
        if h_new is not None:
            s = _max_overlap(h_new)
            if s < best_score:
                best_score = s
                best_h = h_new
                t_init = min(t_init, best_score + 0.05)
        else:
            break

    # ---------------- Phase 2: fast pairwise coordinate polish --------------
    if time.time() < deadline - 5.0:
        h_cd, s_cd = _pairwise_polish(best_h, deadline - 3.0)
        if s_cd < best_score:
            best_score = s_cd
            best_h = h_cd

    # ---------------- Phase 3: high-precision epigraph re-polish ------------
    if time.time() < deadline - 8.0:
        h_new = _run_lbfgs(best_h, best_score + 0.02, 20000.0, 5000.0,
                           deadline - 3.0, 1500, 1e-14)
        if h_new is not None:
            s = _max_overlap(h_new)
            if s < best_score:
                best_score = s
                best_h = h_new
        if time.time() < deadline - 3.0:
            h_cd, s_cd = _pairwise_polish(best_h, deadline - 2.0)
            if s_cd < best_score:
                best_score = s_cd
                best_h = h_cd

    # ---------------- final validity / exact-sum correction ----------------
    best_h = _project_to_feasible(best_h)
    diff = TARGET_SUM - np.sum(best_h)
    if abs(diff) > 1e-12:
        if 0.0 <= best_h[0] + diff <= 1.0:
            best_h[0] += diff
        else:
            best_h = np.clip(best_h * (TARGET_SUM / np.sum(best_h)), 0.0, 1.0)
            best_h = _project_to_feasible(best_h)

    return best_h

# EVOLVE-BLOCK-END
