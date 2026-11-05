"""Third autocorrelation inequality: minimize |2*N*max(conv(h,h))|/sum(h)^2."""
import time
import numpy as np

N = 400
DEADLINE = 132.0  # wall-clock safety margin before the 150s kill

# ---- fixed helper arrays ----
_LAGS = np.arange(-(N - 1), N)
# c_lag = sum_j h[j] h[j+lag];  j in [0, N-1-lag] for lag>=0 else [ -lag, N-1]
_IDX = np.arange(N)
_START = np.where(_LAGS >= 0, 0, -_LAGS)
_LEN = N - np.abs(_LAGS)
_SLICES = [slice(int(s), int(s + ln)) for s, ln in zip(_START, _LEN)]
_LAGS = _LAGS.astype(np.int64)
_LEN = _LEN.astype(np.int64)
_S2 = float(N * (N - 1) * (2 * N - 1) / 6.0)  # sum of squares of 1..N-1 (for normalization)


def _conv(h):
    """All lag autocorrelations via slicing (single pass)."""
    return np.array([np.dot(h[sl], h[sl]) for sl in _SLICES], dtype=np.float64)


def _objective(h):
    s = np.sum(h)
    if s == 0.0:
        return np.inf
    c = _conv(h)
    return abs(2.0 * N * float(np.max(c))) / (s * s)


def _softmax_obj(h, beta):
    s = np.sum(h)
    if s == 0.0:
        return np.inf
    c = _conv(h)
    cm = float(np.max(c))
    # log-sum-exp of c, centered at cm for stability
    z = beta * (c - cm)
    m = float(np.max(z))
    lse = cm + (m + float(np.log(np.sum(np.exp(z - m)))) / beta)
    return abs(2.0 * N * lse) / (s * s)


def _softmax_grad(h, beta):
    s = np.sum(h)
    if s == 0.0:
        return np.zeros(N), np.inf
    c = _conv(h)
    cm = float(np.max(c))
    z = beta * (c - cm)
    m = float(np.max(z))
    w = np.exp(z - m)
    wsum = float(np.sum(w))
    lse = cm + (m + float(np.log(wsum))) / beta
    # d lse / d h[k] = (1/s) * sum_lag w_lag * (h[k+lag] + h[k-lag])
    g = np.zeros(N)
    for lag in range(-(N - 1), N):
        coeff = w[lag + (N - 1)] / (beta * wsum)
        if coeff == 0.0:
            continue
        a = _START[lag + (N - 1)]
        ln = _LEN[lag + (N - 1)]
        j = np.arange(int(a), int(a + ln))
        if lag >= 0:
            g[j] += coeff * h[j + lag]
            g[j + lag] += coeff * h[j]
        else:
            g[j] += coeff * h[j - lag]
            g[j - lag] += coeff * h[j]
    g = g / s
    d_obj_ds = -2.0 * lse / s
    grad = g + d_obj_ds
    return grad, abs(2.0 * N * lse) / (s * s)


def _lbfgs(h0, beta, iters, tol=1e-11):
    h = np.array(h0, dtype=np.float64)
    m = 16
    g = np.zeros(N)
    y = []
    rho = []
    for it in range(iters):
        grad, val = _softmax_grad(h, beta)
        g = grad
        if np.linalg.norm(g) < tol * max(1.0, np.linalg.norm(h)):
            break
        if len(y) > 0:
            svec = np.array(y[-1][0])
            yvec = np.array(y[-1][1])
            sy = float(np.dot(svec, yvec))
            if abs(sy) < 1e-14:
                Hs = svec
            else:
                rho = 1.0 / sy
                v = yvec - rho * float(np.dot(yvec, svec)) * yvec  # not exact; do two-loop below
                # two-loop recursion for Hs
                alpha = rho * float(np.dot(svec, yvec))
                Hs = alpha * svec
                for sv, yv, rv in zip(reversed(y), reversed([None] * len(y)), reversed(rho)):
                    pass
                # simpler: use standard two-loop
                Hs = np.zeros(N)
                for sv, yv, rv in zip(reversed(y), reversed(y), reversed(rho)):
                    pass
                Hs = _hprod(y, rho, svec)
        else:
            Hs = svec
        p = -Hs
        # backtracking line search
        val0 = val
        alpha = 1.0
        for _ in range(40):
            hnew = h + alpha * p
            gnew, valnew = _softmax_grad(hnew, beta)
            if valnew < val0 - 1e-4 * alpha * float(np.dot(g, p)):
                break
            alpha *= 0.5
        else:
            break
        svec = alpha * p
        yvec = gnew - g
        sy = float(np.dot(svec, yvec))
        if sy > 1e-14:
            y.append((svec, yvec))
            rho.append(1.0 / sy)
            if len(y) > m:
                y = y[-m:]
                rho = rho[-m:]
        h = hnew
    return h, _objective(h)


def _hprod(y, rho, s):
    """Two-loop L-BFGS H*s product."""
    q = s.copy()
    alphas = []
    for i in range(len(y) - 1, -1, -1):
        sv, yv = y[i]
        alphas.append(rho[i] * float(np.dot(sv, q)))
        q = q - alphas[-1] * yv
    r = q
    for i in range(len(y)):
        sv, yv = y[i]
        r = r + (alphas[len(y) - 1 - i] - rho[i] * float(np.dot(yv, r))) * sv
    return r


def _spectral_null(k):
    """h[j] = sum of k sines -> vanishes at many lags -> low max autocorrelation."""
    j = np.arange(N, dtype=np.float64)
    h = np.zeros(N)
    for m in range(1, k + 1):
        h += np.sin(2.0 * np.pi * m * j / N)
    if np.sum(h) == 0.0:
        h += 1e-6
    return h


def _random_init(seed):
    rng = np.random.RandomState(seed)
    return rng.randn(N)


def construct():
    t_start = time.time()
    best_h = np.ones(N, dtype=np.float64)
    best_val = _objective(best_h)

    candidates = []
    # spectral-null initial points (vanishing lags)
    for k in (1, 2, 3, 4, 5, 6):
        h = _spectral_null(k)
        v = _objective(h)
        candidates.append((v, h))
    # random inits
    for seed in range(8):
        h = _random_init(seed)
        v = _objective(h)
        candidates.append((v, h))
    candidates.sort(key=lambda x: x[0])

    # coarse optimization with moderate beta
    beta_coarse = 200.0
    for v, h in candidates[:8]:
        if time.time() > DEADLINE - 20.0:
            break
        hopt, vopt = _lbfgs(h, beta_coarse, iters=40)
        if vopt < best_val:
            best_val = vopt
            best_h = hopt
    # fine polish with high beta
    beta_fine = 500.0
    for v, h in candidates[:4]:
        if time.time() > DEADLINE - 10.0:
            break
        hopt, vopt = _lbfgs(h, beta_fine, iters=60)
        if vopt < best_val:
            best_val = vopt
            best_h = hopt
        # also polish the coarse result
        if time.time() > DEADLINE - 8.0:
            break
        hopt2, vopt2 = _lbfgs(best_h, beta_fine, iters=60)
        if vopt2 < best_val:
            best_val = vopt2
            best_h = hopt2

    # final exact-evaluation polish loop
    if time.time() < DEADLINE - 5.0:
        h = np.array(best_h, dtype=np.float64)
        for _ in range(200):
            if time.time() > DEADLINE - 3.0:
                break
            # local steepest-descent on the exact max objective (subgradient)
            c = _conv(h)
            s = np.sum(h)
            if s == 0.0:
                break
            idx = int(np.argmax(c))
            lag = int(_LAGS[idx])
            # subgradient of max(c) w.r.t. h at the active lag
            sg = np.zeros(N)
            a = int(_START[idx])
            ln = int(_LEN[idx])
            j = np.arange(a, a + ln)
            if lag >= 0:
                sg[j] += h[j + lag]
                sg[j + lag] += h[j]
            else:
                sg[j] += h[j - lag]
                sg[j - lag] += h[j]
            # objective f = |2N*max(c)|/s^2 ; assume max>0 (typical)
            f = 2.0 * N * c[idx] / (s * s)
            # gradient of f
            g = (2.0 * N / (s * s)) * sg - 2.0 * f * (2.0 * sg_sum) if False else None
            # recompute properly
            d_f_ds = -2.0 * (2.0 * N * c[idx]) / (s * s)
            grad = (2.0 * N / (s * s)) * sg + d_f_ds * np.ones(N)
            gn = np.linalg.norm(grad)
            if gn < 1e-12:
                break
            p = -grad / gn
            # line search
            improved = False
            alpha = 0.5
            for _ in range(30):
                hnew = h + alpha * p
                vnew = _objective(hnew)
                if vnew < best_val - 1e-12:
                    h = hnew
                    best_val = vnew
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                break

    # guard against zero sum / non-finite
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0.0:
        return np.ones(N)
    return best_h
