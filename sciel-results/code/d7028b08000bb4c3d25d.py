"""Minimize the third autocorrelation inequality constant via projected
gradient descent on a scale-normalized shape with a soft-max surrogate."""
import time
import numpy as np

N = 400
DEADLINE = 120.0          # evaluator wall-clock limit
SAFETY = 8.0              # safety margin (seconds)


def _value(h):
    """Exact objective (lower is better)."""
    s = float(np.sum(h))
    if s == 0.0 or not np.isfinite(s):
        return np.inf
    c = np.convolve(h, h, mode='full')
    return abs(2.0 * N * float(np.max(c)) / (s * s))


def _smooth_value(h, beta):
    """Smooth surrogate: 2N * max_k softplus(beta * c_k) / (sum h)^2.

    softplus(beta*x) -> max(x,0) as beta->inf; the max over lags is recovered
    via log-sum-exp, which is smooth and differentiable.
    """
    c = np.convolve(h, h, mode='full')
    b = beta * c
    m = np.max(b)
    if not np.isfinite(m):
        return np.inf
    z = np.exp(b - m)
    logsum = m + np.log(np.sum(z))
    return logsum / (beta * (float(np.sum(h)) ** 2))


def _smooth_grad(h, beta):
    """Gradient of the smooth surrogate w.r.t. h (sum h fixed implicitly)."""
    c = np.convolve(h, h, mode='full')
    b = beta * c
    m = np.max(b)
    z = np.exp(b - m)
    w = z / np.sum(z)                      # softmax over lags
    # d/dc_k = 2 beta w_k ; dc_k/dh = h convolved with lag-k shift of h
    g = np.zeros(2 * N - 1)
    for k in range(2 * N - 1):
        if w[k] <= 1e-15:
            continue
        if k < N:
            g[k:k + N] += 2.0 * beta * w[k] * h[:N - k]
        else:
            g[k - N:] += 2.0 * beta * w[k] * h[N - k:]
    return g


def _project(h):
    """Project onto sum(h)=1 (removes the sum constraint)."""
    return h - (float(np.sum(h)) - 1.0) / N


def _init():
    """Diverse, sum=1, finite initial shapes."""
    rng = np.random.default_rng(12345)
    inits = [
        np.ones(N),
        np.sin(2.0 * np.pi * np.arange(N) / N),
        np.sin(2.0 * np.pi * np.arange(N) / N) + 0.5 * np.sin(4.0 * np.pi * np.arange(N) / N),
        np.cos(2.0 * np.pi * np.arange(N) / N),
        rng.standard_normal(N),
        rng.standard_normal(N) + 0.5 * np.sin(2.0 * np.pi * np.arange(N) / N),
        np.sign(np.sin(2.0 * np.pi * np.arange(N) / N)) * rng.uniform(0.5, 1.0, N),
        rng.standard_normal(N) + np.linspace(-1, 1, N),
    ]
    out = []
    for v in inits:
        v = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(v)):
            continue
        if abs(float(np.sum(v))) < 1e-8:
            v = v + 0.01 * np.ones(N)
        v = _project(v)
        out.append(v)
    if not out:
        out = [np.ones(N)]
    return out


def _run():
    best = None
    best_v = np.inf
    deadline = time.time() + (DEADLINE - SAFETY)
    inits = _init()
    betas = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0]

    for h0 in inits:
        h = h0.copy()
        # adaptive step, projected gradient on the smooth surrogate
        step = 0.02
        for beta in betas:
            # a few warm steps at this beta
            for it in range(60):
                if time.time() > deadline:
                    break
                g = _smooth_grad(h, beta)
                # projected gradient step
                h_new = _project(h - step * g)
                # backtracking line search on the smooth value
                f_new = _smooth_value(h_new, beta)
                f_old = _smooth_value(h, beta)
                if not np.isfinite(f_new) or f_new > f_old:
                    step *= 0.5
                    continue
                h = h_new
                if step < 0.05:
                    step *= 1.05
            # exact evaluation + polish
            v = _value(h)
            if v < best_v:
                best_v = v
                best = h.copy()
            if time.time() > deadline:
                break

    if best is None:
        best = np.ones(N)
    # final exact polish: a few projected gradient steps on the exact max
    # using the argmax lag (subgradient) to squeeze the peak down.
    h = best.copy()
    for it in range(400):
        if time.time() > deadline:
            break
        c = np.convolve(h, h, mode='full')
        k = int(np.argmax(c))
        if k < N:
            g = np.zeros(N)
            g[:N - k] += h[:N - k]
            g[N - k:] += h[N - k:]
        else:
            g = np.zeros(N)
            g[N - k:] += h[N - k:]
        g = 2.0 * g
        g = g - float(np.mean(g))          # stay on sum=1
        h = h - 1e-4 * g
        h = _project(h)
        v = _value(h)
        if v < best_v:
            best_v = v
            best = h.copy()

    return best


def construct():
    return np.asarray(_run(), dtype=float)
