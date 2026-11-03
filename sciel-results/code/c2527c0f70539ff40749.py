"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_LIMIT = 150.0
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _remaining():
    return max(0.0, _LIMIT - 10.0 - _elapsed())


def _radius(c):
    """True objective: largest common radius for centres c (N,2)."""
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair = np.min(dist) / 2.0
    wall = np.min(0.5 - np.abs(c))
    return float(min(pair, wall))


def _softmin_neg(c, beta):
    """Smooth approximation of -r(c); differentiable, vectorized."""
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, 1e12)
    pair = dist / 2.0
    wall = 0.5 - np.abs(c)
    v = np.concatenate([pair.ravel(), wall.ravel()])
    # log-sum-exp trick for numerical stability
    m = np.min(v)
    return -(m + np.log(np.sum(np.exp(v - m))) / beta)


def _softmin_grad(c, beta):
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, 1e12)
    pair = dist / 2.0
    wall = 0.5 - np.abs(c)
    v = np.concatenate([pair.ravel(), wall.ravel()])
    m = np.min(v)
    w = np.exp(v - m)
    w = w / np.sum(w)
    pair_w = w[:N * N].reshape(N, N)
    wall_w = w[N * N:]
    g = np.zeros_like(c)
    g -= 2.0 * np.sum(pair_w * d, axis=1)
    g += np.sum(wall_w * np.sign(c), axis=0)
    return -beta * g


def _force_relax(c, max_time=1.0):
    """Hard-disk force relaxation: push overlapping circles apart, keep in bounds."""
    c = c.copy()
    r0 = _radius(c)
    best = c.copy()
    best_r = r0
    start = time.time()
    step = 0.004
    repel = 20.0
    for it in range(4000):
        if time.time() - start > max_time:
            break
        d = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=-1))
        np.fill_diagonal(dist, np.inf)
        f = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                dij = dist[i, j]
                if dij < 2 * r0:
                    if dij < 1e-12:
                        dx, dy = 1e-4, 1e-4
                    else:
                        dx, dy = d[i, j]
                    f[i] += repel * np.array([dx, dy]) / dij
                    f[j] -= repel * np.array([dx, dy]) / dij
        # wall forces
        for i in range(N):
            for k in range(2):
                if c[i, k] > 0.5 - r0:
                    f[i, k] -= repel * (c[i, k] - (0.5 - r0))
                elif c[i, k] < -0.5 + r0:
                    f[i, k] += repel * ((-0.5 + r0) - c[i, k])
        c = c + step * f
        c = np.clip(c, -0.5, 0.5)
        if (it % 25 == 0):
            r = _radius(c)
            if r > best_r:
                best_r = r
                best = c.copy()
            # adapt step
            step *= 0.999
    return best, best_r


def _softmin_refine(c, max_time=1.0):
    """L-BFGS-B refinement of the soft-min objective, increasing beta."""
    start = time.time()
    best = c.copy()
    best_r = _radius(c)
    for beta in [15.0, 30.0, 60.0, 120.0, 250.0, 500.0]:
        if time.time() - start > max_time:
            break
        res = minimize(
            _softmin_neg, c, args=(beta,),
            jac=lambda x, b: _softmin_grad(x, b),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200},
        )
        c = res.x
        r = _radius(c)
        if r > best_r:
            best_r = r
            best = c.copy()
    return best, best_r


def _hex_init():
    pts = []
    rows = 10
    for row in range(rows):
        y = -0.5 + 0.5 + (row + 0.5) * (1.0 / rows)
        offset = (0.5 / rows) if (row % 2) else 0.0
        x = -0.5 + offset + (0.5 / rows)
        while x < 0.5:
            pts.append((x, y))
            x += 1.0 / rows
    pts = np.array(pts[:N])
    return pts


def _square_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _random_init(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _jittered_hex(seed):
    rng = np.random.default_rng(seed)
    base = _hex_init()
    return np.clip(base + rng.normal(0, 0.01, size=base.shape), -0.49, 0.49)


def construct():
    inits = [
        ("hex", _hex_init()),
        ("square", _square_init()),
        ("rand0", _random_init(1)),
        ("rand1", _random_init(7)),
        ("jhex0", _jittered_hex(3)),
        ("jhex1", _jittered_hex(11)),
    ]
    best = None
    best_r = -1.0
    for name, c in inits:
        if _remaining() < 2.0:
            break
        # force relax then softmin refine
        c1, r1 = _force_relax(c, max_time=1.5)
        c2, r2 = _softmin_refine(c1, max_time=max(0.5, _remaining() * 0.5))
        if r2 > best_r:
            best_r = r2
            best = c2
        # also try refining the force-relaxed result a bit more if time
        if _remaining() > 2.0:
            c3, r3 = _softmin_refine(c2, max_time=max(0.5, _remaining() * 0.4))
            if r3 > best_r:
                best_r = r3
                best = c3
    if best is None:
        best = _square_init()
    return best
# EVOLVE-BLOCK-END
