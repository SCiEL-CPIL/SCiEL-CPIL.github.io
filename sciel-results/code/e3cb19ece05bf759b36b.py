"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _radius(c):
    """Compute the maximum common radius for centers c of shape (N,2)."""
    n = c.shape[0]
    # Pairwise distances (upper triangle only for efficiency)
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d = np.sqrt((diff ** 2).sum(axis=1))
    rp = d.min() / 2.0
    # Wall distances
    rw = np.minimum(np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
                    np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)).min()
    return min(rp, rw)


def _relax(c, R, iters, eta0, eta1):
    """Relaxation: push circles apart at target radius R, handle walls."""
    c = c.copy()
    n = len(c)
    for it in range(iters):
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t
        # Pairwise forces
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        # Force magnitude: push apart if closer than 2R
        mag = np.maximum(2.0 * R - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, None] * diff).sum(axis=1)
        # Wall forces
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c


def _grow(start, R0, R1, step, iters, eta0, eta1, t0, deadline):
    """Grow radius from R0 to R1, keeping the best configuration found."""
    c = start.copy()
    best_c = c.copy()
    best_r = _radius(c)
    R = R0
    while R <= R1:
        if (time.time() - t0) > deadline - 5:
            break
        c = _relax(c, R, iters, eta0, eta1)
        r = _radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
        if (time.time() - t0) > deadline - 5:
            break
    return best_c, best_r


def _init_7x7_plus():
    """7x7 grid (49 points) plus one extra at origin."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])


def _init_jitter8(rng):
    """8x8 grid with jitter, pick 50 points."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.03, 0.03, c.shape)
    return c


def _init_rand(rng):
    """Random uniform initialization."""
    return rng.uniform(-0.45, 0.45, (N, 2))


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin before 150s kill

    rng = np.random.default_rng(12345)

    # Multiple diverse initializations
    inits = [
        _init_7x7_plus(),
        _init_jitter8(rng),
        _init_rand(rng),
    ]

    best_c = None
    best_r = -1.0

    for start in inits:
        if (time.time() - t0) > deadline - 15:
            break
        # Growth phase: expand from small radius to near-optimal
        c, r = _grow(start, 0.04, 0.0715, 0.0003, 1200, 0.025, 0.002, t0, deadline)
        if r > best_r:
            best_r = r
            best_c = c

    if best_c is None:
        # Fallback: return 7x7 grid
        best_c = _init_7x7_plus()
        best_r = _radius(best_c)

    # Final polish: relax at the current best radius for a few more iterations
    if (time.time() - t0) < deadline - 3:
        r_now = _radius(best_c)
        c_polished = _relax(best_c, r_now, 3000, 0.01, 0.0003)
        r_polished = _radius(c_polished)
        if r_polished >= r_now:
            best_c = c_polished
            best_r = r_polished

    # One more very fine relaxation pass if time permits
    if (time.time() - t0) < deadline - 1:
        r_now = _radius(best_c)
        c_final = _relax(best_c, r_now, 2000, 0.005, 0.0001)
        r_final = _radius(c_final)
        if r_final >= r_now:
            best_c = c_final
            best_r = r_final

    # Ensure validity: clip to square, check finiteness
    best_c = np.clip(best_c, -0.5 + 1e-15, 0.5 - 1e-15)
    if not np.all(np.isfinite(best_c)) or _radius(best_c) <= 0:
        # Fallback to grid
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_c
# EVOLVE-BLOCK-END
