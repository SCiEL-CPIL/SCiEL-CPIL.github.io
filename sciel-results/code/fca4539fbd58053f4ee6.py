"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

# Precompute triangle indices (pair constraints) once
_TRI_I, _TRI_J = np.triu_indices(N, k=1)
_N_PAIRS = len(_TRI_I)

# Precompute wall index maps: for each axis, which g_all slot is +wall (0.5-c) and -wall (c+0.5)
# g_all = [pairs (N_PAIRS), x_pos, x_neg, y_pos, y_neg, z_pos, z_neg]  each wall block size N
_WS = _N_PAIRS

def _get_radius(c):
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    dw = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
    return min(min_pair, np.min(dw))

def _smooth_obj(c_flat, beta):
    c = c_flat.reshape(N, 3)
    if np.any(np.abs(c) > 0.5 + 1e-9):
        return 1e6, np.zeros_like(c_flat)

    diff_full = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d_full = np.sqrt(np.sum(diff_full ** 2, axis=2))
    d_pairs = d_full[_TRI_I, _TRI_J] / 2.0

    g = np.empty(_N_PAIRS + 6 * N)
    g[:_N_PAIRS] = d_pairs
    g[_WS:_WS + N] = 0.5 - c[:, 0]
    g[_WS + N:_WS + 2 * N] = c[:, 0] + 0.5
    g[_WS + 2 * N:_WS + 3 * N] = 0.5 - c[:, 1]
    g[_WS + 3 * N:_WS + 4 * N] = c[:, 1] + 0.5
    g[_WS + 4 * N:_WS + 5 * N] = 0.5 - c[:, 2]
    g[_WS + 5 * N:_WS + 6 * N] = c[:, 2] + 0.5

    neg_beta_g = -beta * g
    lse = logsumexp(neg_beta_g)
    val = -lse / beta

    w = np.exp(neg_beta_g - lse)

    grad_c = np.zeros_like(c)
    diff_ij = c[_TRI_I] - c[_TRI_J]
    d_ij = d_full[_TRI_I, _TRI_J]
    safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
    unit = diff_ij / safe_d[:, np.newaxis]
    wp = w[:_N_PAIRS] * 0.5 / safe_d
    contrib = wp[:, np.newaxis] * unit
    np.add.at(grad_c, _TRI_I, contrib)
    np.add.at(grad_c, _TRI_J, -contrib)

    grad_c[:, 0] -= w[_WS:_WS + N]
    grad_c[:, 0] += w[_WS + N:_WS + 2 * N]
    grad_c[:, 1] -= w[_WS + 2 * N:_WS + 3 * N]
    grad_c[:, 1] += w[_WS + 3 * N:_WS + 4 * N]
    grad_c[:, 2] -= w[_WS + 4 * N:_WS + 5 * N]
    grad_c[:, 2] += w[_WS + 5 * N:_WS + 6 * N]

    return val, -grad_c.flatten()

_BOUNDS = [(-0.4999, 0.4999)] * (N * 3)

def _lbfgs(c_init, beta, max_iter, ftol=1e-18, gtol=1e-12):
    def fun(x):
        v, _ = _smooth_obj(x, beta)
        return v
    def jac(x):
        _, g = _smooth_obj(x, beta)
        return g
    try:
        res = minimize(fun, c_init.flatten(), jac=jac, method='L-BFGS-B',
                       bounds=_BOUNDS,
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                'maxls': 100, 'maxfun': max_iter * 15, 'maxcor': 50})
        return res.x.reshape(N, 3)
    except Exception:
        return c_init.copy()

def _anneal(c_init, deadline, min_stage=1.0):
    # Adaptive beta schedule: start moderate, ramp up aggressively
    betas = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960]
    c = c_init.copy()
    for i, b in enumerate(betas):
        if time.time() > deadline - 3.0:
            break
        rem = deadline - 3.0 - time.time()
        # Allocate more iterations to early (lower beta) stages, less to later
        # Early stages need to escape local minima; later stages need precision
        if i < 4:
            max_iter = int(min(400, max(50, rem * 50 / (1.0 + 0.1 * i))))
        elif i < 8:
            max_iter = int(min(300, max(40, rem * 35 / (1.0 + 0.05 * i))))
        else:
            max_iter = int(min(250, max(30, rem * 25)))
        if max_iter < 10:
            break
        c = _lbfgs(c, b, max_iter)
    return c

def _init_grid():
    m = 4
    t = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in t for y in t for x in t])
    return g[:N].copy()

def _init_bcc():
    nc = 4
    cs = 1.0 / nc
    pts = []
    for iz in range(nc):
        for iy in range(nc):
            for ix in range(nc):
                cx = -0.5 + cs * (ix + 0.5)
                cy = -0.5 + cs * (iy + 0.5)
                cz = -0.5 + cs * (iz + 0.5)
                pts.append([cx, cy, cz])
                pts.append([cx + cs / 2, cy + cs / 2, cz + cs / 2])
    a = np.array(pts)
    a = a[np.all(np.abs(a) <= 0.5, axis=1)]
    if len(a) >= N:
        return a[:N].copy()
    extra = np.random.uniform(-0.45, 0.45, size=(N - len(a), 3))
    return np.vstack([a, extra])

def _init_fcc():
    # FCC lattice: face-centered cubic
    a = 1.0 / 3  # lattice parameter
    pts = []
    for iz in range(4):
        for iy in range(4):
            for ix in range(4):
                cx = -0.5 + a * ix
                cy = -0.5 + a * iy
                cz = -0.5 + a * iz
                pts.append([cx, cy, cz])
                # Face centers
                if ix < 3:
                    pts.append([cx + a/2, cy, cz])
                if iy < 3:
                    pts.append([cx, cy + a/2, cz])
                if iz < 3:
                    pts.append([cx, cy, cz + a/2])
                # Body center
                if ix < 3 and iy < 3 and iz < 3:
                    pts.append([cx + a/2, cy + a/2, cz + a/2])
    a_arr = np.array(pts)
    a_arr = a_arr[np.all(np.abs(a_arr) <= 0.5, axis=1)]
    if len(a_arr) >= N:
        return a_arr[:N].copy()
    extra = np.random.uniform(-0.45, 0.45, size=(N - len(a_arr), 3))
    return np.vstack([a_arr, extra])

def _init_layered():
    # 5 layers of 10 spheres each, staggered
    pts = []
    for layer in range(5):
        z = -0.4 + 0.2 * layer
        for j in range(10):
            x = -0.45 + 0.09 * j
            y_offset = 0.03 * (j % 2)
            y = -0.3 + y_offset
            pts.append([x, y, z])
    a = np.array(pts)
    a = a[np.all(np.abs(a) <= 0.5, axis=1)]
    if len(a) >= N:
        return a[:N].copy()
    extra = np.random.uniform(-0.45, 0.45, size=(N - len(a), 3))
    return np.vstack([a, extra])

def _init_random(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.4, 0.4, size=(N, 3))

def _active_set_refine(c, deadline):
    """Identify binding constraints and do a focused refinement."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    r = _get_radius(c)
    
    # Find the k most critical pairs (closest to each other)
    d_pairs = d[_TRI_I, _TRI_J]
    # Sort by distance, take the closest 30 pairs
    k = min(30, len(d_pairs))
    idx = np.argsort(d_pairs)[:k]
    
    # Do a high-beta optimization focusing on these
    if time.time() > deadline - 2.0:
        return c
    
    # Use very high beta for precision
    for b in [50000, 100000, 200000]:
        if time.time() > deadline - 2.0:
            break
        rem = deadline - 2.0 - time.time()
        max_iter = int(min(200, max(30, rem * 30)))
        c = _lbfgs(c, b, max_iter, ftol=1e-20, gtol=1e-14)
    
    return c

def construct():
    start = time.time()
    deadline = start + 115.0  # 5s safety margin before 120s limit

    np.random.seed(12345)

    # More diverse initializations
    inits = [
        _init_grid(),
        _init_bcc(),
        _init_fcc(),
        _init_layered(),
        _init_random(1),
        _init_random(7),
        _init_random(13),
        _init_random(42),
    ]

    best_c = inits[0].copy()
    best_r = _get_radius(best_c)

    # Phase 1: Anneal from each initialization
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        c = _anneal(init, deadline)
        r = _get_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # Phase 2: Active-set refinement on the best configuration
    if time.time() < deadline - 10.0:
        c = _active_set_refine(best_c, deadline)
        r = _get_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        for b in [100000, 200000, 400000]:
            if time.time() > deadline - 3.0:
                break
            rem = deadline - 3.0 - time.time()
            max_iter = int(min(200, max(40, rem * 40)))
            c = _lbfgs(best_c, b, max_iter, ftol=1e-20, gtol=1e-14)
            r = _get_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()

    # Phase 4: One more pass with even higher beta if time allows
    if time.time() < deadline - 2.0:
        b = 800000
        rem = deadline - 2.0 - time.time()
        max_iter = int(min(150, max(20, rem * 25)))
        c = _lbfgs(best_c, b, max_iter, ftol=1e-20, gtol=1e-15)
        r = _get_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # Final safety: ensure strictly inside cube and finite
    best_c = np.clip(best_c, -0.4999, 0.4999)
    if not np.all(np.isfinite(best_c)):
        best_c = _init_grid()

    return best_c
# EVOLVE-BLOCK-END
