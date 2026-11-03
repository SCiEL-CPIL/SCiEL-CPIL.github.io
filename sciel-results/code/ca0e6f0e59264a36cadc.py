import numpy as np
import time

N = 11
T_START = time.time()
DEADLINE = 142.0  # safety margin before 150s kill

# Unit hexagon (pointy-top), circumradius 1 -> vertices at angles 90,150,...
_HEX = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]
                 for a in (90, 150, 210, 270, 330, 30)])
# Face normals (outward), apothem 1 for unit hex -> cos/sin of 60,120,...
_NRM = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]
                 for a in (60, 120, 180, 240, 300, 0)])
_NRM_T = _NRM.T  # (2,6)
_EPS = 1e-6


def _rot(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _max_apothem(centers, angles):
    """Vectorized: max over all inner-hex vertices of (n . (c + R v)) for 6 normals."""
    Rv = np.array([_rot(angles[k]) @ _HEX.T for k in range(N)])  # (N,2,6)
    Cv = centers[:, None, :] + Rv                                # (N,2,6)
    return float(np.max((Cv @ _NRM_T).max(axis=1)))


def _coll(centers, angles):
    """Max penetration depth (>=0) among all inner-hex pairs; 0 if disjoint."""
    Rv = np.array([_rot(angles[k]) @ _HEX.T for k in range(N)])  # (N,2,6)
    c1 = centers[:, None, :]
    c2 = centers[None, :, :]
    a1 = Rv[:, None, :, :]
    a2 = Rv[None, :, :, :]
    # edge normals of A (from A's rotated verts) and B
    na = a1[:, :, 1:, :] - a1[:, :, :-1, :]
    na = na / (np.linalg.norm(na, axis=1, keepdims=True) + 1e-12)
    nb = a2[:, :, 1:, :] - a2[:, :, :-1, :]
    nb = nb / (np.linalg.norm(nb, axis=1, keepdims=True) + 1e-12)
    # separating axis test across both sets of edge normals
    t = np.maximum.reduce([
        np.abs((c1 - c2) @ na),
        np.abs((a1 - a2) @ na),
        np.abs((c1 - c2) @ nb),
        np.abs((a1 - a2) @ nb),
    ])
    pen = np.max(t) - (np.min(t) + _EPS)
    return float(max(0.0, pen))


def _objective(centers, angles):
    return _max_apothem(centers, angles) + 1e5 * _coll(centers, angles)


def _anneal(centers, angles, max_iters, t0, t1):
    cur_c = centers.copy()
    cur_a = angles.copy()
    cur_o = _objective(cur_c, cur_a)
    best_c, best_a, best_o = cur_c.copy(), cur_a.copy(), cur_o
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        temp = t0 * (t1 / t0) ** (it / max(1, max_iters))
        k = np.random.randint(N)
        nc = cur_c.copy()
        na = cur_a.copy()
        nc[k, 0] += np.random.normal(0, 0.15)
        nc[k, 1] += np.random.normal(0, 0.15)
        na[k] += np.random.normal(0, 0.15)
        no = _objective(nc, na)
        d = no - cur_o
        if d < 0 or np.random.random() < np.exp(-d / max(temp, 1e-9)):
            cur_c, cur_a, cur_o = nc, na, no
            if cur_o < best_o:
                best_c, best_a, best_o = cur_c.copy(), cur_a.copy(), cur_o
    return best_c, best_a, best_o


def _initials():
    """Diverse, mostly-non-overlapping starting layouts."""
    D = np.sqrt(3)
    configs = []
    # hexagonal cluster: center + ring1 + ring2
    pts = [(0.0, 0.0)]
    for k in range(6):
        a = k * np.pi / 3
        pts.append((D * np.cos(a), D * np.sin(a)))
    for k in range(12):
        a = k * np.pi / 6 + np.pi / 6
        pts.append((2 * D * np.cos(a), 2 * D * np.sin(a)))
    pts = pts[:N]
    configs.append((np.array(pts, float), np.zeros(N)))
    # linear
    configs.append((np.array([(i - 5) * D for i in range(N)], float).reshape(N, 2)
                    .copy(), np.zeros(N)))
    # two rows
    row = []
    for i in range(6):
        row.append(((i - 2.5) * D, 0.0))
    for i in range(5):
        row.append(((i - 2) * D + D / 2, 1.5))
    row = row[:N]
    configs.append((np.array(row, float), np.zeros(N)))
    # triangular 3-4-3-1
    tri = []
    for i in range(3):
        tri.append(((i - 1) * D, 0.0))
    for i in range(4):
        tri.append(((i - 1.5) * D + D / 2, 1.5))
    for i in range(3):
        tri.append(((i - 1) * D, 3.0))
    tri.append((0.0, 4.5))
    tri = tri[:N]
    configs.append((np.array(tri, float), np.zeros(N)))
    # random scatter
    rng = np.random.default_rng(1)
    configs.append((rng.uniform(-3 * D, 3 * D, (N, 2)), rng.uniform(-0.5, 0.5, N)))
    return configs


def _worker(args):
    seed, cfg_idx, max_iters = args
    np.random.seed(seed)
    centers, angles = _initials()[cfg_idx]
    return _anneal(centers, angles, max_iters, 0.6, 0.001)


def construct():
    global T_START
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except Exception:
        pass

    cfgs = _initials()
    # Parallel restarts: 16 independent anneals, each with a large budget.
    n_proc = 16
    per_iter = 4000
    jobs = [(1000 + i * 7919, i % len(cfgs), per_iter) for i in range(n_proc)]

    best_c, best_a, best_o = None, None, float('inf')
    try:
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_proc) as pool:
            for bc, ba, bo in pool.imap_unordered(_worker, jobs):
                if bo < best_o:
                    best_c, best_a, best_o = bc, ba, bo
    except Exception:
        # Fallback to sequential if multiprocessing is unavailable.
        for j in jobs:
            if time.time() > DEADLINE:
                break
            bc, ba, bo = _worker(j)
            if bo < best_o:
                best_c, best_a, bo = bc, ba, bo

    if best_c is None:
        bc, ba, bo = _anneal(cfgs[0][0], cfgs[0][1], 8000, 0.6, 0.001)
        best_c, best_a, best_o = bc, ba, bo

    # Sequential local refinement of the global best (fast core -> many iters).
    np.random.seed(0)
    step = 0.05
    for it in range(20000):
        if time.time() > DEADLINE:
            break
        k = np.random.randint(N)
        nc = best_c.copy()
        na = best_a.copy()
        nc[k, 0] += np.random.normal(0, step)
        nc[k, 1] += np.random.normal(0, step)
        na[k] += np.random.normal(0, step)
        no = _objective(nc, na)
        if no < best_o:
            best_c, best_a, best_o = nc, na, no
        step *= 0.9998

    # Center the whole cloud at the origin (minimal enclosing hexagon).
    best_c = best_c - best_c.mean(axis=0)
    best_side = _max_apothem(best_c, best_a)

    inner = [[float(best_c[i, 0]), float(best_c[i, 1]), float(best_a[i])]
             for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0
