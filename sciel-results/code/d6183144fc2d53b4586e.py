import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, beta):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    vmin = np.min(values)
    return vmin - np.log(np.sum(np.exp(-(values - vmin) * beta))) / beta


def _soft_score(points, beta):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    soft_min = _soft_min_value(areas, beta)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, beta):
    s = _soft_score(points, beta)
    if s <= 0:
        return 1e10
    return -s


# ---------- Base topologies ----------

def _base_regular():
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _base_grid_4x4():
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    ys = np.array([-1.5, -0.5, 0.5, 1.5])
    grid = []
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue
            grid.append([xs[i], ys[j]])
    pts = np.array(grid, dtype=np.float64)
    return pts - np.mean(pts, axis=0)


def _base_grid_3x5():
    xs = np.array([-1.0, 0.0, 1.0])
    ys = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    return pts - np.mean(pts, axis=0)


def _base_grid_5x3():
    xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    ys = np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    return pts - np.mean(pts, axis=0)


def _base_hex_staggered():
    """Staggered/hexagonal-like grid."""
    rows = []
    for i in range(4):
        row_x = np.array([-1.5, -0.5, 0.5, 1.5]) + (0.5 if i % 2 else 0)
        row_y = np.full(4, i * 1.0 - 1.5)
        rows.append(np.column_stack([row_x, row_y]))
    pts = np.vstack(rows)[:13]
    return pts - np.mean(pts, axis=0)


def _base_grid_3x3_plus_4():
    xs = np.array([-1.0, 0.0, 1.0])
    ys = np.array([-1.0, 0.0, 1.0])
    grid = []
    for i in range(3):
        for j in range(3):
            grid.append([xs[i], ys[j]])
    ring_ang = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    for a in ring_ang:
        grid.append([2.0 * np.cos(a), 2.0 * np.sin(a)])
    pts = np.array(grid, dtype=np.float64)
    return pts - np.mean(pts, axis=0)


def _base_two_ring(n_out=7, r_in=0.55, offset=np.pi/7):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(offset, 2 * np.pi + offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
    return np.vstack([outer, inner])


def _base_three_ring(n1=5, n2=4, n3=4, r2=0.6, r3=0.3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(np.pi/n1, 2*np.pi + np.pi/n1, n2, endpoint=False)
    a3 = np.linspace(np.pi/n3, 2*np.pi + np.pi/n3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _base_quasi_random(seed=42):
    golden = (1 + np.sqrt(5)) / 2
    angles = np.array([(i * (1.0/golden)) % 1.0 for i in range(N)]) * 2 * np.pi
    radii = 0.8 + 0.4 * np.abs(np.sin(angles * 2 + 1.0))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _base_perturbed_regular(seed=0, sigma=0.05):
    base = _base_regular()
    rng = np.random.RandomState(seed)
    return base + sigma * rng.randn(N, 2)


# ---------- Affine transform helpers ----------

def _apply_affine(base, A_flat):
    A = A_flat.reshape(2, 2)
    return base @ A.T


def _neg_soft_affine(base, beta):
    def obj(A_flat):
        pts = _apply_affine(base, A_flat)
        return _neg_soft_score(pts, beta)
    return obj


def _neg_affine_true(base):
    def obj(A_flat):
        pts = _apply_affine(base, A_flat)
        return _neg_score(pts)
    return obj


# ---------- Annealed soft-min optimization over affine parameters ----------

def _annealed_soft_affine(base, deadline, initial_beta=30.0, final_beta=500.0,
                          n_anneal_steps=8, max_iters_per_step=300):
    A0 = np.array([1.0, 0.0, 0.0, 1.0])
    best_A = A0.copy()
    best_pts = _apply_affine(base, A0)
    best_s = _score(best_pts)

    betas = np.geomspace(initial_beta, final_beta, n_anneal_steps)

    for beta in betas:
        if time.time() >= deadline - 5:
            break

        obj = _neg_soft_affine(base, beta)

        try:
            res = minimize(obj, best_A, method='L-BFGS-B',
                          options={'maxiter': max_iters_per_step, 'gtol': 1e-10, 'ftol': 1e-12})
            cand_pts = _apply_affine(base, res.x)
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best_A = res.x.copy()
                best_pts = cand_pts
        except Exception:
            pass

        if time.time() >= deadline - 5:
            break
        try:
            res = minimize(obj, best_A, method='Powell',
                          options={'maxiter': max_iters_per_step // 2, 'xatol': 1e-10, 'fatol': 1e-10})
            cand_pts = _apply_affine(base, res.x)
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best_A = res.x.copy()
                best_pts = cand_pts
        except Exception:
            pass

    return best_pts, best_s


# ---------- Full-coordinate polish ----------

def _optimize_full(points, deadline, max_iters_powell=400):
    best = points.copy()
    best_s = _score(best)

    if time.time() < deadline - 8:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': max_iters_powell, 'xatol': 1e-9, 'fatol': 1e-9},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    if time.time() < deadline - 4:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    return best, best_s


# ---------- Full-coordinate annealed soft-min ----------

def _annealed_soft_full(points, deadline, initial_beta=50.0, final_beta=800.0,
                        n_anneal_steps=6, max_iters_per_step=200):
    best = points.copy()
    best_s = _score(best)

    betas = np.geomspace(initial_beta, final_beta, n_anneal_steps)

    for beta in betas:
        if time.time() >= deadline - 5:
            break

        def obj(x):
            pts = x.reshape(N, 2)
            return _neg_soft_score(pts, beta)

        try:
            res = minimize(obj, best.flatten(), method='L-BFGS-B',
                          options={'maxiter': max_iters_per_step, 'gtol': 1e-9, 'ftol': 1e-12})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand
        except Exception:
            pass

        if time.time() >= deadline - 5:
            break
        try:
            res = minimize(obj, best.flatten(), method='Powell',
                          options={'maxiter': max_iters_per_step // 2, 'xatol': 1e-9, 'fatol': 1e-9})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand
        except Exception:
            pass

    return best, best_s


# ---------- Differential Evolution global search ----------

def _de_objective(x_flat):
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score


def _run_de(deadline, max_iter=40):
    """Run differential evolution to find a good global starting point."""
    time_remaining = deadline - time.time()
    if time_remaining < 20:
        return None

    bounds = [(-2.0, 2.0)] * (2 * N)

    try:
        result = differential_evolution(
            _de_objective,
            bounds,
            maxiter=max_iter,
            popsize=25,
            mutation=(0.4, 0.9),
            recombination=0.8,
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol',
            callback=None,
        )
        if result is not None and result.fun < 1e9:
            return result.x
    except Exception:
        pass

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    np.random.seed(42)

    best_points = None
    best_score = 0.0

    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # ---------- Phase 1: Evaluate diverse base topologies ----------
    bases = [
        _base_regular(),
        _base_grid_4x4(),
        _base_grid_3x5(),
        _base_grid_5x3(),
        _base_hex_staggered(),
        _base_grid_3x3_plus_4(),
    ]
    for n_out in [6, 7, 8]:
        for r_in in [0.45, 0.55, 0.65]:
            for a_off in [0.0, np.pi / n_out, np.pi / (2.0 * n_out)]:
                bases.append(_base_two_ring(n_out, r_in, a_off))
    bases.append(_base_three_ring(5, 4, 4, 0.6, 0.3))
    bases.append(_base_three_ring(6, 4, 3, 0.55, 0.25))
    for seed in [42, 123, 999]:
        bases.append(_base_quasi_random(seed))
    for seed in [0, 1, 2]:
        bases.append(_base_perturbed_regular(seed, sigma=0.04))

    for base in bases:
        _update(base)

    if best_points is None:
        best_points = _base_regular()
        best_score = _score(best_points)

    # ---------- Phase 2: Differential Evolution global search ----------
    if time.time() < deadline - 60:
        de_result = _run_de(deadline, max_iter=40)
        if de_result is not None:
            _update(de_result.reshape(N, 2))

    # ---------- Phase 3: Affine-optimized annealing from top grid bases ----------
    affine_bases = [
        _base_grid_4x4(),
        _base_grid_3x5(),
        _base_grid_5x3(),
        _base_hex_staggered(),
        _base_regular(),
        _base_two_ring(7, 0.55, np.pi/7),
        _base_three_ring(5, 4, 4, 0.6, 0.3),
        _base_quasi_random(42),
    ]

    for base in affine_bases:
        if time.time() >= deadline - 30:
            break
        try:
            pts, s = _annealed_soft_affine(base, deadline,
                                           initial_beta=20.0, final_beta=400.0,
                                           n_anneal_steps=6, max_iters_per_step=150)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 4: Full-coordinate annealed soft-min from best ----------
    if best_points is not None and time.time() < deadline - 20:
        try:
            pts, s = _annealed_soft_full(best_points, deadline,
                                         initial_beta=40.0, final_beta=600.0,
                                         n_anneal_steps=5, max_iters_per_step=120)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 5: Full-coordinate polish ----------
    if best_points is not None and time.time() < deadline - 12:
        try:
            pts, s = _optimize_full(best_points, deadline, max_iters_powell=300)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 6: Final ultra-fine Nelder-Mead ----------
    if best_points is not None and time.time() < deadline - 4:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = _base_regular()

    return best_points
