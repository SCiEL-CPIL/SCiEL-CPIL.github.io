import numpy as np
import time
from scipy.optimize import minimize
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


def _soft_min_value(values, temperature):
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    s = _soft_score(points)
    if s <= 0:
        return 1e10
    return -s


def _penalty_objective_radial(x, mu):
    """
    Penalty objective in radial coords: minimize -t/hull_area + mu * sum(max(0, t - area_i)^2)
    x = [t, r1,...,rN, theta1,...,thetaN]
    """
    t = x[0]
    r = np.clip(x[1:1+N], 1e-6, 1.0)
    theta = x[1+N:1+2*N]
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    areas = _triangle_areas_vectorized(pts)
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e10
    return -t / hull_a + mu * penalty


def _penalty_optimize_radial(points, deadline, mu_start=10.0, mu_end=1e8, n_mu_steps=15,
                              max_iter_per_mu=120):
    """
    Penalty-based optimization in radial coordinates.
    """
    best = points.copy()
    best_s = _score(best)
    
    areas = _triangle_areas_vectorized(best)
    t_init = float(np.min(areas)) * 0.85
    
    r0 = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    theta0 = np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)
    r0 = np.clip(r0, 1e-6, 1.0)
    
    x0 = np.zeros(1 + 2 * N)
    x0[0] = t_init
    x0[1:1+N] = r0
    x0[1+N:] = theta0
    
    mus = np.geomspace(mu_start, mu_end, n_mu_steps)
    
    for mu in mus:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(
                lambda x: _penalty_objective_radial(x, mu),
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_mu, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            x0 = res.x
            r = np.clip(x0[1:1+N], 1e-6, 1.0)
            theta = x0[1+N:]
            cand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best = cand_pts.copy()
        except Exception:
            pass
    
    # Additional polish pass with higher mu
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _penalty_objective_radial(x, mu_end),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            x0 = res.x
            r = np.clip(x0[1:1+N], 1e-6, 1.0)
            theta = x0[1+N:]
            cand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best = cand_pts.copy()
        except Exception:
            pass
    
    return best, best_s


def _seed_regular():
    """Regular 13-gon."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    """Two-ring configuration."""
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_center_ring(n_center, r_inner, angle_offset=0.0):
    """Configuration with points at center and points on a ring."""
    n_ring = N - n_center
    pts = []
    for i in range(n_center):
        if n_center == 1:
            pts.append([0.0, 0.0])
        else:
            ang = 2 * np.pi * i / n_center
            pts.append([0.1 * np.cos(ang), 0.1 * np.sin(ang)])
    
    a_ring = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_ring, endpoint=False)
    ring = np.column_stack([np.cos(a_ring), np.sin(a_ring)])
    pts.extend(ring)
    return np.array(pts)


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    """Three-ring configuration."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular(seed=42):
    """Quasi-regular: regular polygon with small random perturbations."""
    rng = np.random.RandomState(seed)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + 0.03 * rng.randn(N)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _seed_centered_rings():
    """Center + inner ring + outer ring."""
    pts = np.zeros((N, 2))
    for i in range(6):
        a = 2 * np.pi * i / 6
        pts[1 + i] = [0.5 * np.cos(a), 0.5 * np.sin(a)]
    for i in range(6):
        a = 2 * np.pi * i / 6
        pts[7 + i] = [1.0 * np.cos(a), 1.0 * np.sin(a)]
    return pts


def _seed_golden_angle(seed=0):
    """Points on a circle with golden angle spacing."""
    rng = np.random.RandomState(seed)
    n = N
    angles = []
    golden = 2 * np.pi * (1 - 1 / (1 + 5 ** 0.5) / 2)
    for i in range(n):
        angles.append(i * golden)
    angles = np.array(angles) % (2 * np.pi)
    r = np.ones(n)
    r += 0.02 * rng.randn(n)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def _annealed_soft_minimize(x0, deadline, max_temp=0.1, min_temp=1e-4, n_steps=8):
    """
    Anneal the soft-min temperature and optimize at each step.
    Returns the best point configuration found.
    """
    best_points = x0.copy()
    best_score = _score(best_points)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() > deadline - 10:
            break
        
        def obj(x):
            pts = x.reshape(N, 2)
            return _neg_soft_score(pts, temp)
        
        try:
            res = minimize(
                obj,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass
        
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    obj,
                    best_points.flatten(),
                    method='Powell',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass
    
    return best_points, best_score


def _perturb_restart(points, sigma=0.05):
    return points + sigma * np.random.randn(*points.shape)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
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
    
    # ---------- Generate diverse seeds ----------
    seeds = []
    
    seeds.append(_seed_regular())
    
    for n_c in [1, 2, 3]:
        for r_in in [0.5, 0.7, 0.9]:
            seeds.append(_seed_center_ring(n_c, r_in))
    
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.5, 0.7]:
            for a_off in [0.0, np.pi / n_out, np.pi / (2 * n_out)]:
                seeds.append(_seed_two_ring(n_out, r_in, a_off))
    
    three_ring_configs = [
        (5, 4, 4, 0.6, 0.3, 0.0, 0.0),
        (6, 4, 3, 0.6, 0.3, 0.0, 0.0),
        (7, 4, 2, 0.6, 0.3, 0.0, 0.0),
        (6, 5, 2, 0.5, 0.25, np.pi / 6, 0.0),
    ]
    for (n1, n2, n3, r2, r3, o2, o3) in three_ring_configs:
        seeds.append(_seed_three_ring(n1, n2, n3, r2, r3, o2, o3))
    
    for i in range(5):
        seeds.append(_seed_quasi_regular(seed=42 + i))
    
    seeds.append(_seed_centered_rings())
    for i in range(3):
        seeds.append(_seed_golden_angle(seed=i))
    
    # ---------- Evaluate all seeds ----------
    for seed in seeds:
        _update(seed)
    
    if best_points is None:
        best_points = _seed_regular()
        best_score = _score(best_points)
    
    # ---------- Annealed soft-min optimization from best seed ----------
    if time.time() < deadline - 20:
        try:
            pts, s = _annealed_soft_minimize(best_points, deadline, max_temp=0.15, min_temp=1e-5, n_steps=12)
            _update(pts)
        except Exception:
            pass
    
    # ---------- Penalty-based radial optimization from best solution ----------
    if time.time() < deadline - 25:
        try:
            pts, s = _penalty_optimize_radial(best_points, deadline, mu_start=10.0, mu_end=1e8, n_mu_steps=15, max_iter_per_mu=120)
            _update(pts)
        except Exception:
            pass
    
    # ---------- Perturbation restarts from best solution ----------
    n_restarts = 8
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        sigma = 0.02 + 0.01 * i
        perturbed = _perturb_restart(best_points, sigma)
        
        remaining_budget = max(10, deadline - time.time() - 5)
        try:
            pts, s = _annealed_soft_minimize(perturbed, time.time() + remaining_budget * (1.0 - 0.05 * i), 
                                             max_temp=0.08, min_temp=1e-5, n_steps=8)
            _update(pts)
        except Exception:
            pass
        
        # Also try penalty radial from perturbed
        if time.time() < deadline - 10:
            try:
                pts, s = _penalty_optimize_radial(perturbed, deadline, mu_start=50.0, mu_end=1e8, n_mu_steps=10, max_iter_per_mu=80)
                _update(pts)
            except Exception:
                pass
    
    # ---------- Final L-BFGS-B polish on the true (non-smooth) objective ----------
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # ---------- Final Powell polish ----------
    if best_points is not None and time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # ---------- Final ultra-fine Nelder-Mead ----------
    if best_points is not None and time.time() < deadline - 1.5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass
    
    if best_points is None:
        best_points = _seed_regular()
    
    return best_points
