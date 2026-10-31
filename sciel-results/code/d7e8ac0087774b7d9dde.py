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
        if np.ptp(points, axis=0).max() < 1e-12:
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


def _soft_min_value(values, temperature):
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    shifted = np.clip(shifted, 0, 100)
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.min(areas) < 1e-15:
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


# ---------- Radial representation ----------

def _radial_to_cartesian(x_flat):
    n = len(x_flat) // 2
    radii = np.maximum(x_flat[0::2], 0.01)
    angles = x_flat[1::2]
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _cartesian_to_radial(points):
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    angles = np.arctan2(points[:, 1], points[:, 0])
    x = np.zeros(2 * len(points))
    x[0::2] = radii
    x[1::2] = angles
    return x


# ---------- Penalty-based formulation in radial coords ----------

def _penalty_objective_radial(x, mu):
    t = x[0]
    r = np.clip(x[1:1+N], 0.01, 1.5)
    theta = x[1+N:1+2*N]
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    areas = _triangle_areas_vectorized(pts)
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e10
    return -t / hull_a + mu * penalty


def _penalty_optimize_radial(points, deadline, mu_start=10.0, mu_end=1e7, n_mu_steps=12,
                              max_iter_per_mu=150):
    best = points.copy()
    best_s = _score(best)
    
    areas = _triangle_areas_vectorized(best)
    t_init = float(np.min(areas)) * 0.8
    
    r0 = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    theta0 = np.arctan2(points[:, 1], points[:, 0])
    r0 = np.clip(r0, 0.01, 1.5)
    
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
            r = np.clip(x0[1:1+N], 0.01, 1.5)
            theta = x0[1+N:]
            cand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best = cand_pts.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _penalty_objective_radial(x, mu_end),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            x0 = res.x
            r = np.clip(x0[1:1+N], 0.01, 1.5)
            theta = x0[1+N:]
            cand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best = cand_pts.copy()
        except Exception:
            pass
    
    return best, best_s


# ---------- Fixed-angle support-line representation ----------

def _polygon_from_fixed_angles(offsets):
    n = len(offsets)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    vertices = np.zeros((n, 2))
    for i in range(n):
        i1 = i
        i2 = (i + 1) % n
        n1 = normals[i1]
        n2 = normals[i2]
        d1 = offsets[i1]
        d2 = offsets[i2]
        
        det = n1[0] * n2[1] - n1[1] * n2[0]
        if abs(det) < 1e-12:
            return None
        vertices[i, 0] = (d1 * n2[1] - d2 * n1[1]) / det
        vertices[i, 1] = (n1[0] * d2 - n2[0] * d1) / det
    
    for j in range(n):
        vals = vertices @ normals[j]
        if np.max(vals) > offsets[j] + 1e-6:
            return None
    
    return vertices


def _score_from_fixed_offsets(offsets):
    pts = _polygon_from_fixed_angles(offsets)
    if pts is None:
        return 0.0
    return _score(pts)


def _offsets_from_points(points):
    n = len(points)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    offsets = np.array([np.max(points @ normals[j]) for j in range(n)])
    return offsets


def _penalty_objective_fixed_offsets(x, mu):
    offsets = np.maximum(x, 0.01)
    pts = _polygon_from_fixed_angles(offsets)
    if pts is None:
        return 1e10
    t = x[0] if len(x) > N else float(np.min(_triangle_areas_vectorized(pts)))
    areas = _triangle_areas_vectorized(pts)
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e10
    return -t / hull_a + mu * penalty


# ---------- Annealed soft-min in radial coords ----------

def _neg_soft_score_radial(x_flat, temperature):
    points = _radial_to_cartesian(x_flat)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(points)
    if np.min(areas) < 1e-15:
        return 1e10
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 1e10
    s = soft_min / hull_area
    return -s


def _annealed_soft_minimize_radial(x0, deadline, max_temp=0.05, min_temp=1e-4, n_steps=8):
    best = x0.copy()
    best_s = _score(_radial_to_cartesian(best))
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 5:
            break
        if time.time() < deadline - 4:
            try:
                res = minimize(
                    _neg_soft_score_radial, best, args=(temp,),
                    method='Powell',
                    options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand = res.x
                pts = _radial_to_cartesian(cand)
                s = _score(pts)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
    
    return best, best_s


# ---------- Coordinate-descent polishing ----------

def _coordinate_descent_polish(points, deadline, n_rounds=50):
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(77)
    
    for rnd in range(n_rounds):
        if time.time() >= deadline - 3:
            break
        
        improved = False
        for idx in range(N):
            if time.time() >= deadline - 3:
                break
            base = best[idx].copy()
            best_s_base = best_s
            
            for axis in range(2):
                for scale in [0.02, 0.01, 0.005, 0.001, -0.02, -0.01, -0.005, -0.001]:
                    cand = best.copy()
                    cand[idx, axis] = base[axis] + scale
                    s = _score(cand)
                    if s > best_s + 1e-15:
                        best = cand
                        best_s = s
                        base = cand[idx].copy()
                        improved = True
                        break
        
        if not improved:
            break
    
    return best, best_s


# ---------- Local search with adaptive steps ----------

def _local_search_cartesian(points, deadline, n_iter=400, sigma_start=0.03, sigma_end=0.0003):
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(123)
    
    for it in range(n_iter):
        if time.time() >= deadline - 3:
            break
        
        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress
        
        idx = rng.randint(N)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-15)
        
        for scale in [1.0, 0.5, 0.2, 2.0, 0.1]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction
            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break
    
    return best, best_s


# ---------- Seed generators ----------

def _seed_regular():
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular(seed_val=42):
    rng = np.random.RandomState(seed_val)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + 0.05 * rng.randn(N)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _seed_random_ring(rng):
    n_outer = int(rng.integers(5, 10))
    n_inner = N - n_outer
    if n_inner > 0:
        o_angles = np.sort(rng.uniform(0, 2 * np.pi, n_outer))
        i_angles = np.sort(rng.uniform(0, 2 * np.pi, n_inner))
        r_in = rng.uniform(0.3, 0.7)
        o = np.column_stack([np.cos(o_angles), np.sin(o_angles)])
        i = np.column_stack([r_in * np.cos(i_angles), r_in * np.sin(i_angles)])
        return np.vstack([o, i])
    else:
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        return np.column_stack([np.cos(angles), np.sin(angles)])


def _seed_concentric():
    n_out = 7
    n_in = 6
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(np.pi / n_in, 2 * np.pi + np.pi / n_in, n_in, endpoint=False)
    r_in = 0.6
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
    return np.vstack([outer, inner])


# ---------- Polish true objective ----------

def _polish_true_objective_cartesian(x0, deadline):
    best = x0.copy()
    best_s = _score(best.reshape(N, 2))
    
    if time.time() < deadline - 4:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)) if _score(x.reshape(N, 2)) > 0 else 1e10,
                best, method='Powell',
                options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-12},
            )
            cand = res.x
            s = _score(cand.reshape(N, 2))
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)) if _score(x.reshape(N, 2)) > 0 else 1e10,
                best, method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            cand = res.x
            s = _score(cand.reshape(N, 2))
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Generate diverse seeds
    rng = np.random.default_rng(42)
    seeds = [
        _seed_regular(),
        _seed_two_ring(7, 0.55, np.pi / 6),
        _seed_two_ring(6, 0.5, np.pi / 5),
        _seed_two_ring(8, 0.4, np.pi / 4),
        _seed_three_ring(5, 4, 4, 0.65, 0.35, np.pi / 5, np.pi / 4),
        _seed_three_ring(6, 4, 3, 0.6, 0.3, np.pi / 3, np.pi / 6),
        _seed_quasi_regular(42),
        _seed_quasi_regular(123),
        _seed_concentric(),
    ]
    for _ in range(8):
        seeds.append(_seed_random_ring(rng))
    
    for seed in seeds:
        _update_best(seed)
    
    # Phase 1: Penalty-based optimization in radial coords (primary structural approach)
    for i, seed in enumerate(seeds[:8]):
        if time.time() > deadline - 60:
            break
        try:
            pts_opt, s_opt = _penalty_optimize_radial(seed, deadline, mu_start=5.0, mu_end=1e6, n_mu_steps=10, max_iter_per_mu=120)
            _update_best(pts_opt)
        except Exception:
            pass
    
    # Phase 2: Annealed soft-min in radial coords
    for i, seed in enumerate(seeds[:6]):
        if time.time() > deadline - 40:
            break
        try:
            radial_seed = _cartesian_to_radial(seed)
            x_opt, s_opt = _annealed_soft_minimize_radial(radial_seed, deadline, max_temp=0.05, min_temp=1e-4, n_steps=6)
            pts = _radial_to_cartesian(x_opt)
            _update_best(pts)
        except Exception:
            pass
    
    # Phase 3: Coordinate-descent polish on best
    if best_points is not None and time.time() < deadline - 25:
        try:
            cd_pts, s_cd = _coordinate_descent_polish(best_points, deadline, n_rounds=40)
            _update_best(cd_pts)
        except Exception:
            pass
    
    # Phase 4: Local search
    if best_points is not None and time.time() < deadline - 15:
        try:
            ls_points, s_ls = _local_search_cartesian(best_points, deadline, n_iter=300)
            _update_best(ls_points)
        except Exception:
            pass
    
    # Phase 5: Final penalty polish
    if best_points is not None and time.time() < deadline - 10:
        try:
            pts_opt, s_opt = _penalty_optimize_radial(best_points, deadline, mu_start=10.0, mu_end=1e7, n_mu_steps=8, max_iter_per_mu=150)
            _update_best(pts_opt)
        except Exception:
            pass
    
    # Phase 6: Coordinate-descent again
    if best_points is not None and time.time() < deadline - 8:
        try:
            cd_pts, s_cd = _coordinate_descent_polish(best_points, deadline, n_rounds=30)
            _update_best(cd_pts)
        except Exception:
            pass
    
    # Phase 7: Final local search + polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            ls_points, s_ls = _local_search_cartesian(best_points, deadline, n_iter=200, sigma_start=0.01, sigma_end=0.0001)
            _update_best(ls_points)
        except Exception:
            pass
    
    if best_points is not None and time.time() < deadline - 3:
        try:
            x_final, s_final = _polish_true_objective_cartesian(best_points.flatten(), deadline)
            _update_best(x_final.reshape(N, 2))
        except Exception:
            pass
    
    if best_points is None:
        best_points = _seed_regular()
    
    return best_points
