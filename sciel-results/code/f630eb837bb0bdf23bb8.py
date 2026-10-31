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
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


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


def _seed_quasi_regular():
    base = _seed_regular()
    rng = np.random.RandomState(42)
    radial_pert = rng.normal(0, 0.05, N)
    angular_pert = rng.normal(0, 0.05, N)
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = base_angles + angular_pert
    radii = 1.0 + radial_pert
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _sdp_gram_optimization(points, deadline, max_iters=100):
    """
    SDP-inspired Gram matrix optimization.
    We parameterize points via a Gram matrix G = X^T X (N x N, PSD).
    Triangle area^2 for (i,j,k) = 0.25 * det([[1,1,1],[x_i,x_j,x_k],[y_i,y_j,y_k]])^2
    In terms of Gram: area^2(i,j,k) = 0.25 * (G_ii*G_jj - G_ij^2 + G_jj*G_kk - G_jk^2 + G_kk*G_ii - G_ik^2 
                                               + 2*G_ij*G_jk*G_ik - G_ii*G_jk^2 - G_jj*G_ik^2 - G_kk*G_ij^2)
    This is complex. Instead, we use a simpler approach:
    Optimize the Gram matrix entries directly with constraints G >= 0 (PSD) and area constraints.
    """
    # Start from the given points
    best = points.copy()
    best_s = _score(best)
    
    # Try to improve using the Gram matrix representation
    # G[i,j] = dot(p_i, p_j)
    # We optimize G entries to maximize min triangle area subject to G being PSD
    # For simplicity, we use a heuristic: adjust points to equalize triangle areas
    
    # Identify the minimum area triangle
    areas = _triangle_areas_vectorized(best)
    min_idx = np.argmin(areas)
    i, j, k = _TRIPLE_IDX[min_idx]
    
    # Try to increase the minimum area by moving points
    # Use a simple gradient-free approach on the Gram matrix
    for iteration in range(max_iters):
        if time.time() >= deadline - 5:
            break
        
        # Recompute areas
        areas = _triangle_areas_vectorized(best)
        min_area = np.min(areas)
        min_indices = np.where(areas <= min_area * 1.01)[0]
        
        # For each minimum triangle, try to increase its area
        for mi in min_indices[:3]:  # Only try a few
            ii, jj, kk = _TRIPLE_IDX[mi]
            
            # Compute gradient of area w.r.t. points
            pi = best[ii]
            pj = best[jj]
            pk = best[kk]
            
            # Area = 0.5 * |cross(pj-pi, pk-pi)|
            v1 = pj - pi
            v2 = pk - pi
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            
            if cross >= 0:
                # Gradient w.r.t. pi
                grad_pi = np.array([-v2[1], v2[0]])
                # Gradient w.r.t. pj
                grad_pj = np.array([v2[1], -v2[0]])
                # Gradient w.r.t. pk
                grad_pk = np.array([-v1[1], v1[0]])
            else:
                grad_pi = np.array([v2[1], -v2[0]])
                grad_pj = np.array([-v2[1], v2[0]])
                grad_pk = np.array([v1[1], -v1[0]])
            
            # Move points slightly in the gradient direction
            step = 0.001 * min_area
            new_best = best.copy()
            new_best[ii] += step * grad_pi
            new_best[jj] += step * grad_pj
            new_best[kk] += step * grad_pk
            
            s = _score(new_best)
            if s > best_s:
                best_s = s
                best = new_best
    
    return best, best_s


def _annealed_soft_minimize(points, deadline, initial_temp=0.05, final_temp=0.001,
                            n_anneal_steps=8, max_iters_per_step=200):
    best = points.copy()
    best_s = _score(best)
    
    temps = np.geomspace(initial_temp, final_temp, n_anneal_steps)
    
    for temp in temps:
        if time.time() >= deadline - 5:
            break
        
        x0 = best.flatten()
        
        def obj(x):
            pts = x.reshape(N, 2)
            return _neg_soft_score(pts, temp)
        
        try:
            res = minimize(
                obj, x0, method='L-BFGS-B',
                options={'maxiter': max_iters_per_step, 'gtol': 1e-8},
            )
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
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': max_iters_per_step // 2, 'xatol': 1e-8, 'fatol': 1e-8},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand
        except Exception:
            pass
    
    return best, best_s


def _optimize_from_points(points, deadline, max_iters_powell=300):
    best = points.copy()
    best_s = _score(best)

    x0 = best.flatten()

    if time.time() < deadline - 6:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                x0,
                method='Powell',
                options={'maxiter': max_iters_powell, 'xatol': 1e-8, 'fatol': 1e-8},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

    # ---------- Seed candidates ----------
    _update(_seed_regular())
    
    for n_out in [5, 6, 7, 8]:
        for r_in in [0.3, 0.5, 0.7]:
            for a_off in [0.0, np.pi / n_out, np.pi / (2.0 * n_out)]:
                _update(_seed_two_ring(n_out, r_in, a_off))
    
    for (n1, n2, n3, r2, r3, o2, o3) in [
        (5, 4, 4, 0.6, 0.3, 0.0, 0.0),
        (6, 4, 3, 0.6, 0.3, 0.0, 0.0),
        (7, 4, 2, 0.6, 0.3, 0.0, 0.0),
    ]:
        _update(_seed_three_ring(n1, n2, n3, r2, r3, o2, o3))
    
    _update(_seed_quasi_regular())

    # ---------- Annealed soft-min optimization from multiple seeds ----------
    seeds = [
        _seed_regular(),
        _seed_two_ring(6, 0.5, np.pi / 6),
        _seed_two_ring(7, 0.5, np.pi / 7),
        _seed_three_ring(6, 4, 3, 0.6, 0.3, 0.0, 0.0),
        _seed_quasi_regular(),
    ]
    
    for seed in seeds:
        if time.time() >= deadline - 20:
            break
        try:
            pts, s = _annealed_soft_minimize(seed, deadline,
                                             initial_temp=0.05, final_temp=0.001,
                                             n_anneal_steps=6, max_iters_per_step=150)
            _update(pts)
        except Exception:
            pass

    # ---------- SDP-inspired Gram optimization ----------
    if best_points is not None and time.time() < deadline - 15:
        try:
            pts, s = _sdp_gram_optimization(best_points, deadline, max_iters=50)
            _update(pts)
        except Exception:
            pass

    # ---------- Full-coordinate polish ----------
    if best_points is not None and time.time() < deadline - 10:
        try:
            pts, s = _optimize_from_points(best_points, deadline, max_iters_powell=300)
            _update(pts)
        except Exception:
            pass

    # ---------- Final ultra-fine Nelder-Mead ----------
    if best_points is not None and time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
