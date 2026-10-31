import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triple indices for vectorized triangle area computation
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    """Compute areas of all C(N,3) triangles in vectorized form."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
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
    """Score = min triangle area / convex hull area."""
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
    """Negative score for minimization."""
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, temperature):
    """Smooth approximation of min using log-sum-exp."""
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    """Score using soft-min instead of hard min."""
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
    """Negative soft score for minimization."""
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


def _neg_epigraph(x, t):
    """
    Epigraph formulation: minimize over point coordinates, with t as auxiliary.
    We want to maximize t such that all triangle areas >= t * hull_area.
    Here we minimize -t subject to soft constraints.
    x = flattened points (2N values), t is separate.
    """
    pts = x.reshape(N, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(pts)
    # We want min(areas) / hull_area >= t, i.e., areas >= t * hull_area
    # Penalty: sum of max(0, t*hull_area - area_i)^2
    violations = np.maximum(0, t * hull_area - areas)
    penalty = np.sum(violations ** 2)
    # Objective: -t + penalty
    return -t + penalty


def _optimize_epigraph(points, t_init, deadline):
    """
    Optimize (points, t) jointly using L-BFGS-B.
    """
    x0 = np.concatenate([points.flatten(), [t_init]])
    
    def obj(x):
        pts = x[:2*N].reshape(N, 2)
        t = x[2*N]
        if t < 0:
            return 1e10
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 1e10
        areas = _triangle_areas_vectorized(pts)
        violations = np.maximum(0, t * hull_area - areas)
        penalty = np.sum(violations ** 2)
        return -t + penalty
    
    try:
        res = minimize(
            obj, x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12},
        )
        pts = res.x[:2*N].reshape(N, 2)
        t = res.x[2*N]
        s = _score(pts)
        return pts, s
    except Exception:
        return points.copy(), _score(points)


# ---------- Seed generators ----------

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
    ang = 2 * np.pi * rng.permutation(N) / N
    r = np.ones(N) * (1.0 + 0.02 * rng.randn(N))
    r = np.clip(r, 0.7, 1.3)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


# ---------- Optimization routines ----------

def _annealed_soft_minimize(points, deadline, max_temp=0.1, min_temp=1e-6, n_steps=8):
    """
    Anneal the soft-min temperature and optimize at each step.
    Uses L-BFGS-B at each temperature level.
    """
    best = points.copy()
    best_s = _score(best)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 5:
            break
        
        try:
            def obj(x):
                pts = x.reshape(N, 2)
                return _neg_soft_score(pts, temp)
            
            res = minimize(
                obj, best.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s


def _polish_true_objective(points, deadline):
    """Polish using the true (non-smooth) objective with Powell then Nelder-Mead."""
    best = points.copy()
    best_s = _score(best)
    
    # Powell
    if time.time() < deadline - 4:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    # Nelder-Mead fine polish
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s


def _local_search(points, deadline, n_iter=300, sigma_start=0.015, sigma_end=0.0002):
    """
    Coordinate-wise local search with adaptive step size.
    """
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
        
        for scale in [1.0, 0.5, 0.2, 0.1, 2.0, 0.05]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction
            
            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break
    
    return best, best_s


def _multi_restart_polish(points, deadline, n_restarts=3):
    """
    Multiple restarts of local optimization from slightly perturbed starting points.
    """
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(99)
    
    for r in range(n_restarts):
        if time.time() >= deadline - 3:
            break
        
        # Perturb
        perturbation = rng.randn(N, 2) * 0.01 * (r + 1)
        init = best + perturbation
        s_init = _score(init)
        
        # Quick Powell polish
        if time.time() < deadline - 2:
            try:
                res = minimize(
                    lambda x: _neg_score(x.reshape(N, 2)),
                    init.flatten(),
                    method='Powell',
                    options={'maxiter': 150, 'xatol': 1e-12, 'fatol': 1e-12},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
    
    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    best_points = None
    best_score = -1

    # ---------- Phase A: Quick coordinate optimization from multiple inits ----------
    inits = []
    inits.append(_seed_regular())
    inits.append(_seed_quasi_regular(42))
    inits.append(_seed_quasi_regular(123))
    inits.append(_seed_two_ring(7, 0.5, 0.0))
    inits.append(_seed_two_ring(6, 0.4, np.pi / 6))
    inits.append(_seed_centered_rings())
    inits.append(_seed_golden_angle(0))
    inits.append(_seed_golden_angle(7))

    for i, init in enumerate(inits):
        if time.time() >= deadline - 30:
            break
        try:
            pts, s = _annealed_soft_minimize(init, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # ---------- Phase B: Epigraph optimization from best ----------
    if best_points is not None and time.time() < deadline - 20:
        try:
            t_init = max(best_score * 0.95, 0.01)
            pts, s = _optimize_epigraph(best_points, t_init, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # ---------- Phase C: Polish best with true objective ----------
    if best_points is not None and time.time() < deadline - 10:
        try:
            pts, s = _polish_true_objective(best_points, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # ---------- Phase D: Multi-restart polish ----------
    if best_points is not None and time.time() < deadline - 8:
        try:
            pts, s = _multi_restart_polish(best_points, deadline, n_restarts=4)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # ---------- Phase E: Local search ----------
    if best_points is not None and time.time() < deadline - 5:
        try:
            pts, s = _local_search(best_points, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # ---------- Phase F: Final Nelder-Mead polish ----------
    if best_points is not None and time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 4000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True},
            )
            pts = res.x.reshape(N, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # ---------- Phase G: One more epigraph pass ----------
    if best_points is not None and time.time() < deadline - 1:
        try:
            t_init = max(best_score * 0.98, 0.01)
            pts, s = _optimize_epigraph(best_points, t_init, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = _seed_regular()

    return best_points
