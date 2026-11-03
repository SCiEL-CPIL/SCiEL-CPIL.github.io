"""
Heilbronn n=14: Maximize minimum triangle area / convex hull area.

Strategy:
  1. Use the soft-min (log-sum-exp) objective from the inspiration to make the
     min-triangle-area term differentiable, enabling SLSQP.
  2. Run SLSQP from multiple diverse initial configurations (regular polygon,
     perturbed, two-ring, Fekete-like).
  3. Polish the best result with Nelder-Mead and Powell on the hard objective.
  4. Explicit deadline guard: return best result found before 135 s.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14
_START_TIME = time.time()
DEADLINE = 135.0  # safety margin before 150 s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

# ---------------------------------------------------------------------------
# Precompute triangle index triples for vectorized area computation
# ---------------------------------------------------------------------------
_idx = np.arange(N)
_II, _JJ, _KK = np.meshgrid(_idx, _idx, _idx, indexing='ij')
_mask = (_II < _JJ) & (_JJ < _KK)
_T_I = _II[_mask].ravel()
_T_J = _JJ[_mask].ravel()
_T_K = _KK[_mask].ravel()
N_TRI = len(_T_I)

def _all_areas(points):
    """Return array of all C(N,3) triangle areas (vectorized)."""
    pi = points[_T_I]
    pj = points[_T_J]
    pk = points[_T_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)

def _score(points):
    """min triangle area / convex hull area."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    areas = _all_areas(points)
    min_a = np.min(areas)
    if min_a < 1e-14:
        return 0.0
    return min_a / hull_area

# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
def _objective_soft(p, beta=50.0):
    """
    Differentiable objective: -soft_min(areas) / hull_area.
    soft_min uses log-sum-exp for numerical stability.
    """
    points = p.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
    if hull_area < 1e-10:
        return 1e10
    areas = _all_areas(points)
    if len(areas) == 0:
        return 1e10
    min_a = np.min(areas)
    shifted = areas - min_a
    # soft min: min_a - (1/beta)*log(sum(exp(-beta*shifted)))
    soft_min_val = min_a - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
    score = soft_min_val / hull_area
    return -score

def _objective_hard(p):
    """Hard objective: -score."""
    points = p.reshape(N, 2)
    return -_score(points)

# ---------------------------------------------------------------------------
# Initial configurations
# ---------------------------------------------------------------------------
def _generate_initials():
    inits = []
    rng = np.random.RandomState(42)

    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    # 2. Perturbed regular polygons
    for _ in range(4):
        noise = rng.uniform(-0.08, 0.08, (N, 2))
        inits.append(np.column_stack([np.cos(angles), np.sin(angles)]) + noise)

    # 3. Two-ring: inner 4 + outer 10
    inner_a = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer_a = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi / 10
    inner = 0.5 * np.column_stack([np.cos(inner_a), np.sin(inner_a)])
    outer = np.column_stack([np.cos(outer_a), np.sin(outer_a)])
    inits.append(np.vstack((inner, outer)))

    # 4. Fekete-like via repulsion (maximize min pairwise distance)
    pts_f = rng.uniform(-1, 1, (N, 2))
    norms = np.linalg.norm(pts_f, axis=1)
    pts_f[norms > 1.0] /= norms[norms > 1.0, None]

    def _min_dist_obj(flat):
        pts = flat.reshape(N, 2)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return -np.min(d)

    if _time_left() > 100:
        try:
            res = minimize(_min_dist_obj, pts_f.flatten(), method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10})
            inits.append(res.x.reshape(N, 2))
        except Exception:
            pass

    # 5. Random points in unit disk
    for _ in range(3):
        r = np.sqrt(rng.rand(N))
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))

    return inits

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    inits = _generate_initials()

    best_params = inits[0].flatten().copy()
    best_score = _score(inits[0])

    # Phase 1: SLSQP with soft objective from each initial
    for init in inits:
        if _time_left() < 25:
            break
        try:
            res = minimize(
                _objective_soft,
                init.flatten(),
                method='SLSQP',
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            cand_score = _score(res.x.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead polish on hard objective
    if _time_left() > 15:
        try:
            res_nm = minimize(
                _objective_hard,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True}
            )
            cand_score = _score(res_nm.x.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_params = res_nm.x.copy()
        except Exception:
            pass

    # Phase 3: Powell polish
    if _time_left() > 10:
        try:
            res_pow = minimize(
                _objective_hard,
                best_params,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-14,
                         'adaptive': True}
            )
            cand_score = _score(res_pow.x.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_params = res_pow.x.copy()
        except Exception:
            pass

    # Normalize to unit hull area
    points = best_params.reshape(N, 2)
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(points)
        hull_area = hull.volume

    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        points = points * scale

    return points
