"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points.

Structural change: replace the symmetry-constrained + generic local-search pipeline with a
targeted worst-triangle simulated-annealing search seeded by Fermat-spiral and diverse ring
initializations, followed by a smooth-min anneal and a final Powell refinement.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 140.0  # safety margin before 150s kill

# Pre-compute all C(13,3)=286 triple indices for fast vectorized scoring
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)
_num_triples = len(_triple_i)


def _triangle_areas(pts):
    """Vectorized computation of all triangle areas."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)

    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0

    areas = _triangle_areas(pts)
    min_area = np.min(areas)

    if min_area < 1e-15:
        return 0.0

    return min_area / hull_area


def _soft_min(areas, beta):
    """Smooth approximation of min(areas) using log-sum-exp."""
    neg = -areas / beta
    max_neg = np.max(neg)
    log_sum = max_neg + np.log(np.sum(np.exp(neg - max_neg)))
    return -beta * log_sum


def _smooth_score(pts, beta):
    """Differentiable surrogate for the score."""
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return -1e10
    areas = _triangle_areas(pts)
    sm = _soft_min(areas, beta)
    return sm / hull_area


def _neg_smooth_score(params, beta):
    """Negative smooth score for minimization."""
    pts = params.reshape(N, 2)
    return -_smooth_score(pts, beta)


def _neg_score(params):
    """Negative exact score for minimization."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)


def _fermat_spiral(n=N, radius=1.0, center=(0.0, 0.0)):
    """Fermat spiral: points at r = sqrt(i/n) * radius, theta = i * golden_angle."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    idx = np.arange(n)
    r = radius * np.sqrt((idx + 0.5) / n)
    theta = idx * golden_angle
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y])


def _regular_polygon(n_points, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])


def _rings_config(n1, n2, n3, r1=1.0, r2=0.5, r3=0.25, p1=0.0, p2=0.0, p3=0.0):
    pts_list = []
    if n1 > 0:
        a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + p1
        pts_list.append(np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)]))
    if n2 > 0:
        a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + p2
        pts_list.append(np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]))
    if n3 > 0:
        a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + p3
        pts_list.append(np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]))
    return np.vstack(pts_list)[:N]


def _normalize_unit_area(pts):
    hull_area = _hull_area(pts)
    if hull_area > 1e-12:
        return pts * np.sqrt(1.0 / hull_area)
    return pts


def _simulate_annealing(pts, t_start, budget=25.0, seed=0):
    """
    Simulated annealing targeting the worst (smallest-area) triangle.
    At each step, pick the worst triangle, perturb one of its 3 points,
    accept based on Metropolis criterion on the score.
    """
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_score = _score_points(current)
    best = current.copy()
    best_score = current_score

    # Annealing schedule
    t_end = t_start + budget
    n_steps = 0
    T_init = 0.05
    T_min = 1e-6
    # We'll run until time budget, adapting step size
    step_size_init = 0.1
    step_size_min = 0.001
    total_steps_target = 20000

    # Estimate steps per second to budget time
    # Just run until deadline
    while time.time() < t_end:
        n_steps += 1
        progress = min(n_steps / total_steps_target, 1.0)
        T = T_init * (T_min / T_init) ** progress
        step_size = step_size_init * (step_size_min / step_size_init) ** progress

        areas = _triangle_areas(current)
        worst_idx = np.argmin(areas)
        # Pick one of the 3 points in the worst triangle
        tri_points = [_triple_i[worst_idx], _triple_j[worst_idx], _triple_k[worst_idx]]
        pt_idx = tri_points[rng.randint(3)]

        # Perturb the chosen point
        delta = rng.normal(0, step_size, size=2)
        new_pt = current.copy()
        new_pt[pt_idx] += delta

        new_score = _score_points(new_pt)

        # Metropolis acceptance
        if new_score >= current_score:
            current = new_pt
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        else:
            accept_prob = np.exp((new_score - current_score) / T)
            if rng.random() < accept_prob:
                current = new_pt
                current_score = new_score

    return best, best_score


def _smooth_anneal(params, t_start, budget=10.0):
    """Anneal the smooth surrogate with decreasing beta."""
    current = params.copy()
    best_score = -_neg_smooth_score(current, 0.05)
    best_params = current.copy()

    betas = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    t_end = t_start + budget
    for beta in betas:
        if time.time() > t_end:
            break
        try:
            res = minimize(
                _neg_smooth_score,
                current,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-10}
            )
            candidate = res.x
            score = _score_points(candidate.reshape(N, 2))
            if score > best_score:
                best_score = score
                best_params = candidate.copy()
            current = candidate
        except Exception:
            pass

    return best_params, best_score


def _powell_refine(params, t_start, budget=8.0):
    """Final Powell refinement on the exact objective."""
    t_end = t_start + budget
    if time.time() > t_end:
        return params, _score_points(params.reshape(N, 2))
    try:
        res = minimize(
            _neg_score,
            params,
            method='Powell',
            options={'maxiter': 200, 'maxfun': 10000, 'xtol': 1e-10, 'ftol': 1e-10}
        )
        candidate = res.x
        score = _score_points(candidate.reshape(N, 2))
        return candidate, score
    except Exception:
        return params, _score_points(params.reshape(N, 2))


def construct():
    t_start = time.time()

    best_params = None
    best_score = 0.0

    # Generate diverse initializations
    inits = []

    # Fermat spiral
    inits.append(_fermat_spiral(N, radius=1.0))

    # Regular 13-gon
    inits.append(_regular_polygon(N, radius=1.0))

    # Two-ring configurations
    for n1, n2, r2, p2 in [(9, 4, 0.5, 0.0), (9, 4, 0.5, np.pi / 4), (8, 5, 0.45, 0.0),
                           (8, 5, 0.45, np.pi / 5), (7, 6, 0.55, 0.0), (7, 6, 0.55, np.pi / 6)]:
        inits.append(_rings_config(n1, n2, 0, r1=1.0, r2=r2, p1=0.0, p2=p2))

    # Three-ring configurations
    for n1, n2, n3, r2, r3, p2 in [(5, 4, 4, 0.6, 0.3, 0.0), (5, 4, 4, 0.6, 0.3, np.pi / 4),
                                   (6, 4, 3, 0.55, 0.25, 0.0), (6, 4, 3, 0.55, 0.25, np.pi / 4)]:
        inits.append(_rings_config(n1, n2, n3, r1=1.0, r2=r2, r3=r3, p1=0.0, p2=p2))

    # Random
    rng0 = np.random.RandomState(42)
    inits.append(rng0.uniform(-1, 1, size=(N, 2)))

    # Normalize all to unit hull area
    for i in range(len(inits)):
        inits[i] = _normalize_unit_area(inits[i])

    # Phase 1: Simulated annealing from each initialization
    # Budget: ~25s per init, but we have 10 inits. Use a smaller budget per init.
    # Total SA budget: ~100s
    sa_budget_per_init = 8.0
    for idx, init in enumerate(inits):
        if time.time() > DEADLINE - 20.0:
            break
        try:
            sa_pts, sa_score = _simulate_annealing(init, t_start, budget=sa_budget_per_init, seed=100 + idx)
            if sa_score > best_score:
                best_score = sa_score
                best_params = sa_pts.flatten()
        except Exception:
            pass

    # Phase 2: Smooth anneal from the best SA result (and a couple of others)
    candidates_for_smooth = []
    if best_params is not None:
        candidates_for_smooth.append(best_params.copy())
    # Also add the top few SA results if we tracked them; for simplicity, use init + best
    if len(inits) > 0:
        candidates_for_smooth.append(inits[0].flatten())  # Fermat spiral
    if len(inits) > 1:
        candidates_for_smooth.append(inits[1].flatten())  # Regular polygon

    for cand in candidates_for_smooth:
        if time.time() > DEADLINE - 15.0:
            break
        try:
            sp, ss = _smooth_anneal(cand, t_start, budget=8.0)
            if ss > best_score:
                best_score = ss
                best_params = sp
        except Exception:
            pass

    # Phase 3: Final Powell refinement
    if best_params is not None:
        try:
            fp, fs = _powell_refine(best_params, t_start, budget=10.0)
            if fs > best_score:
                best_score = fs
                best_params = fp
        except Exception:
            pass

    # Safety: if no valid result, return regular polygon
    if best_params is None:
        poly13 = _regular_polygon(N, radius=1.0)
        best_params = _normalize_unit_area(poly13).flatten()

    return best_params.reshape(N, 2)
