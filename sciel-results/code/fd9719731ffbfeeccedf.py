import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute triple index arrays for vectorized area computation
_TRIPLES = list(combinations(range(N), 3))
_TI = np.array([t[0] for t in _TRIPLES], dtype=np.int32)
_TJ = np.array([t[1] for t in _TRIPLES], dtype=np.int32)
_TK = np.array([t[2] for t in _TRIPLES], dtype=np.int32)


def _triangle_areas(pts):
    """Vectorized computation of all triangle areas."""
    x = pts[:, 0]
    y = pts[:, 1]
    v1x = x[_TJ] - x[_TI]
    v1y = y[_TJ] - y[_TI]
    v2x = x[_TK] - x[_TI]
    v2y = y[_TK] - y[_TI]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def compute_score(points):
    """Return (min_triangle_area / hull_area, hull_area)."""
    area_hull = _hull_area(points)
    if area_hull < 1e-9:
        return 0.0, 0.0
    areas = _triangle_areas(points)
    return float(np.min(areas)) / area_hull, area_hull


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 142.0  # Leave 8s margin before the 150s kill
    hard_deadline = start_time + time_limit

    def _soft_min_score(points, beta=50.0, area_ref=1.0):
        """Smooth surrogate for the min-triangle-area / hull-area ratio (to MINIMIZE)."""
        area_hull = _hull_area(points)
        if area_hull < 1e-9:
            return 1e6
        areas = _triangle_areas(points)
        neg = -beta * areas
        m = np.max(neg)
        if m < -1e6:
            return 1e6
        soft_min = -(m + np.log(np.sum(np.exp(neg - m)))) / beta
        soft_ratio = soft_min / area_hull
        pen = 0.5 * (1.0 - area_hull / area_ref) ** 2
        return -soft_ratio + pen

    def _soft_objective(pts_flat, beta=50.0, area_ref=1.0):
        return _soft_min_score(pts_flat.reshape(-1, 2), beta, area_ref)

    def _hard_score_neg(pts_flat):
        s, _ = compute_score(pts_flat.reshape(-1, 2))
        return -s

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), phase=0.0):
        if n == 1:
            return np.array([center, center])
        a = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
        return np.column_stack([center[0] + radius * np.cos(a),
                                center[1] + radius * np.sin(a)])

    def _ring_config(r1, n1, r2=0.0, n2=0, n3=0, phase1=0.0, phase2=0.0, r3=0.0, phase3=0.0):
        parts = []
        if n1 > 0:
            parts.append(_regular_polygon(n1, r1, phase=phase1))
        if n2 > 0:
            parts.append(_regular_polygon(n2, r2, phase=phase2))
        if n3 == 1:
            parts.append(np.array([[0.0, 0.0]]))
        elif n3 > 1:
            parts.append(_regular_polygon(n3, r3 if r3 > 0 else 0.25 * max(r2, 0.3), phase=phase3))
        if not parts:
            return _regular_polygon(N, 1.0)
        return np.vstack(parts)

    def _refine_soft_min(pts, deadline):
        """Annealed soft-min L-BFGS-B."""
        current = pts.copy()
        current_s, _ = compute_score(current)
        area_ref = max(_hull_area(current), 1e-6)
        for beta in [8.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0]:
            if time.time() > deadline - 6.0:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b, area_ref),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > current_s:
                    current_s = s
                    current = cand.copy()
            except Exception:
                pass
        return current

    def _refine_nelder_mead(pts, deadline, maxiter=400):
        """Nelder-Mead polish on the exact score."""
        current = pts.copy()
        current_s, _ = compute_score(current)
        if time.time() > deadline - 5.0:
            return current
        try:
            res = opt.minimize(
                _hard_score_neg,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'adaptive': True, 'xatol': 1e-11, 'fatol': 1e-13},
            )
            cand = res.x.reshape(-1, 2)
            s, _ = compute_score(cand)
            if s > current_s:
                current = cand.copy()
        except Exception:
            pass
        return current

    def _targeted_worst_triangle_opt(pts, deadline):
        """
        STRUCTURAL refinement: identify the worst (smallest-area) triangles and
        iteratively nudge the points they contain directly to grow the minimum
        triangle area. This attacks the non-differentiable bottleneck head-on,
        complementing the gradient-based soft-min surrogates.
        """
        current = pts.copy()
        current_s, _ = compute_score(current)
        if time.time() > deadline - 5.0:
            return current

        step = 0.05

        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            areas = _triangle_areas(current)
            min_area = np.min(areas)
            threshold = min_area * 1.03
            worst = np.where(areas <= threshold)[0]
            if len(worst) == 0:
                break
            if len(worst) > 6:
                worst = worst[np.argsort(areas[worst])[:6]]

            improved = False
            for wi in worst:
                i, j, k = _TRIPLES[wi]
                centroid = (current[i] + current[j] + current[k]) / 3.0
                for pt_idx in [i, j, k]:
                    vec = current[pt_idx] - centroid
                    norm = np.linalg.norm(vec)
                    if norm < 1e-9:
                        continue
                    unit = vec / norm
                    # Try multiple step sizes for finer control
                    for sgn in (1.0, -1.0):
                        for mult in [1.0, 0.5, 0.25]:
                            trial = current.copy()
                            trial[pt_idx] = current[pt_idx] + sgn * step * mult * unit
                            ts, _ = compute_score(trial)
                            if ts > current_s + 1e-12:
                                current = trial
                                current_s = ts
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                step *= 0.8
                if step < 1e-6:
                    break
        return current

    def _barycentric_parametric_search(deadline):
        """
        NEW design family: model the 13 points as a regular 13-gon with a subset
        of points displaced radially inward, and solve for the optimal radial
        positions and angular rotations on a small continuous parameter space
        (grid + Nelder-Mead). Leverages the known near-symmetric structure of
        optimal Heilbronn configurations.
        """
        best = None
        best_s = -float('inf')

        def _eval_params(n_inner, r_inner, rot_outer, rot_inner):
            a_outer = np.linspace(0, 2 * np.pi, N - n_inner, endpoint=False) + rot_outer
            outer = np.column_stack([np.cos(a_outer), np.sin(a_outer)])
            if n_inner == 0:
                return outer
            a_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rot_inner
            inner = r_inner * np.column_stack([np.cos(a_inner), np.sin(a_inner)])
            return np.vstack([outer, inner])

        # grid over (n_inner, r_inner, rot_inner)
        for n_inner in [1, 2, 3, 4]:
            for r_inner in [0.30, 0.40, 0.50, 0.60, 0.70]:
                for rot_inner in [0.0, np.pi / max(n_inner, 1)]:
                    if time.time() > deadline - 6.0:
                        break
                    cand = _eval_params(n_inner, r_inner, 0.0, rot_inner)
                    s, _ = compute_score(cand)
                    if s > best_s:
                        best_s = s
                        best = cand.copy()

        # Nelder-Mead polish on the continuous (r_inner, rot_inner) for the best n_inner
        if best is not None and time.time() < deadline - 4.0:
            # recover n_inner from best by matching point count of inner ring is ambiguous;
            # instead re-polish each n_inner with a small NM over (r_inner, rot_inner)
            for n_inner in [1, 2, 3, 4]:
                if time.time() > deadline - 4.0:
                    break
                def _obj(p):
                    r_inner = float(np.clip(p[0], 0.05, 0.95))
                    rot_inner = float(p[1])
                    cand = _eval_params(n_inner, r_inner, 0.0, rot_inner)
                    s, _ = compute_score(cand)
                    return -s
                try:
                    res = opt.minimize(
                        _obj,
                        np.array([0.5, 0.0]),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-9, 'fatol': 1e-12},
                    )
                    cand = _eval_params(n_inner, float(np.clip(res.x[0], 0.05, 0.95)), 0.0, float(res.x[1]))
                    s, _ = compute_score(cand)
                    if s > best_s:
                        best_s = s
                        best = cand.copy()
                except Exception:
                    pass
        return best

    # ---- Build diverse starting configurations ----
    starts = []
    # Symmetric ring configs (strong geometry prior)
    starts.append(_ring_config(1.0, 12, n3=1))
    starts.append(_ring_config(1.0, 6, 0.5, 6, 1))
    starts.append(_ring_config(1.0, 7, 0.45, 5, 1))
    starts.append(_ring_config(1.0, 8, 0.4, 4, 1))
    starts.append(_ring_config(1.0, 5, 0.45, 7, 1))
    starts.append(_ring_config(1.0, 13, 0.0, 0, 0))
    starts.append(_ring_config(1.0, 12, n3=1, phase1=np.pi / 12))
    starts.append(_ring_config(1.0, 9, 0.6, 3, 1))
    starts.append(_ring_config(1.0, 10, 0.5, 2, 1))
    starts.append(_ring_config(1.0, 11, 0.4, 1, 1))
    starts.append(_ring_config(1.0, 6, 0.65, 6, 1))
    starts.append(_ring_config(1.0, 6, 0.35, 6, 1))
    # More phase variations
    starts.append(_ring_config(1.0, 12, n3=1, phase1=np.pi / 6))
    starts.append(_ring_config(1.0, 6, 0.5, 6, 1, phase1=np.pi / 6, phase2=np.pi / 3))
    starts.append(_ring_config(1.0, 7, 0.45, 5, 1, phase1=np.pi / 7, phase2=np.pi / 5))
    # k_hull / k_inner polygon splits
    for k_hull in [3, 4, 5, 6, 7, 8, 10]:
        k_inner = N - k_hull
        hull = _regular_polygon(k_hull, 1.0)
        if k_inner == 0:
            starts.append(hull)
        elif k_inner == 1:
            starts.append(np.vstack([hull, np.array([[0.0, 0.0]])]))
        else:
            inner = _regular_polygon(k_inner, 0.35, phase=np.pi / k_inner)
            starts.append(np.vstack([hull, inner]))

    # Three-ring configurations
    for n1, n2, n3 in [(6, 4, 3), (5, 4, 4), (7, 3, 3), (8, 3, 2), (5, 5, 3), (6, 5, 2)]:
        for r2 in [0.4, 0.55]:
            for r3 in [0.2, 0.35]:
                starts.append(_ring_config(1.0, n1, r2, n2, n3,
                                           phase1=0.0, phase2=np.pi / max(n2, 1),
                                           r3=r3, phase3=np.pi / max(n3, 1)))

    # Random in disk
    rng = np.random.RandomState(123)
    for _ in range(6):
        r = rng.rand(N) ** 0.5
        theta = rng.rand(N) * 2 * np.pi
        starts.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))

    # ---- Search: allocate a per-start budget ----
    best_points = None
    best_score = -float('inf')
    n_starts = len(starts)
    reserve_final = 22.0
    search_budget = time_limit - reserve_final
    per_start = search_budget / n_starts

    for idx, st in enumerate(starts):
        if time.time() > hard_deadline - 16:
            break
        remaining_starts = n_starts - idx
        deadline = min(hard_deadline - 16,
                       hard_deadline - 16 + (remaining_starts - 1) * per_start)
        deadline = max(deadline, start_time + 5.0)
        try:
            cand = _refine_soft_min(st, deadline)
            cand = _targeted_worst_triangle_opt(cand, deadline)
            cand = _refine_nelder_mead(cand, deadline, maxiter=300)
            s, _ = compute_score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            s, _ = compute_score(st)
            if s > best_score:
                best_score = s
                best_points = st.copy()

    # ---- Barycentric parametric search (new design family) ----
    if time.time() < hard_deadline - 16:
        try:
            bp = _barycentric_parametric_search(hard_deadline - 16)
            if bp is not None:
                bp = _targeted_worst_triangle_opt(bp, hard_deadline - 10)
                s, _ = compute_score(bp)
                if s > best_score:
                    best_score = s
                    best_points = bp.copy()
        except Exception:
            pass

    # ---- Final global polish on the best found ----
    if best_points is not None and time.time() < hard_deadline - 10:
        for beta in [500.0, 1000.0, 2000.0]:
            if time.time() > hard_deadline - 10:
                break
            area_ref = max(_hull_area(best_points), 1e-6)
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b, area_ref),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

        if time.time() < hard_deadline - 10:
            best_points = _targeted_worst_triangle_opt(best_points, hard_deadline - 8)
            best_score, _ = compute_score(best_points)

        if time.time() < hard_deadline - 6:
            best_points = _refine_nelder_mead(best_points, hard_deadline - 3, maxiter=500)
            best_score, _ = compute_score(best_points)

        # Final ultra-precise polish
        if time.time() < hard_deadline - 3:
            area_ref = max(_hull_area(best_points), 1e-6)
            try:
                res = opt.minimize(
                    lambda x: _soft_objective(x, 5000.0, area_ref),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 120},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(N, 1.0)

    return best_points

# EVOLVE-BLOCK-END
