"""Optimized program for Heilbronn convex regions: 13 points maximizing min triangle area / hull area.

Structural change: angular/radial parameterization (13 angles + 13 radii = 26 params)
with L-BFGS-B on a smooth log-sum-exp surrogate, then full-2D L-BFGS-B polish,
then targeted worst-triangle refinement.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
TIME_LIMIT = 138.0  # safety margin before 150s kill

# Precompute triangle indices for speed
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_N_TRI = len(_TRI_INDICES)


def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles. Returns array of shape (C(n,3),)."""
    # pts shape (n, 2)
    i_idx, j_idx, k_idx = _TRI_INDICES[:, 0], _TRI_INDICES[:, 1], _TRI_INDICES[:, 2]
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    # area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume is area
    except Exception:
        return 1.0


def _score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        min_area = np.min(_all_triangle_areas(pts))
        return min_area / hull_area
    except Exception:
        return 0.0


def _soft_min_score(pts, temp=0.01):
    """Smooth surrogate: soft-min of triangle areas / hull area using log-sum-exp."""
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        areas = _all_triangle_areas(pts)
        # soft_min = -t * log(mean(exp(-areas/t)))
        # For stability: soft_min = max(areas) - t*log(mean(exp(-(areas-max)/t)))
        a_max = np.max(areas)
        exp_terms = np.exp(-(areas - a_max) / temp)
        soft_min_val = a_max - temp * np.log(np.mean(exp_terms))
        return soft_min_val / hull_area
    except Exception:
        return 0.0


def _soft_min_score_neg(pts, temp=0.01):
    """Negative soft-min score for minimization."""
    return -_soft_min_score(pts, temp)


def _angular_to_cartesian(params):
    """Convert (angles, radii) to (x, y) coordinates.
    params shape (26,): first 13 are angles, next 13 are radii.
    """
    angles = params[:N]
    radii = np.abs(params[N:])  # ensure non-negative
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def _cartesian_to_angular(pts):
    """Convert (x, y) to (angles, radii)."""
    x, y = pts[:, 0], pts[:, 1]
    radii = np.sqrt(x**2 + y**2)
    angles = np.arctan2(y, x)
    # Unwrap angles to [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)
    return np.concatenate([angles, radii])


def _angular_score(params, temp=0.01):
    """Score in angular parameterization (for minimization, returns negative score)."""
    pts = _angular_to_cartesian(params)
    return -_soft_min_score(pts, temp)


def _angular_score_exact(params):
    """Exact score in angular parameterization (for minimization)."""
    pts = _angular_to_cartesian(params)
    return -_score(pts)


def _full2d_score_neg(pts_flat, temp=0.01):
    """Full 2D score for L-BFGS-B (minimization form)."""
    pts = pts_flat.reshape(N, 2)
    return -_soft_min_score(pts, temp)


def _full2d_score_neg_exact(pts_flat):
    """Full 2D exact score for L-BFGS-B (minimization form)."""
    pts = pts_flat.reshape(N, 2)
    return -_score(pts)


def _generate_initializations():
    """Generate multiple initial configurations."""
    inits = []

    # 1. Regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts_circle)

    # 2. Halton sequence in unit square
    def _halton(n, base):
        result = np.zeros(n)
        f = 1.0
        for i in range(1, n + 1):
            x = 0.0
            i_temp = i
            while i_temp > 0:
                f /= base
                x += f * (i_temp % base)
                i_temp //= base
            result[i - 1] = x
        return result

    x = _halton(N, 2)
    y = _halton(N, 3)
    pts_halton = np.column_stack([x, y])
    inits.append(pts_halton)

    # 3. Halton on circle
    angles_h = 2 * np.pi * _halton(N, 2)
    radii_h = 0.5 + 0.5 * _halton(N, 3)
    pts_halton_c = np.column_stack([radii_h * np.cos(angles_h), radii_h * np.sin(angles_h)])
    inits.append(pts_halton_c)

    # 4. Concentric rings: 7 outer, 6 inner
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    inner = 0.45 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    pts_rings = np.vstack([outer, inner])
    inits.append(pts_rings)

    # 5. 9 boundary + 4 interior
    angles_b = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    b_pts = np.column_stack([np.cos(angles_b), np.sin(angles_b)])
    i_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    i_radii = np.array([0.3, 0.5, 0.3, 0.5])
    i_pts = np.column_stack([i_radii * np.cos(i_angles), i_radii * np.sin(i_angles)])
    pts_mixed = np.vstack([b_pts, i_pts])
    inits.append(pts_mixed)

    # 6. Random with fixed seed
    rng = np.random.RandomState(123)
    pts_rand = rng.random((N, 2))
    inits.append(pts_rand)

    # 7. 12 on circle + 1 center
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    circle_12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    center = np.array([[0.0, 0.0]])
    pts_center = np.vstack([circle_12, center])
    inits.append(pts_center)

    # 8. Structured: 13 points with varying radii
    angles_s = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii_s = 1.0 - 0.3 * np.sin(np.arange(N) * np.pi / 5)
    pts_struct = np.column_stack([radii_s * np.cos(angles_s), radii_s * np.sin(angles_s)])
    inits.append(pts_struct)

    return inits


def _lbfgs_angular(params_init, time_budget, start_time):
    """L-BFGS-B optimization in angular parameterization."""
    deadline = start_time + time_budget

    best_params = params_init.copy()
    best_val = _angular_score(params_init, temp=0.01)

    # Multi-temperature schedule
    temps = [0.05, 0.02, 0.005, 0.001]
    for temp in temps:
        if time.time() > deadline - 3:
            break

        def objective(p):
            if time.time() > deadline - 1:
                raise RuntimeError("time out")
            return _angular_score(p, temp=temp)

        try:
            result = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    return best_params, -best_val


def _lbfgs_full2d(pts_init, time_budget, start_time):
    """L-BFGS-B optimization in full 2D parameterization."""
    deadline = start_time + time_budget
    x0 = pts_init.flatten()

    best_x = x0.copy()
    best_val = _full2d_score_neg(x0, temp=0.01)

    temps = [0.02, 0.005, 0.001]
    for temp in temps:
        if time.time() > deadline - 2:
            break

        def objective(x):
            if time.time() > deadline - 1:
                raise RuntimeError("time out")
            return _full2d_score_neg(x, temp=temp)

        try:
            result = minimize(
                objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 8000}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass

    return best_x.reshape(N, 2), -best_val


def _worst_triangle_refinement(pts, time_budget, start_time):
    """Targeted refinement: identify worst (smallest area) triangles and do
    local L-BFGS-B focused on improving them."""
    deadline = start_time + time_budget
    best_pts = pts.copy()
    best_score = _score(pts)

    for iteration in range(5):
        if time.time() > deadline - 2:
            break

        areas = _all_triangle_areas(best_pts)
        hull_area = _hull_area(best_pts)
        min_area = np.min(areas)
        # Find triangles near the minimum (bottom 10%)
        threshold = np.percentile(areas, 10)
        worst_mask = areas <= threshold

        if not np.any(worst_mask):
            break

        # Get indices of worst triangles
        worst_indices = np.where(worst_mask)[0]
        worst_tri = _TRI_INDICES[worst_indices]

        # Build a focused objective that penalizes the worst triangles more
        def focused_objective(x):
            p = x.reshape(N, 2)
            a = _all_triangle_areas(p)
            ha = _hull_area(p)
            if ha < 1e-12:
                return 1.0
            # Penalize worst triangles heavily
            worst_areas = a[worst_mask]
            # Use soft-min on worst triangles with very small temp
            wm = np.min(worst_areas)
            penalty = -wm / ha
            # Plus regular soft-min on all
            all_sm = -_soft_min_score(p, temp=0.005)
            return 0.5 * penalty + 0.5 * all_sm

        x0 = best_pts.flatten()
        try:
            result = minimize(
                focused_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000}
            )
            if result.fun < focused_objective(x0):
                candidate = result.x.reshape(N, 2)
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_pts = candidate
        except Exception:
            pass

    return best_pts, best_score


def _nelder_mead_fallback(pts, time_budget, start_time):
    """Fallback Nelder-Mead if L-BFGS-B doesn't improve much."""
    deadline = start_time + time_budget
    n_params = N * 2
    x0 = pts.flatten()

    # Simplex
    simplex = [x0.copy()]
    for i in range(n_params):
        xi = x0.copy()
        step = 0.01
        xi[i] += step
        simplex.append(xi)

    scores = [_score(simplex[i].reshape(N, 2)) for i in range(n_params + 1)]
    best_score = max(scores)
    best_x = simplex[np.argmax(scores)].copy()

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    max_iter = min(2000, int(time_budget * 100))

    for it in range(max_iter):
        if time.time() > deadline - 1:
            break

        order = np.argsort(scores)[::-1]
        simplex = [simplex[i].copy() for i in order]
        scores = [scores[i] for i in order]

        worst = simplex[-1]
        worst_score = scores[-1]
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_ref = centroid + alpha * (centroid - worst)
        s_ref = _score(x_ref.reshape(N, 2))

        if scores[0] >= s_ref > worst_score:
            simplex[-1] = x_ref
            scores[-1] = s_ref
        elif s_ref >= scores[0]:
            # Expansion
            x_exp = centroid + gamma * (centroid - worst)
            s_exp = _score(x_exp.reshape(N, 2))
            if s_exp > s_ref:
                simplex[-1] = x_exp
                scores[-1] = s_exp
            else:
                simplex[-1] = x_ref
                scores[-1] = s_ref
        else:
            # Contraction
            x_con = centroid + rho * (worst - centroid)
            s_con = _score(x_con.reshape(N, 2))
            if s_con > worst_score:
                simplex[-1] = x_con
                scores[-1] = s_con
            else:
                # Shrink
                for i in range(1, n_params + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    scores[i] = _score(simplex[i].reshape(N, 2))

        cur_best = max(scores)
        if cur_best > best_score:
            best_score = cur_best
            best_x = simplex[np.argmax(scores)].copy()

        if abs(scores[0] - worst_score) < 1e-12:
            break

    return best_x.reshape(N, 2), best_score


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    inits = _generate_initializations()

    # Evaluate all initializations
    best_pts = None
    best_score = -1
    for init in inits:
        if time.time() > deadline - 60:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()

    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_pts)

    # Phase 1: Angular L-BFGS-B on each promising initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 40:
            break
        s_init = _score(init)
        if s_init < best_score * 0.4:
            continue

        remaining = deadline - 40 - time.time()
        if remaining < 5:
            break

        # Convert to angular params
        ang_params = _cartesian_to_angular(init)
        ang_params_opt, ang_score = _lbfgs_angular(ang_params, remaining, start_time)
        pts_opt = _angular_to_cartesian(ang_params_opt)

        # Also try full 2D L-BFGS-B from this starting point
        remaining2 = deadline - 40 - time.time()
        if remaining2 > 5:
            pts_opt2, score_opt2 = _lbfgs_full2d(pts_opt, remaining2, start_time)
            if score_opt2 > ang_score:
                ang_score = score_opt2
                pts_opt = pts_opt2

        if ang_score > best_score:
            best_score = ang_score
            best_pts = pts_opt

    # Phase 2: Full 2D L-BFGS-B polish on best
    time_left = deadline - 15 - time.time()
    if time_left > 5 and best_pts is not None:
        pts_polished, score_polished = _lbfgs_full2d(best_pts, time_left, start_time)
        if score_polished > best_score:
            best_score = score_polished
            best_pts = pts_polished

    # Phase 3: Worst-triangle targeted refinement
    time_left = deadline - 8 - time.time()
    if time_left > 3 and best_pts is not None:
        pts_refined, score_refined = _worst_triangle_refinement(best_pts, time_left, start_time)
        if score_refined > best_score:
            best_score = score_refined
            best_pts = pts_refined

    # Phase 4: Final Nelder-Mead fallback if time remains
    time_left = deadline - 3 - time.time()
    if time_left > 5 and best_pts is not None:
        pts_nm, score_nm = _nelder_mead_fallback(best_pts, time_left, start_time)
        if score_nm > best_score:
            best_score = score_nm
            best_pts = pts_nm

    # Final evaluation
    final_score = _score(best_pts)
    if final_score > best_score:
        best_score = final_score

    return best_pts
