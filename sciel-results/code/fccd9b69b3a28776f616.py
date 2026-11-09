import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 118.0  # safety margin before 120s limit

# Precompute all triple indices for speed
_ALL_TRIPLES = np.array(list(combinations(range(N), 3)))
_NUM_TRIPLES = _ALL_TRIPLES.shape[0]

def _all_triangle_areas(points):
    """Compute all triangle areas as a vector."""
    p1 = points[_ALL_TRIPLES[:, 0]]
    p2 = points[_ALL_TRIPLES[:, 1]]
    p3 = points[_ALL_TRIPLES[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    if len(points) < 3:
        return 0.0
    return float(np.min(_all_triangle_areas(points)))

def _hull_area(points):
    """Compute the convex hull area of the point set."""
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 1.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return _min_triangle_area(points) / hull_area

def _soft_min_triangle_area(points, beta=200.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp (soft-min).
    soft_min(a) = -(1/beta) * log(sum(exp(-beta * a_i)))
    As beta -> inf, this approaches the true min.
    """
    areas = _all_triangle_areas(points)
    areas = np.clip(areas, 1e-15, None)
    neg_beta_a = -beta * areas
    max_val = np.max(neg_beta_a)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_a - max_val)))
    soft_min = -log_sum / beta
    return float(soft_min)

def _soft_objective(x, beta=200.0):
    """
    Smooth objective: -soft_min_triangle_area / hull_area.
    Returns a smooth scalar to be minimized.
    """
    pts = x.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-8:
        return 1e10
    sm = _soft_min_triangle_area(pts, beta=beta)
    if sm < 1e-15:
        return 1e10
    return -sm / hull_area

def _exact_objective(x):
    """Exact nonsmooth objective for final polish."""
    pts = x.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-8:
        return 1e10
    min_area = _min_triangle_area(pts)
    return -min_area / hull_area

def _regular_ngon(n, radius=1.0, phase=0.0, center=None):
    """Generate n points on a regular polygon."""
    angles = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
    if center is None:
        center = np.array([0.0, 0.0])
    return np.column_stack((center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)))

def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf
    rng = np.random.RandomState(42)

    def _time_remaining():
        return DEADLINE - (time.time() - start_time)

    def _update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        try:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # === Build diverse initial configurations ===
    configs = []

    # 1. Regular 13-gon
    configs.append(_regular_ngon(13, radius=1.0))

    # 2. 12-gon + center
    ring12 = _regular_ngon(12, radius=1.0)
    configs.append(np.vstack([ring12, np.array([[0.0, 0.0]])]))

    # 3. 11-gon + 2 interior
    ring11 = _regular_ngon(11, radius=1.0)
    interior2 = np.array([[0.0, 0.0], [0.3, 0.0]])
    configs.append(np.vstack([ring11, interior2]))

    # 4. 10-gon + 3 inner (triangle)
    ring10 = _regular_ngon(10, radius=1.0)
    inner3 = _regular_ngon(3, radius=0.45, phase=np.pi/6)
    configs.append(np.vstack([ring10, inner3]))

    # 5. 9-gon + 4 inner (square)
    ring9 = _regular_ngon(9, radius=1.0)
    inner4 = _regular_ngon(4, radius=0.45, phase=np.pi/4)
    configs.append(np.vstack([ring9, inner4]))

    # 6. 8-gon + 5 inner (pentagon)
    ring8 = _regular_ngon(8, radius=1.0)
    inner5 = _regular_ngon(5, radius=0.5, phase=np.pi/5)
    configs.append(np.vstack([ring8, inner5]))

    # 7. 7-gon + 6 inner (hexagon)
    ring7 = _regular_ngon(7, radius=1.0)
    inner6 = _regular_ngon(6, radius=0.5, phase=0.0)
    configs.append(np.vstack([ring7, inner6]))

    # 8. Two concentric rings: 7 outer + 6 inner
    outer7 = _regular_ngon(7, radius=1.0)
    inner6 = _regular_ngon(6, radius=0.55, phase=np.pi/6)
    configs.append(np.vstack([outer7, inner6]))

    # 9. Three layers: 6 outer + 5 middle + 2 inner
    outer6 = _regular_ngon(6, radius=1.0)
    middle5 = _regular_ngon(5, radius=0.65, phase=np.pi/5)
    inner2 = np.array([[0.0, 0.0], [0.2, 0.0]])
    configs.append(np.vstack([outer6, middle5, inner2]))

    # 10. Perturbed regular 13-gon
    base = _regular_ngon(13, radius=1.0)
    perturbed = base + rng.normal(0, 0.03, base.shape)
    configs.append(perturbed)

    # 11. Slightly elliptical 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ellipse = np.column_stack((1.05 * np.cos(angles), 0.95 * np.sin(angles)))
    configs.append(ellipse)

    # 12. 12-gon with center and one vertex slightly inward
    ring12c = _regular_ngon(12, radius=1.0)
    ring12c[0] = ring12c[0] * 0.92
    configs.append(np.vstack([ring12c, np.array([[0.0, 0.0]])]))

    # 13. Two circles with different splits
    for split in [6, 7, 8]:
        for r_inner in [0.4, 0.5, 0.6, 0.7]:
            n_outer = split
            n_inner = N - n_outer
            outer = _regular_ngon(n_outer, radius=1.0)
            inner = _regular_ngon(n_inner, radius=r_inner, phase=np.pi/n_inner)
            configs.append(np.vstack([outer, inner]))

    # 14. 5 outer + 8 inner
    outer5 = _regular_ngon(5, radius=1.0)
    inner8 = _regular_ngon(8, radius=0.55, phase=np.pi/8)
    configs.append(np.vstack([outer5, inner8]))

    # 15. 4 outer + 9 inner
    outer4 = _regular_ngon(4, radius=1.0)
    inner9 = _regular_ngon(9, radius=0.55, phase=np.pi/9)
    configs.append(np.vstack([outer4, inner9]))

    # 16. Three rings: 5 + 4 + 4
    r1 = _regular_ngon(5, radius=1.0)
    r2 = _regular_ngon(4, radius=0.65, phase=np.pi/4)
    r3 = _regular_ngon(4, radius=0.35, phase=np.pi/4 + np.pi/8)
    configs.append(np.vstack([r1, r2, r3]))

    # 17. Three rings: 6 + 4 + 3
    r1 = _regular_ngon(6, radius=1.0)
    r2 = _regular_ngon(4, radius=0.6, phase=np.pi/4)
    r3 = _regular_ngon(3, radius=0.3, phase=0.0)
    configs.append(np.vstack([r1, r2, r3]))

    # 18. Three rings: 4 + 5 + 4
    r1 = _regular_ngon(4, radius=1.0)
    r2 = _regular_ngon(5, radius=0.65, phase=np.pi/5)
    r3 = _regular_ngon(4, radius=0.35, phase=np.pi/4)
    configs.append(np.vstack([r1, r2, r3]))

    # 19. Golden ratio spiral
    theta = np.linspace(0, 4 * np.pi, N)
    r = 0.9 * np.sqrt(np.linspace(0.1, 1.0, N))
    spiral = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    configs.append(spiral)

    # 20. Fibonacci spiral
    phi = (1 + np.sqrt(5)) / 2
    theta = np.arange(N) * 2 * np.pi / phi
    r = 0.9 * np.sqrt(np.linspace(0.05, 1.0, N))
    fib_spiral = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    configs.append(fib_spiral)

    # 21. Offset centers for 12-gon
    for dx in [-0.1, 0.1, -0.05, 0.05, 0.0]:
        for dy in [-0.1, 0.1, -0.05, 0.05, 0.0]:
            if dx == 0 and dy == 0:
                continue
            ring12_off = _regular_ngon(12, radius=1.0)
            center = np.array([dx, dy])
            configs.append(np.vstack([ring12_off, center]))

    # 22. Perturbed 12-gons with varying perturbations
    for perturb in [0.01, 0.02, 0.03, 0.05, 0.08]:
        for seed in [1, 2, 3, 4, 5]:
            rng2 = np.random.RandomState(seed)
            ring12 = _regular_ngon(12, radius=1.0)
            ring12_pert = ring12 + rng2.randn(12, 2) * perturb
            configs.append(np.vstack([ring12_pert, np.array([[0.0, 0.0]])]))

    # 23. Random near 12-gon with varying radii
    for seed in range(20):
        rng2 = np.random.RandomState(seed)
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rng2.randn(12) * 0.02
        radii = 1.0 + rng2.randn(12) * 0.06
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        center = rng2.randn(2) * 0.06
        configs.append(np.vstack([pts, center]))

    # Evaluate all initial configs
    for cfg in configs:
        _update_best(cfg)

    # === Phase 1: Progressive soft-min optimization ===
    # Sort configs by score and optimize top ones
    config_scores = []
    for cfg in configs:
        s = _score(cfg)
        config_scores.append((s, cfg))
    config_scores.sort(key=lambda x: -x[0])

    # Limit number of configs to optimize
    max_configs = 10
    configs_to_optimize = config_scores[:max_configs]

    # Progressive beta values for soft-min
    betas = [50.0, 100.0, 200.0, 400.0]

    for score_val, cfg in configs_to_optimize:
        if _time_remaining() < 20.0:
            break

        x0 = cfg.flatten()

        for beta in betas:
            if _time_remaining() < 12.0:
                break

            # L-BFGS-B with soft-min objective
            try:
                result = minimize(
                    _soft_objective, x0, args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < 1e5:
                    opt_pts = result.x.reshape(-1, 2)
                    opt_score = _score(opt_pts)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_pts.copy()
                        x0 = result.x
            except Exception:
                pass

            # SLSQP refinement
            if _time_remaining() < 8.0:
                continue
            try:
                result2 = minimize(
                    _soft_objective, x0, args=(beta,),
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                if result2.fun < 1e5:
                    opt_pts = result2.x.reshape(-1, 2)
                    opt_score = _score(opt_pts)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_pts.copy()
                        x0 = result2.x
            except Exception:
                pass

    # === Phase 2: Nelder-Mead polish with exact objective ===
    if best_points is not None and _time_remaining() > 15.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, x0, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            opt_pts = result.x.reshape(-1, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # === Phase 3: Additional Nelder-Mead from other good starts ===
    if _time_remaining() > 10.0:
        for s, cfg in config_scores[:5]:
            if _time_remaining() < 5.0:
                break
            x0 = cfg.flatten()
            try:
                result = minimize(
                    _exact_objective, x0, method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                opt_pts = result.x.reshape(-1, 2)
                opt_score = _score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass

    # === Phase 4: Final high-precision L-BFGS-B polish with very sharp soft-min ===
    if best_points is not None and _time_remaining() > 5.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _soft_objective, x0, args=(1000.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < 1e5:
                opt_pts = result.x.reshape(-1, 2)
                opt_score = _score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
        except Exception:
            pass

    # === Phase 5: Perturb-and-reoptimize loop ===
    if best_points is not None and _time_remaining() > 20.0:
        x_best = best_points.flatten()
        for pert_scale in [0.002, 0.005, 0.01, 0.02, 0.03]:
            if _time_remaining() < 10.0:
                break
            for _ in range(5):
                if _time_remaining() < 5.0:
                    break
                x_pert = x_best + rng.normal(0, pert_scale, len(x_best))
                try:
                    res_p = minimize(
                        _soft_objective, x_pert, args=(500.0,),
                        method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-13}
                    )
                    if res_p.fun < 1e5:
                        pts = res_p.x.reshape(-1, 2)
                        s = _score(pts)
                        if s > best_score:
                            best_score = s
                            best_points = pts.copy()
                            x_best = res_p.x
                except Exception:
                    pass

    # === Phase 6: Powell refinement ===
    if best_points is not None and _time_remaining() > 15.0:
        x_best = best_points.flatten()
        try:
            res_pw = minimize(
                _soft_objective, x_best, args=(500.0,),
                method='Powell',
                options={'maxiter': 2000, 'ftol': 1e-11, 'xtol': 1e-11}
            )
            if res_pw.fun < 1e5:
                pts = res_pw.x.reshape(-1, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass

    # === Final high-precision polish ===
    if best_points is not None and _time_remaining() > 10.0:
        x_best = best_points.flatten()
        try:
            res_polish = minimize(
                _soft_objective, x_best, args=(600.0,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_polish.fun < 1e5:
                pts = res_polish.x.reshape(-1, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass

    # Final fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_points)

    return best_points
