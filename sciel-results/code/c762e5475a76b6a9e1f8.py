import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def compute_min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def get_score(points):
    """Compute score: min triangle area / convex hull area."""
    N = len(points)
    if N < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except:
        return 0.0
    min_area = compute_min_triangle_area(points)
    if min_area <= 0:
        return 0.0
    return min_area / hull_area

def soft_min(areas, tau):
    """Soft minimum with log-sum-exp stability and clipping."""
    min_val = np.min(areas)
    shifted = areas - min_val
    shifted = np.maximum(shifted, -50.0 / tau)
    log_sum = np.log(np.sum(np.exp(-tau * shifted)))
    return min_val - (1.0 / tau) * log_sum

def compute_all_areas(points):
    """Compute all triangle areas for n points (vectorized)."""
    n = len(points)
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def construct():
    start_time = time.time()
    deadline = start_time + 142.0
    N = 13
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def time_left():
        return deadline - time.time()

    def params_to_points(params):
        """Convert radial + angular perturbation params to points."""
        # params: [r_0..r_12, delta_theta_0..delta_theta_12] (26 params)
        radii = params[:N]
        dtheta = params[N:]
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles = base_angles + dtheta
        # Enforce angular ordering by sorting
        order = np.argsort(angles)
        angles = np.sort(angles)
        radii = radii[order]
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        return pts

    def points_to_params(pts):
        """Convert points to radial + angular perturbation params."""
        # Center at origin
        pts_c = pts - np.mean(pts, axis=0)
        radii = np.linalg.norm(pts_c, axis=1)
        angles = np.arctan2(pts_c[:, 1], pts_c[:, 0])
        # Sort by angle
        order = np.argsort(angles)
        radii = radii[order]
        angles = angles[order]
        # Compute angular deviations from regular spacing
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        dtheta = angles - base_angles
        # Unwrap to handle wrap-around
        dtheta = np.unwrap(dtheta)
        return np.concatenate([radii, dtheta])

    def smooth_objective(params, tau):
        try:
            pts = params_to_points(params)
            areas = compute_all_areas(pts)
            sm = soft_min(areas, tau)
            try:
                hull = ConvexHull(pts)
                hull_area = hull.volume
                if hull_area < 1e-10:
                    return 1e6
            except:
                return 1e6
            score = sm / hull_area
            # Barrier for angular spacing: penalize deviations from uniform spacing
            radii = params[:N]
            dtheta = params[N:]
            # Penalize large angular deviations
            angular_penalty = np.sum(dtheta ** 2) * 0.01
            # Penalize very small radii (points too close to center)
            radial_penalty = np.sum(np.maximum(0, 0.3 - radii) ** 2) * 0.1
            return -score + angular_penalty + radial_penalty
        except:
            return 1e6

    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        try:
            s = get_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except:
            pass

    def make_regular_ngon(n, center=None, radius=1.0, phase=0.0):
        angles = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
        if center is None:
            center = np.array([0.0, 0.0])
        return np.column_stack((center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)))

    # Generate diverse initial configurations
    initial_configs = []

    # 1. Regular 13-gon
    initial_configs.append(make_regular_ngon(13, radius=1.0))

    # 2. Radial jitter: alternating radii
    radii_alt = np.array([1.0, 0.85] * 6 + [1.0])
    angles13 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    alt_pts = np.column_stack((radii_alt * np.cos(angles13), radii_alt * np.sin(angles13)))
    initial_configs.append(alt_pts)

    # 3. Radial jitter: random radii
    radii_rand = 1.0 + rng.normal(0, 0.1, N)
    pts_rand = np.column_stack((radii_rand * np.cos(angles13), radii_rand * np.sin(angles13)))
    initial_configs.append(pts_rand)

    # 4. Angular perturbation: small random angular shifts
    dtheta_rand = rng.normal(0, 0.05, N)
    angles_pert = angles13 + dtheta_rand
    pts_pert = np.column_stack((np.cos(angles_pert), np.sin(angles_pert)))
    initial_configs.append(pts_pert)

    # 5. Combined: radial + angular perturbations
    radii_comb = 1.0 + rng.normal(0, 0.08, N)
    angles_comb = angles13 + rng.normal(0, 0.04, N)
    pts_comb = np.column_stack((radii_comb * np.cos(angles_comb), radii_comb * np.sin(angles_comb)))
    initial_configs.append(pts_comb)

    # 6. Two-ring: 7 outer + 6 inner
    outer7 = make_regular_ngon(7, radius=1.0)
    inner6 = make_regular_ngon(6, radius=0.55, phase=np.pi / 6)
    initial_configs.append(np.vstack([outer7, inner6]))

    # 7. Three-ring: 5 outer + 5 middle + 3 inner
    outer5 = make_regular_ngon(5, radius=1.0)
    mid5 = make_regular_ngon(5, radius=0.65, phase=np.pi / 5)
    inner3 = make_regular_ngon(3, radius=0.35)
    initial_configs.append(np.vstack([outer5, mid5, inner3]))

    # 8. Perturbed regular polygon (Cartesian noise)
    base = make_regular_ngon(13, radius=1.0)
    perturbed = base + rng.normal(0, 0.05, base.shape)
    initial_configs.append(perturbed)

    # 9. Elliptical distribution
    ellipse = np.column_stack((1.2 * np.cos(angles13), 0.8 * np.sin(angles13)))
    initial_configs.append(ellipse)

    # 10. Superellipse-like shape
    n_sup = 4
    sup_x = np.sign(np.cos(angles13)) * np.abs(np.cos(angles13)) ** (2.0 / n_sup)
    sup_y = np.sign(np.sin(angles13)) * np.abs(np.sin(angles13)) ** (2.0 / n_sup)
    sup_pts = np.column_stack((sup_x, sup_y))
    initial_configs.append(sup_pts)

    # 11. 12-gon + center
    ring12 = make_regular_ngon(12, radius=1.0)
    initial_configs.append(np.vstack([ring12, np.array([[0.0, 0.0]])]))

    # 12. 11-gon + 2 interior
    ring11 = make_regular_ngon(11, radius=1.0)
    interior2 = np.array([[0.3, 0.0], [-0.3, 0.0]])
    initial_configs.append(np.vstack([ring11, interior2]))

    # 13. 10-gon + 3 inner triangle
    outer10 = make_regular_ngon(10, radius=1.0)
    inner3b = make_regular_ngon(3, radius=0.45)
    initial_configs.append(np.vstack([outer10, inner3b]))

    # 14. 9-gon + 4 inner square
    outer9 = make_regular_ngon(9, radius=1.0)
    inner4 = np.array([[0.45, 0.45], [-0.45, 0.45], [-0.45, -0.45], [0.45, -0.45]])
    initial_configs.append(np.vstack([outer9, inner4]))

    # Evaluate all initial configs
    for cfg in initial_configs:
        update_best(cfg)

    # Sort configs by initial score to prioritize promising ones
    config_scores = []
    for cfg in initial_configs:
        try:
            s = get_score(cfg)
            config_scores.append((s, cfg))
        except:
            config_scores.append((0.0, cfg))
    config_scores.sort(key=lambda x: -x[0])

    # === Optimize each configuration in radial+angular param space (26 params) ===
    for score_init, cfg in config_scores:
        if time_left() < 12.0:
            break

        x0 = points_to_params(cfg)

        # Progressive tau: start loose, tighten
        taus = [50, 100, 200, 400, 800, 1500]

        for tau in taus:
            if time_left() < 6.0:
                break

            # L-BFGS-B (primary, uses gradient)
            try:
                res = minimize(smooth_objective, x0, args=(tau,), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12})
                if res.fun < 1e5:
                    # Get actual score (not including penalty)
                    pts = params_to_points(res.x)
                    s = get_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
                        x0 = res.x
            except:
                pass

            # SLSQP as secondary
            if time_left() < 4.0:
                continue
            try:
                res2 = minimize(smooth_objective, x0, args=(tau,), method='SLSQP',
                                options={'maxiter': 100, 'ftol': 1e-10})
                if res2.fun < 1e5:
                    pts = params_to_points(res2.x)
                    s = get_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
                        x0 = res2.x
            except:
                pass

        # Final high-tau refinement for this config
        if time_left() > 8.0:
            try:
                res_final = minimize(smooth_objective, x0, args=(3000,), method='L-BFGS-B',
                                     options={'maxiter': 150, 'ftol': 1e-14})
                if res_final.fun < 1e5:
                    pts = params_to_points(res_final.x)
                    s = get_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
            except:
                pass

    # === Global Refinement on the best point found so far ===
    if best_points is not None and time_left() > 30.0:
        x_best = points_to_params(best_points)

        # L-BFGS-B with very high tau (near-exact)
        try:
            res_lbfgs = minimize(smooth_objective, x_best, args=(5000,), method='L-BFGS-B',
                                 options={'maxiter': 300, 'ftol': 1e-15})
            if res_lbfgs.fun < 1e5:
                pts = params_to_points(res_lbfgs.x)
                update_best(pts)
        except:
            pass

        # Nelder-Mead with high tau
        if time_left() > 15.0:
            try:
                res_nm = minimize(smooth_objective, x_best, args=(2000,), method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res_nm.fun < 1e5:
                    pts = params_to_points(res_nm.x)
                    update_best(pts)
            except:
                pass

    # === Exact local search (nonsmooth min) ===
    if best_points is not None and time_left() > 15.0:
        def exact_objective(params):
            try:
                pts = params_to_points(params)
                areas = compute_all_areas(pts)
                min_area = np.min(areas)
                try:
                    hull = ConvexHull(pts)
                    hull_area = hull.volume
                    if hull_area < 1e-10:
                        return 1e6
                except:
                    return 1e6
                return -min_area / hull_area
            except:
                return 1e6

        x_best = points_to_params(best_points)

        # Nelder-Mead on exact (nonsmooth) objective
        try:
            res_exact = minimize(exact_objective, x_best, method='Nelder-Mead',
                                options={'maxiter': 4000, 'xatol': 1e-13, 'fatol': 1e-13})
            if res_exact.fun < 1e5:
                pts = params_to_points(res_exact.x)
                update_best(pts)
        except:
            pass

        # Powell on exact objective
        if time_left() > 8.0:
            try:
                res_pw_exact = minimize(exact_objective, x_best, method='Powell',
                                       options={'maxiter': 1500, 'ftol': 1e-14, 'xtol': 1e-14})
                if res_pw_exact.fun < 1e5:
                    pts = params_to_points(res_pw_exact.x)
                    update_best(pts)
            except:
                pass

    # === Local refinement with small perturbations around the best point ===
    if best_points is not None and time_left() > 10.0:
        x_best = points_to_params(best_points)

        for pert_scale in [0.002, 0.005, 0.01, 0.02]:
            if time_left() < 5.0:
                break
            for _ in range(3):
                if time_left() < 3.0:
                    break
                x_pert = x_best + rng.normal(0, pert_scale, len(x_best))
                try:
                    res_p = minimize(smooth_objective, x_pert, args=(3000,),
                                     method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-14})
                    if res_p.fun < 1e5:
                        pts = params_to_points(res_p.x)
                        update_best(pts)
                        if best_points is not None and get_score(best_points) > best_score:
                            x_best = points_to_params(best_points)
                except:
                    pass

    # Final fallback: regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = get_score(best_points)

    return best_points
