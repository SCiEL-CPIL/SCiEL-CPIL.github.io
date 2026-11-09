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
    deadline = start_time + 145.0
    N = 13
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def time_left():
        return deadline - time.time()

    # === Normalization: fix p0=[0,0], p1=[1,0] -> 22 free params ===
    def params_to_points(params):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(N - 2, 2)
        return pts

    def points_to_params(pts):
        """Normalize: translate p0 to origin, rotate p1 to x-axis, scale |p1-p0|=1."""
        p0 = pts[0]
        p1 = pts[1]
        d = p1 - p0
        dist = np.linalg.norm(d)
        if dist < 1e-10:
            return np.zeros(2 * (N - 2))
        pts_t = pts - p0
        angle = np.arctan2(d[1], d[0])
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])
        pts_r = pts_t @ R.T
        pts_s = pts_r / dist
        return pts_s[2:].flatten()

    def smooth_objective(params, tau):
        """-soft_min(areas) / hull_area in normalized 22-param space."""
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
            return -score
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

    # === Direct structural approach: build configurations from geometric principles ===
    def make_regular_ngon(n, center=None, radius=1.0, phase=0.0):
        angles = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
        if center is None:
            center = np.array([0.0, 0.0])
        return np.column_stack((center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)))

    initial_configs = []

    # 1. Regular 13-gon
    initial_configs.append(make_regular_ngon(13, radius=1.0))

    # 2. 12-gon ring + 1 center point
    ring12 = make_regular_ngon(12, radius=1.0)
    initial_configs.append(np.vstack([ring12, np.array([[0.0, 0.0]])]))

    # 3. 11-gon ring + 2 points (one center, one offset)
    ring11 = make_regular_ngon(11, radius=1.0)
    interior2 = np.array([[0.0, 0.0], [0.3, 0.0]])
    initial_configs.append(np.vstack([ring11, interior2]))

    # 4. 10-gon ring + 3 inner (triangle pattern)
    ring10 = make_regular_ngon(10, radius=1.0)
    inner3 = make_regular_ngon(3, radius=0.45, phase=np.pi/6)
    initial_configs.append(np.vstack([ring10, inner3]))

    # 5. 9-gon ring + 4 inner (square pattern)
    ring9 = make_regular_ngon(9, radius=1.0)
    inner4 = make_regular_ngon(4, radius=0.45, phase=np.pi/4)
    initial_configs.append(np.vstack([ring9, inner4]))

    # 6. 8-gon ring + 5 inner (pentagon pattern)
    ring8 = make_regular_ngon(8, radius=1.0)
    inner5 = make_regular_ngon(5, radius=0.5, phase=np.pi/5)
    initial_configs.append(np.vstack([ring8, inner5]))

    # 7. 7-gon ring + 6 inner (hexagon pattern)
    ring7 = make_regular_ngon(7, radius=1.0)
    inner6 = make_regular_ngon(6, radius=0.5, phase=0.0)
    initial_configs.append(np.vstack([ring7, inner6]))

    # 8. Two concentric rings: 7 outer + 6 inner
    outer7 = make_regular_ngon(7, radius=1.0)
    inner6 = make_regular_ngon(6, radius=0.55, phase=np.pi/6)
    initial_configs.append(np.vstack([outer7, inner6]))

    # 9. Three layers: 6 outer + 5 middle + 2 inner
    outer6 = make_regular_ngon(6, radius=1.0)
    middle5 = make_regular_ngon(5, radius=0.65, phase=np.pi/5)
    inner2 = np.array([[0.0, 0.0], [0.2, 0.0]])
    initial_configs.append(np.vstack([outer6, middle5, inner2]))

    # 10. Perturbed regular 13-gon
    base = make_regular_ngon(13, radius=1.0)
    perturbed = base + rng.normal(0, 0.03, base.shape)
    initial_configs.append(perturbed)

    # 11. Slightly elliptical 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ellipse = np.column_stack((1.05 * np.cos(angles), 0.95 * np.sin(angles)))
    initial_configs.append(ellipse)

    # 12. 12-gon with center and one vertex slightly inward
    ring12c = make_regular_ngon(12, radius=1.0)
    ring12c[0] = ring12c[0] * 0.92
    initial_configs.append(np.vstack([ring12c, np.array([[0.0, 0.0]])]))

    # 13. 13 points on two circles (radial optimization)
    for split in [6, 7, 8]:
        for r_inner in [0.4, 0.5, 0.6, 0.7]:
            n_outer = split
            n_inner = N - n_outer
            outer = make_regular_ngon(n_outer, radius=1.0)
            inner = make_regular_ngon(n_inner, radius=r_inner, phase=np.pi/n_inner)
            initial_configs.append(np.vstack([outer, inner]))

    # Evaluate all initial configs
    for cfg in initial_configs:
        update_best(cfg)

    # === New: Polar Coordinate Optimization with Angular Regularization ===
    # Model points as (r_i, theta_i). Fix theta_0 = 0.
    # Params: r_0..r_12 (13), theta_1..theta_12 (12). Total 25.
    
    def polar_to_points(params):
        r = params[:N]
        theta = np.zeros(N)
        theta[0] = 0.0
        theta[1:] = params[N:]
        # Ensure radii are positive
        r = np.maximum(r, 1e-6)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))

    def points_to_polar(pts):
        # Translate to center of mass? Or just origin?
        # The score is invariant to translation. Let's center at origin for stability.
        pts_c = pts - np.mean(pts, axis=0)
        r = np.sqrt(pts_c[:, 0]**2 + pts_c[:, 1]**2)
        theta = np.arctan2(pts_c[:, 1], pts_c[:, 0])
        # Unwrap angles to be close to ideal
        ideal_theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        # Adjust theta to be in [0, 2pi) and close to ideal order
        # This is tricky. Let's just use the raw angles and let the penalty guide them.
        # However, we fixed theta[0]=0 in polar_to_points.
        # So we need to rotate the whole set so that the first point is at angle 0?
        # Or just pick the first point as reference.
        
        # For the initial guess, we will just use the regular polygon.
        # For optimization, we assume the topology is preserved (no crossing).
        
        r_out = np.zeros(N)
        theta_out = np.zeros(N)
        r_out[0] = r[0]
        theta_out[0] = 0.0
        # The rest: we need to map theta[1:] to the relative angles.
        # This is ambiguous if the points cross.
        # Given the "regular polygon relaxation" strategy, we assume they don't cross.
        # We will just return the raw angles for the initial guess, and rely on the optimizer.
        # But the parameter space expects theta[1:] to be the actual angles.
        # If we start from a regular polygon, theta[i] = 2*pi*i/N.
        # So initial params: r=1, theta[1:]=2*pi*i/N.
        
        # We only use this function for debugging or if we want to restart from a cartesian point.
        # For now, we will only start from the regular polygon in polar space.
        return None # Not used for initialization in this block, just for reference

    def polar_objective(params, angular_penalty_weight):
        try:
            pts = polar_to_points(params)
            areas = compute_all_areas(pts)
            sm = soft_min(areas, 200.0) # High tau for sharpness
            try:
                hull = ConvexHull(pts)
                hull_area = hull.volume
                if hull_area < 1e-10:
                    return 1e6
            except:
                return 1e6
            
            score = sm / hull_area
            
            # Angular penalty: sum of (theta_i - ideal_i)^2
            ideal_theta = np.linspace(0, 2*np.pi, N, endpoint=False)
            theta = np.zeros(N)
            theta[0] = 0.0
            theta[1:] = params[N:]
            
            # Handle angle wrapping for penalty
            # We want theta_i to be close to ideal_theta_i
            # Since we start near regular polygon, no wrapping issues expected initially.
            # If they wrap, the penalty will be large, pushing them back.
            diff = theta - ideal_theta
            # Wrap diff to [-pi, pi]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            angular_penalty = np.sum(diff**2)
            
            return -score + angular_penalty_weight * angular_penalty
        except:
            return 1e6

    # Run polar optimization
    if time_left() > 30.0:
        # Initialize with regular polygon
        r_init = np.ones(N)
        theta_init = np.linspace(0, 2*np.pi, N, endpoint=False)
        theta_init[0] = 0.0
        x0 = np.concatenate((r_init, theta_init[1:]))
        
        # Try different angular penalty weights
        for weight in [0.01, 0.1, 1.0]:
            if time_left() < 15.0:
                break
            try:
                res = minimize(polar_objective, x0, args=(weight,), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-8})
                if res.fun < 1e5:
                    pts = polar_to_points(res.x)
                    update_best(pts)
                    # If improved, use as new start
                    if best_points is not None and get_score(pts) > best_score - 1e-6:
                         # We can't easily update x0 from best_points to polar without robust mapping.
                         # So we just stick with the current x0 or reset to regular if it failed.
                         pass
            except:
                pass

    # === Optimize each configuration in normalized 22-param space ===
    # Prioritize configs by their initial score
    config_scores = []
    for cfg in initial_configs:
        s = get_score(cfg)
        config_scores.append((s, cfg))
    config_scores.sort(key=lambda x: -x[0])

    for score_val, cfg in config_scores:
        if time_left() < 15.0:
            break

        x0 = points_to_params(cfg)

        # Progressive tau: start loose, tighten
        taus = [50, 100, 200, 300]

        for tau in taus:
            if time_left() < 10.0:
                break

            # SLSQP (uses gradient)
            try:
                res = minimize(smooth_objective, x0, args=(tau,), method='SLSQP',
                               options={'maxiter': 200, 'ftol': 1e-9})
                if res.fun < 1e5:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res.x)
                        x0 = res.x
            except:
                pass

            # L-BFGS-B for refinement
            if time_left() < 5.0:
                continue
            try:
                res2 = minimize(smooth_objective, x0, args=(tau,), method='L-BFGS-B',
                                options={'maxiter': 150, 'ftol': 1e-11})
                if res2.fun < 1e5:
                    current_score = -res2.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res2.x)
                        x0 = res2.x
            except:
                pass

        # Final high-tau refinement for this config
        if time_left() > 12.0:
            try:
                res_final = minimize(smooth_objective, x0, args=(400,), method='L-BFGS-B',
                                     options={'maxiter': 100, 'ftol': 1e-12})
                if res_final.fun < 1e5:
                    current_score = -res_final.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res_final.x)
            except:
                pass

    # === Global Refinement on the best point found so far ===
    if best_points is not None and time_left() > 25.0:
        x_best = points_to_params(best_points)
        
        # Nelder-Mead with high tau for sharpness
        try:
            res_nm = minimize(smooth_objective, x_best, args=(400,), method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res_nm.fun < 1e5:
                pts = params_to_points(res_nm.x)
                update_best(pts)
        except:
            pass

    # === Powell refinement ===
    if best_points is not None and time_left() > 20.0:
        x_best = points_to_params(best_points)
        
        try:
            res_pw = minimize(smooth_objective, x_best, args=(400,), method='Powell',
                             options={'maxiter': 1500, 'ftol': 1e-10, 'xtol': 1e-10})
            if res_pw.fun < 1e5:
                pts = params_to_points(res_pw.x)
                update_best(pts)
        except:
            pass

    # === Local refinement with small perturbations around the best point ===
    if best_points is not None and time_left() > 20.0:
        x_best = points_to_params(best_points)

        # Try several perturbation scales
        for pert_scale in [0.003, 0.005, 0.01, 0.02]:
            if time_left() < 10.0:
                break
            for _ in range(6):
                if time_left() < 5.0:
                    break
                x_pert = x_best + rng.normal(0, pert_scale, len(x_best))
                try:
                    res_p = minimize(smooth_objective, x_pert, args=(400,),
                                     method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
                    if res_p.fun < 1e5:
                        pts = params_to_points(res_p.x)
                        update_best(pts)
                        if get_score(pts) > best_score:
                            x_best = res_p.x
                except:
                    pass

    # Final fallback: regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = get_score(best_points)

    return best_points
