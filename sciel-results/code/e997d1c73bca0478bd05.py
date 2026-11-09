import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

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
    deadline = start_time + 140.0
    N = 13
    best_score = 0.0
    best_points = None

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

    # === Generate diverse initial configurations ===
    def make_regular_ngon(n, center=None, radius=1.0, phase=0.0):
        angles = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
        if center is None:
            center = np.array([0.0, 0.0])
        return np.column_stack((center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)))

    initial_configs = []

    # 1. Regular 13-gon
    initial_configs.append(make_regular_ngon(13, radius=1.0))

    # 2. Regular 12-gon + center
    ring12 = make_regular_ngon(12, radius=1.0)
    initial_configs.append(np.vstack([ring12, np.array([[0.0, 0.0]])]))

    # 3. Regular 11-gon + 2 interior
    ring11 = make_regular_ngon(11, radius=1.0)
    interior2 = np.array([[0.3, 0.0], [-0.3, 0.0]])
    initial_configs.append(np.vstack([ring11, interior2]))

    # 4. 7 outer + 6 inner concentric
    outer7 = make_regular_ngon(7, radius=1.0)
    inner6 = make_regular_ngon(6, radius=0.55, phase=np.pi / 6)
    initial_configs.append(np.vstack([outer7, inner6]))

    # 5. 9 outer + 4 inner (square)
    outer9 = make_regular_ngon(9, radius=1.0)
    inner4 = np.array([[0.45, 0.45], [-0.45, 0.45], [-0.45, -0.45], [0.45, -0.45]])
    initial_configs.append(np.vstack([outer9, inner4]))

    # 6. 10 outer + 3 inner (triangle)
    outer10 = make_regular_ngon(10, radius=1.0)
    inner3 = make_regular_ngon(3, radius=0.45)
    initial_configs.append(np.vstack([outer10, inner3]))

    # 7. 8 outer + 5 inner
    outer8 = make_regular_ngon(8, radius=1.0)
    inner5 = make_regular_ngon(5, radius=0.5, phase=np.pi / 5)
    initial_configs.append(np.vstack([outer8, inner5]))

    # 8. Delaunay-inspired: start from a slightly perturbed regular polygon
    #    and use the Delaunay triangulation to guide a mesh-like distribution
    base = make_regular_ngon(13, radius=1.0)
    rng = np.random.RandomState(42)
    perturbed = base + rng.normal(0, 0.05, base.shape)
    initial_configs.append(perturbed)

    # Evaluate all initial configs
    for cfg in initial_configs:
        update_best(cfg)

    # === Optimize each configuration in normalized 22-param space ===
    for cfg_idx, cfg in enumerate(initial_configs):
        if time_left() < 12.0:
            break

        x0 = points_to_params(cfg)

        # Progressive tau: start loose, tighten
        taus = [30, 60, 100, 150]

        for tau in taus:
            if time_left() < 8.0:
                break

            # SLSQP (uses gradient)
            try:
                res = minimize(smooth_objective, x0, args=(tau,), method='SLSQP',
                               options={'maxiter': 300, 'ftol': 1e-9})
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
                                options={'maxiter': 200, 'ftol': 1e-11})
                if res2.fun < 1e5:
                    current_score = -res2.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res2.x)
                        x0 = res2.x
            except:
                pass

        # Final high-tau refinement
        if time_left() > 10.0:
            try:
                res_final = minimize(smooth_objective, x0, args=(200,), method='L-BFGS-B',
                                     options={'maxiter': 150, 'ftol': 1e-12})
                if res_final.fun < 1e5:
                    current_score = -res_final.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res_final.x)
            except:
                pass

    # === Delaunay-guided refinement: use triangulation structure to avoid
    #     symmetric local minima by enforcing mesh-like vertex distribution ===
    if best_points is not None and time_left() > 25.0:
        # Get the Delaunay triangulation of the current best points
        try:
            tri = Delaunay(best_points)
            simplices = tri.simplices  # shape (M, 3)
        except:
            simplices = None

        if simplices is not None:
            # Build a penalty that encourages vertices to be well-distributed
            # according to the Delaunay mesh: penalize small edges in the mesh
            # to enforce a more uniform distribution
            def delaunay_guided_objective(params, tau):
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

                    # Add a small penalty for very short edges in the Delaunay mesh
                    # to encourage a more uniform distribution
                    try:
                        tri2 = Delaunay(pts)
                        edges = tri2.simplices
                        # Compute edge lengths
                        edge_lens = []
                        for tri in edges:
                            a, b, c = tri
                            edge_lens.append(np.linalg.norm(pts[a] - pts[b]))
                            edge_lens.append(np.linalg.norm(pts[b] - pts[c]))
                            edge_lens.append(np.linalg.norm(pts[c] - pts[a]))
                        edge_lens = np.array(edge_lens)
                        min_edge = np.min(edge_lens)
                        # Penalize if any edge is too short (relative to typical)
                        typical_edge = np.mean(edge_lens)
                        if typical_edge > 1e-10:
                            penalty = 0.01 * max(0, 0.3 - min_edge / typical_edge)
                        else:
                            penalty = 0.0
                    except:
                        penalty = 0.0

                    return -(score - penalty)
                except:
                    return 1e6

            x0 = points_to_params(best_points)
            for tau in [100, 150, 200]:
                if time_left() < 8.0:
                    break
                try:
                    res = minimize(delaunay_guided_objective, x0, args=(tau,),
                                   method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-11})
                    if res.fun < 1e5:
                        pts = params_to_points(res.x)
                        update_best(pts)
                        x0 = res.x
                except:
                    pass

    # === Local refinement with small perturbations around the best point ===
    if best_points is not None and time_left() > 15.0:
        x_best = points_to_params(best_points)

        for pert_scale in [0.01, 0.05]:
            if time_left() < 10.0:
                break
            for _ in range(5):
                if time_left() < 5.0:
                    break
                x_pert = x_best + np.random.normal(0, pert_scale, len(x_best))
                try:
                    res_p = minimize(smooth_objective, x_pert, args=(200,),
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
