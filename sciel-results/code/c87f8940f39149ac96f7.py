import numpy as np
import time
import warnings
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    def exact_score(pts):
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_to_unit_max(pts):
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return pts.copy()
        return pts / d_max

    def soft_min_distance(pts, beta=50.0):
        dists = pdist(pts)
        if np.max(dists) < 1e-10:
            return 0.0
        m = np.min(dists)
        shifted = (dists - m) / beta
        exp_terms = np.exp(-shifted)
        log_mean_exp = np.log(np.mean(exp_terms))
        return m - beta * log_mean_exp

    def smooth_objective_scaled(params, beta=50.0):
        """
        Smooth objective: minimize -soft_min_distance after scaling to unit max.
        """
        pts = params.reshape((n, 2))
        pts_norm = normalize_to_unit_max(pts)
        return -soft_min_distance(pts_norm, beta=beta)

    def exact_objective_scaled(params):
        """
        Exact objective: minimize -min_distance after scaling to unit max.
        """
        pts = params.reshape((n, 2))
        pts_norm = normalize_to_unit_max(pts)
        dists = pdist(pts_norm)
        d_min = np.min(dists)
        return -d_min

    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None or not np.all(np.isfinite(pts)):
            return
        s = exact_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Phase 1: Multi-restart L-BFGS-B on smooth surrogate
    if time.time() < deadline - 120:
        init_points_list = []

        # Hexagonal lattice based initializations
        # Generate a hexagonal grid of points
        hex_points = []
        # (i, j) -> (i + j/2, j * sqrt(3)/2)
        for j in range(-3, 4):
            for i in range(-3, 4):
                x = i + j / 2.0
                y = j * np.sqrt(3) / 2.0
                hex_points.append([x, y])
        hex_points = np.array(hex_points)
        
        # Find the 19 points closest to origin
        dists_to_origin = np.linalg.norm(hex_points, axis=1)
        sorted_indices = np.argsort(dists_to_origin)
        top_19 = hex_points[sorted_indices[:19]]
        
        # We need to remove 3 points to get 16.
        # Try removing different combinations.
        init_points_list.append(top_19[:16])
        
        for mask in [
            [0, 1, 2],
            [16, 17, 18],
            [0, 8, 16],
            [2, 5, 10],
            [1, 4, 7]
        ]:
            remaining = [i for i in range(19) if i not in mask]
            init_points_list.append(top_19[remaining])

        # Grid
        x = np.arange(4)
        y = np.arange(4)
        xx, yy = np.meshgrid(x, y)
        grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
        init_points_list.append(grid_pts)

        # Circle
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        init_points_list.append(circle_pts)

        # Random inits
        for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            rng = np.random.RandomState(seed)
            rand_pts = rng.uniform(-2, 2, size=(n, 2))
            init_points_list.append(rand_pts)

        for init_pts in init_points_list:
            if time.time() > deadline - 120:
                break
            try:
                res = minimize(
                    lambda p: smooth_objective_scaled(p, beta=50.0),
                    init_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if np.all(np.isfinite(res.x)):
                    update_best(res.x.reshape((n, 2)))
            except Exception:
                pass

    # Phase 2: Adaptive beta annealing refinement
    if best_points is not None and time.time() < deadline - 60:
        beta_schedule = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0]
        for beta in beta_schedule:
            if time.time() > deadline - 60:
                break
            try:
                res_smooth = minimize(
                    lambda p: smooth_objective_scaled(p, beta=beta),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if np.all(np.isfinite(res_smooth.x)):
                    pts_smooth = res_smooth.x.reshape((n, 2))
                    s_smooth = exact_score(pts_smooth)
                    if s_smooth > best_score:
                        best_score = s_smooth
                        best_points = pts_smooth.copy()
            except Exception:
                pass

        # Phase 2b: Exact objective refinement with Powell
        if best_points is not None and time.time() < deadline - 30:
            try:
                res = minimize(
                    exact_objective_scaled,
                    best_points.flatten(),
                    method='Powell',
                    options={'maxiter': 2000, 'ftol': 1e-14, 'xtol': 1e-14}
                )
                if np.all(np.isfinite(res.x)):
                    update_best(res.x.reshape((n, 2)))
            except Exception:
                pass

    # Phase 3: Additional diverse local optimizations
    if time.time() < deadline - 90:
        def obj_full2(params):
            pts = params.reshape((n, 2))
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-10:
                return 0.0
            return -(d_min / d_max)

        # More random initializations with Powell
        if time.time() < deadline - 60:
            for seed in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                if time.time() > deadline - 60:
                    break
                rng = np.random.RandomState(seed)
                rand_pts = rng.uniform(-2, 2, size=(n, 2))
                try:
                    res = minimize(obj_full2, rand_pts.flatten(), method='Powell', options={'maxiter': 500, 'ftol': 1e-14})
                    if np.all(np.isfinite(res.x)):
                        update_best(res.x.reshape((n, 2)))
                except Exception:
                    pass

    # Phase 4: High-precision final polish
    if best_points is not None and time.time() < deadline - 10:
        def obj_full_highprec(params):
            pts = params.reshape((n, 2))
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-10:
                return 0.0
            return -(d_min / d_max)

        # Nelder-Mead with tight but realistic tolerance
        try:
            res = minimize(
                obj_full_highprec,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if np.all(np.isfinite(res.x)):
                update_best(res.x.reshape((n, 2)))
        except Exception:
            pass

        # Final Powell with tight tolerance
        if time.time() > deadline - 3:
            try:
                res = minimize(
                    obj_full_highprec,
                    best_points.flatten(),
                    method='Powell',
                    options={'maxiter': 5000, 'ftol': 1e-14, 'xtol': 1e-14}
                )
                if np.all(np.isfinite(res.x)):
                    update_best(res.x.reshape((n, 2)))
            except Exception:
                pass

    if best_points is None:
        # Absolute fallback
        x = np.arange(4)
        y = np.arange(4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((xx.ravel(), yy.ravel()))

    return best_points
