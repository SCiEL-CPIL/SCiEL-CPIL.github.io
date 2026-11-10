import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0

    def compute_ratio(points):
        """Compute the actual score: (d_min/d_max)^2"""
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def get_cost_fast(pts):
        """Calculate (d_max / d_min)^2. Lower is better."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        mask = ~np.eye(pts.shape[0], dtype=bool)
        
        dist_sq_min = np.where(mask, dist_sq, np.inf)
        d_min_sq = np.min(dist_sq_min)
        d_max_sq = np.max(dist_sq)
        
        if d_min_sq <= 1e-12 or d_max_sq <= 1e-12:
            return np.inf
            
        return d_max_sq / d_min_sq

    def get_hexagonal_lattice_init():
        """Generate a hexagonal lattice initialization with 16 central points."""
        R = 10
        points = []
        a = 1.0
        b = np.sqrt(3) / 2
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                x = i * a + j * 0.5 * a
                y = j * b
                points.append([x, y])
        
        points = np.array(points)
        
        centroid = np.mean(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        idx = np.argsort(dists)
        selected = points[idx[:n]]
        
        selected = selected - np.mean(selected, axis=0)
        return selected

    def get_grid_init():
        """4x4 grid."""
        x = np.linspace(-1, 1, 4)
        y = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def get_random_init():
        """Random initialization."""
        return np.random.randn(n, d) * 0.5

    def equidistance_energy(params):
        """
        Energy function that is minimized when all pairwise distances are equal.
        E = sum(1/d^2 + d^2). Minimum at d=1 for each pair.
        """
        pts = params.reshape(n, d)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        mask = ~np.eye(n, dtype=bool)
        dist_sq = np.where(mask, dist_sq, 0)
        
        # Avoid division by zero
        dist_sq_safe = np.maximum(dist_sq, 1e-12)
        
        energy = np.sum(1.0 / dist_sq_safe + dist_sq_safe)
        return energy

    def equidistance_grad(params):
        """Gradient of equidistance energy."""
        pts = params.reshape(n, d)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        mask = ~np.eye(n, dtype=bool)
        dist_sq = np.where(mask, dist_sq, 0)
        dist_sq_safe = np.maximum(dist_sq, 1e-12)
        
        # d/dx_i (1/d^2 + d^2) = (-2/d^4 + 2d^2) * d/dx_i d^2
        # d^2 = sum (x_i - x_j)^2
        # d/dx_i d^2 = 2 * (x_i - x_j)
        # So term is (-2/d^4 + 2d^2) * 2 * (x_i - x_j) = 2 * (d^2 - 1/d^2) * (x_i - x_j)
        
        coeff = 2.0 * (dist_sq_safe - 1.0 / dist_sq_safe)
        # coeff is n x n
        # Gradient for point i is sum_j coeff[i,j] * (x_i - x_j)
        # diff[i,j] = x_i - x_j
        grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        return grad.flatten()

    def smooth_ratio_objective(params):
        """
        Smooth approximation of d_max^2 / d_min^2.
        """
        pts = params.reshape(n, d)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        mask = ~np.eye(n, dtype=bool)
        dist_sq = np.where(mask, dist_sq, 0)
        
        valid_dsq = dist_sq[dist_sq > 1e-12]
        if len(valid_dsq) == 0:
            return 1e10
        
        # Adaptive beta
        median_dsq = np.median(valid_dsq)
        beta = 0.05 * median_dsq
        beta = max(beta, 1e-4)
        
        # Soft Max
        exp_sq = np.exp(dist_sq / beta)
        log_sum_exp_max = beta * np.log(np.sum(exp_sq))
        
        # Soft Min
        exp_neg_sq = np.exp(-dist_sq / beta)
        log_sum_exp_min = -beta * np.log(np.sum(exp_neg_sq))
        
        d_min_sq_smooth = log_sum_exp_min
        d_max_sq_smooth = log_sum_exp_max
        
        if d_min_sq_smooth <= 1e-12 or d_max_sq_smooth <= 1e-12:
            return 1e10
            
        return d_max_sq_smooth / d_min_sq_smooth

    def polish_with_l_bfgs(pts, budget):
        """Polish using L-BFGS-B on the smooth ratio objective."""
        t_start = time.time()
        x0 = pts.flatten()
        try:
            res = minimize(
                smooth_ratio_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success and np.all(np.isfinite(res.x)):
                new_pts = res.x.reshape(n, d)
                return new_pts
        except Exception:
            pass
        return pts

    def polish_with_nelder_mead(pts, budget):
        """Polish using Nelder-Mead on the exact cost."""
        t_start = time.time()
        x0 = pts.flatten()
        try:
            res = minimize(
                get_cost_fast,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if hasattr(res, 'fun') and np.isfinite(res.fun):
                new_pts = res.x.reshape(n, d)
                return new_pts
        except Exception:
            pass
        return pts

    # 1. Generate Initializations
    initializations = [
        get_hexagonal_lattice_init(),
        get_grid_init(),
        get_random_init(),
        get_random_init(),
        get_random_init()
    ]
    
    # Add perturbed hex
    for _ in range(2):
        init = get_hexagonal_lattice_init() + np.random.randn(n, d) * 0.2
        init -= np.mean(init, axis=0)
        initializations.append(init)

    best_points = None
    best_cost = np.inf
    best_score = -1.0

    # 2. Equidistance Energy Minimization Phase
    # This phase finds a configuration where all distances are roughly equal,
    # which is a strong candidate for the max-min ratio optimum.
    for init_points in initializations:
        if time.time() - start_time > deadline - safety_margin - 60:
            break
            
        x0 = init_points.flatten()
        try:
            res = minimize(
                equidistance_energy,
                x0,
                jac=equidistance_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success and np.all(np.isfinite(res.x)):
                pts = res.x.reshape(n, d)
                cost = get_cost_fast(pts)
                score = 1.0 / cost
                if cost < best_cost:
                    best_cost = cost
                    best_points = pts
                    best_score = score
        except Exception:
            pass

    # If equidistance phase didn't find anything good, fall back to initializations
    if best_points is None:
        for init_points in initializations:
            cost = get_cost_fast(init_points)
            if cost < best_cost:
                best_cost = cost
                best_points = init_points
                best_score = 1.0 / cost

    # 3. Smooth Ratio Polish (L-BFGS)
    if best_points is not None:
        time_elapsed = time.time() - start_time
        budget = deadline - time_elapsed - safety_margin - 30
        if budget > 5:
            polished_pts = polish_with_l_bfgs(best_points, budget)
            cost = get_cost_fast(polished_pts)
            if cost < best_cost:
                best_cost = cost
                best_points = polished_pts
                best_score = 1.0 / cost

    # 4. Exact Nelder-Mead Polish
    if best_points is not None:
        time_elapsed = time.time() - start_time
        budget = deadline - time_elapsed - safety_margin
        if budget > 5:
            polished_pts = polish_with_nelder_mead(best_points, budget)
            cost = get_cost_fast(polished_pts)
            if cost < best_cost:
                best_cost = cost
                best_points = polished_pts
                best_score = 1.0 / cost

    # 5. Final Check and Random Restarts if time permits
    if best_points is not None:
        time_elapsed = time.time() - start_time
        remaining = deadline - time_elapsed - safety_margin
        if remaining > 20:
            # Try a few random restarts with quick polish
            for _ in range(3):
                if time.time() - start_time > deadline - safety_margin:
                    break
                rand_init = get_random_init()
                # Quick equidistance polish
                x0 = rand_init.flatten()
                try:
                    res = minimize(
                        equidistance_energy,
                        x0,
                        jac=equidistance_grad,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
                    )
                    if res.success and np.all(np.isfinite(res.x)):
                        pts = res.x.reshape(n, d)
                        cost = get_cost_fast(pts)
                        if cost < best_cost:
                            best_cost = cost
                            best_points = pts
                            best_score = 1.0 / cost
                except Exception:
                    pass

    # Ensure valid output
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = initializations[0]

    return best_points

# EVOLVE-BLOCK-END
