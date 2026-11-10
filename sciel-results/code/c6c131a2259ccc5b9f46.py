# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin

    n = 16
    m = n * (n - 1) // 2  # number of pairwise distances

    def score(pts):
        """Compute R = (d_min / d_max)^2"""
        d = pdist(pts, 'euclidean')
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def neg_ratio_flat(flat_pts):
        """Nonsmooth objective for Powell"""
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts, 'euclidean')
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return -(d_min / d_max) ** 2

    def _compute_distances(pts):
        """Compute pairwise distances and their gradients."""
        d = pdist(pts, 'euclidean')
        # Compute gradients: d_dk/d_pts for each distance k
        # d_k = ||p_i - p_j||, grad wrt p_i = (p_i - p_j)/d_k, grad wrt p_j = -(p_i - p_j)/d_k
        # We need to map from pdist ordering to (i,j) pairs
        idx_i, idx_j = np.triu_indices(n, k=1)
        diffs = pts[idx_i] - pts[idx_j]  # shape (m, 2)
        dists = np.linalg.norm(diffs, axis=1)
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        # Gradient of distance k wrt point i: (p_i - p_j) / d_k
        # Gradient of distance k wrt point j: -(p_i - p_j) / d_k
        grad_d = diffs / dists_safe[:, None]  # shape (m, 2)
        return d, dists, idx_i, idx_j, grad_d

    def soft_neg_ratio_flat(flat_pts, beta_min=0.05, beta_max=0.05):
        """
        Smooth surrogate: -soft_min(d)^2 / soft_max(d)^2 using log-sum-exp.
        soft_min(d) = -beta * log(sum(exp(-d/beta)))
        soft_max(d) = beta * log(sum(exp(d/beta)))
        """
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts, 'euclidean')
        
        # Soft min: -beta * log(sum(exp(-d/beta)))
        neg_d_over_beta = -d / beta_min
        neg_d_over_beta -= np.max(neg_d_over_beta)  # numerical stability
        soft_min = -beta_min * np.log(np.sum(np.exp(neg_d_over_beta)))
        
        # Soft max: beta * log(sum(exp(d/beta)))
        d_over_beta = d / beta_max
        d_over_beta -= np.max(d_over_beta)  # numerical stability
        soft_max = beta_max * np.log(np.sum(np.exp(d_over_beta)))
        
        if soft_max < 1e-10:
            return 0.0
        return -(soft_min / soft_max) ** 2

    def soft_neg_ratio_grad_flat(flat_pts, beta_min=0.05, beta_max=0.05):
        """
        Smooth surrogate with analytic gradient.
        Returns (value, gradient)
        """
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts, 'euclidean')
        idx_i, idx_j = np.triu_indices(n, k=1)
        diffs = pts[idx_i] - pts[idx_j]  # shape (m, 2)
        dists = np.linalg.norm(diffs, axis=1)
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        grad_d = diffs / dists_safe[:, None]  # shape (m, 2)

        # Soft min
        neg_d_over_beta = -d / beta_min
        neg_d_over_beta -= np.max(neg_d_over_beta)
        exp_neg = np.exp(neg_d_over_beta)
        sum_exp_neg = np.sum(exp_neg)
        soft_min = -beta_min * np.log(sum_exp_neg)
        
        # Gradient of soft_min wrt d_k: -exp(-d_k/beta) / sum(exp(-d_j/beta)) = -exp_neg[k] / sum_exp_neg
        grad_soft_min_wrt_d = -exp_neg / sum_exp_neg  # shape (m,)

        # Soft max
        d_over_beta = d / beta_max
        d_over_beta -= np.max(d_over_beta)
        exp_pos = np.exp(d_over_beta)
        sum_exp_pos = np.sum(exp_pos)
        soft_max = beta_max * np.log(sum_exp_pos)
        
        # Gradient of soft_max wrt d_k: exp(d_k/beta) / sum(exp(d_j/beta)) = exp_pos[k] / sum_exp_pos
        grad_soft_max_wrt_d = exp_pos / sum_exp_pos  # shape (m,)

        if soft_max < 1e-10:
            return 0.0, np.zeros(2 * n)

        ratio = soft_min / soft_max
        val = -ratio ** 2

        # d(val)/d(soft_min) = -2 * ratio / soft_max
        # d(val)/d(soft_max) = 2 * ratio * soft_min / soft_max^2 = 2 * ratio^2 / soft_max
        dval_dsoft_min = -2 * ratio / soft_max
        dval_dsoft_max = 2 * ratio * soft_min / (soft_max ** 2)

        # Chain rule: dval/d(pts) = sum_k [dval_dd_k * d(d_k)/d(pts)]
        # dval_dd_k = dval_dsoft_min * grad_soft_min_wrt_d[k] + dval_dsoft_max * grad_soft_max_wrt_d[k]
        dval_dd = dval_dsoft_min * grad_soft_min_wrt_d + dval_dsoft_max * grad_soft_max_wrt_d  # shape (m,)

        # Now accumulate gradient wrt each point
        grad_pts = np.zeros((n, 2))
        for k in range(m):
            i = idx_i[k]
            j = idx_j[k]
            # d(d_k)/d(p_i) = (p_i - p_j) / d_k = grad_d[k]
            # d(d_k)/d(p_j) = -(p_i - p_j) / d_k = -grad_d[k]
            grad_pts[i] += dval_dd[k] * grad_d[k]
            grad_pts[j] += dval_dd[k] * (-grad_d[k])

        return val, grad_pts.flatten()

    def make_hexagonal_initial():
        """
        Generate a 16-point configuration based on a hexagonal lattice.
        """
        pts = []
        # Center
        pts.append([0.0, 0.0])
        
        # Inner ring (6 points, radius 1)
        for k in range(6):
            angle = k * 2 * np.pi / 6
            pts.append([np.cos(angle), np.sin(angle)])
            
        # Outer ring (9 points, radius sqrt(3))
        r_outer = np.sqrt(3)
        for k in range(9):
            angle = k * 2 * np.pi / 9
            pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
            
        pts = np.array(pts[:n])
        pts -= np.mean(pts, axis=0)
        return pts

    def make_grid_initial():
        """
        Generate a 4x4 grid configuration.
        """
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack((xx.ravel(), yy.ravel()))
        pts -= np.mean(pts, axis=0)
        return pts

    def make_circle_initial():
        """
        Generate 16 points evenly spaced on a circle.
        """
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        pts -= np.mean(pts, axis=0)
        return pts

    def make_random_initial(seed):
        """
        Generate random points in a unit square.
        """
        rng = np.random.RandomState(seed)
        pts = rng.rand(n, 2) * 2 - 1
        pts -= np.mean(pts, axis=0)
        return pts

    def improve_powell(pts, max_iter=1000):
        """
        Local optimization using Powell's method.
        """
        x0 = pts.flatten()
        res = minimize(
            neg_ratio_flat,
            x0,
            method='Powell',
            options={'maxiter': max_iter, 'xtol': 1e-10, 'ftol': 1e-10}
        )
        best_pts = res.x.reshape(n, 2)
        return best_pts

    def improve_lbfgs_smooth(pts, max_iter=2000, beta_min=0.05, beta_max=0.05):
        """
        Local optimization using L-BFGS-B with smooth surrogate and analytic gradient.
        """
        x0 = pts.flatten()
        
        def fun_and_grad(x):
            val, grad = soft_neg_ratio_grad_flat(x, beta_min, beta_max)
            return val, grad
        
        res = minimize(
            fun_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        best_pts = res.x.reshape(n, 2)
        return best_pts

    def final_polish(pts, beta_min=0.02, beta_max=0.02, max_iter=3000):
        """
        Final high-precision polish with tighter beta values.
        """
        x0 = pts.flatten()
        
        def fun_and_grad(x):
            val, grad = soft_neg_ratio_grad_flat(x, beta_min, beta_max)
            return val, grad
        
        res = minimize(
            fun_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        best_pts = res.x.reshape(n, 2)
        
        # Also try Powell for exact non-smooth objective
        res_powell = minimize(
            neg_ratio_flat,
            best_pts.flatten(),
            method='Powell',
            options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-12}
        )
        powell_pts = res_powell.x.reshape(n, 2)
        
        # Return the better of the two
        if score(powell_pts) > score(best_pts):
            return powell_pts
        return best_pts

    # Generate initial configurations
    initial_configs = [
        make_hexagonal_initial(),
        make_grid_initial(),
        make_circle_initial(),
        make_random_initial(1),
        make_random_initial(42),
        make_random_initial(123),
        make_random_initial(999),
    ]
    
    # Add perturbed versions of the best structured ones
    rng = np.random.RandomState(0)
    base_hex = make_hexagonal_initial()
    for noise_scale in [0.05, 0.1, 0.2]:
        perturbed = base_hex + rng.randn(n, 2) * noise_scale
        perturbed -= np.mean(perturbed, axis=0)
        initial_configs.append(perturbed)
        
    base_grid = make_grid_initial()
    for noise_scale in [0.05, 0.1]:
        perturbed = base_grid + rng.randn(n, 2) * noise_scale
        perturbed -= np.mean(perturbed, axis=0)
        initial_configs.append(perturbed)

    best_pts = None
    best_val = -np.inf

    # Optimize each configuration
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        # 1. L-BFGS-B with smooth surrogate (fast)
        opt_pts = improve_lbfgs_smooth(init_pts, max_iter=1000, beta_min=0.05, beta_max=0.05)
        val = score(opt_pts)
        
        if val > best_val:
            best_val = val
            best_pts = opt_pts
        
        # 2. If we have time, do a second pass with Powell on the best found so far
        if time.time() < deadline - 30:
             if val > best_val - 0.01:
                 polished_pts = improve_powell(opt_pts, max_iter=500)
                 polished_val = score(polished_pts)
                 if polished_val > best_val:
                     best_val = polished_val
                     best_pts = polished_pts

    # Final high-precision polish on the best candidate
    if best_pts is not None and time.time() < deadline - 5:
        final_pts = final_polish(best_pts, beta_min=0.02, beta_max=0.02, max_iter=2000)
        final_val = score(final_pts)
        if final_val > best_val:
            best_pts = final_pts
            best_val = final_val
    
    # Ensure we have a valid result
    if best_pts is None:
        best_pts = make_hexagonal_initial()
    
    return best_pts

# EVOLVE-BLOCK-END
