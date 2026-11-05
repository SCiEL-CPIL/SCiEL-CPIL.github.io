"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = 8
    
    def scale_to_diameter_2(pts):
        """Scale point set so max pairwise distance is exactly 2."""
        pts = np.array(pts, dtype=float)
        # Compute all pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        max_dist = np.max(dists)
        if max_dist < 1e-15:
            return pts
        return pts * (2.0 / max_dist)
    
    def log_product_of_distances(pts):
        """Compute log of product of all ordered pairwise distances after scaling to diameter 2."""
        pts = scale_to_diameter_2(pts)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        # Mask diagonal (distance 0)
        mask = ~np.eye(n, dtype=bool)
        d = dists[mask]
        # Filter out zeros to avoid log(0)
        d = d[d > 1e-15]
        if len(d) == 0:
            return -np.inf
        return np.sum(np.log(d))
    
    def neg_log_product(params):
        """Negative log-product for minimization. params is 2n vector (x0,y0,...,x7,y7)."""
        pts = params.reshape(n, 2)
        val = log_product_of_distances(pts)
        if np.isinf(val) or np.isnan(val):
            return 1e10
        return -val
    
    def score_from_params(params):
        """Compute the actual score Delta/n^n."""
        pts = scale_to_diameter_2(params.reshape(n, 2))
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        mask = ~np.eye(n, dtype=bool)
        d = dists[mask]
        d = d[d > 1e-15]
        if len(d) == 0:
            return 0.0
        delta = np.prod(d)
        score = delta / (n ** n)
        return score
    
    # Generate diverse initial points
    np.random.seed(42)
    initial_points = []
    
    # 1. Regular octagon (baseline)
    angles_reg = 2 * np.pi * np.arange(8) / 8
    initial_points.append(np.column_stack([np.cos(angles_reg), np.sin(angles_reg)]).flatten())
    
    # 2. Perturbed regular octagon
    for trial in range(5):
        angles = 2 * np.pi * np.arange(8) / 8 + np.random.normal(0, 0.1, 8)
        radii = 1.0 + np.random.normal(0, 0.1, 8)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_points.append(pts.flatten())
    
    # 3. Antipodal pair structure: 4 pairs of antipodal points
    for trial in range(3):
        angles = np.sort(np.random.uniform(0, np.pi, 4))
        pts = []
        for a in angles:
            pts.append([np.cos(a), np.sin(a)])
            pts.append([-np.cos(a), -np.sin(a)])
        initial_points.append(np.array(pts).flatten())
    
    # 4. Grid-based perturbation
    for trial in range(3):
        angles = 2 * np.pi * np.arange(8) / 8 + np.random.uniform(-0.3, 0.3, 8)
        radii = 1.0 + np.random.uniform(-0.2, 0.2, 8)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_points.append(pts.flatten())
    
    # 5. Random points in disk
    for trial in range(5):
        r = np.sqrt(np.random.uniform(0, 1, 8))
        theta = 2 * np.pi * np.random.uniform(0, 1, 8)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initial_points.append(pts.flatten())
    
    # 6. Two concentric squares
    for trial in range(3):
        r1 = 1.0
        r2 = 0.5 + np.random.uniform(-0.1, 0.1)
        a1 = 2 * np.pi * np.arange(4) / 4 + np.random.uniform(0, np.pi/4)
        a2 = 2 * np.pi * np.arange(4) / 4 + np.random.uniform(0, np.pi/4)
        pts1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
        pts2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        initial_points.append(np.vstack([pts1, pts2]).flatten())
    
    # 7. Rectangle-based: 2x4 grid perturbed
    for trial in range(3):
        xs = np.random.uniform(-1, 1, 8)
        ys = np.random.uniform(-1, 1, 8)
        initial_points.append(np.column_stack([xs, ys]).flatten())
    
    # Optimize from each starting point
    best_params = None
    best_score = 0.0
    
    for i, init in enumerate(initial_points):
        if time.time() > deadline - 10.0:
            break
        
        try:
            # Use Nelder-Mead (derivative-free) for robustness
            result = minimize(
                neg_log_product,
                init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.fun < 1e9:  # Valid result
                s = score_from_params(result.x)
                if s > best_score:
                    best_score = s
                    best_params = result.x.copy()
        except Exception:
            continue
    
    # If we found something better, refine with L-BFGS-B
    if best_params is not None and time.time() < deadline - 15.0:
        try:
            # Compute gradient numerically for L-BFGS-B
            def neg_log_product_grad(params):
                val = neg_log_product(params)
                if val > 1e9:
                    return val, np.zeros(2*n)
                # Numerical gradient
                eps = 1e-8
                grad = np.zeros(2*n)
                for j in range(2*n):
                    params_plus = params.copy()
                    params_plus[j] += eps
                    params_minus = params.copy()
                    params_minus[j] -= eps
                    grad[j] = (neg_log_product(params_plus) - neg_log_product(params_minus)) / (2*eps)
                return val, grad
            
            result2 = minimize(
                neg_log_product_grad,
                best_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-12}
            )
            
            if result2.fun < 1e9:
                s2 = score_from_params(result2.x)
                if s2 > best_score:
                    best_score = s2
                    best_params = result2.x.copy()
        except Exception:
            pass
    
    # If no improvement found, fall back to regular octagon
    if best_params is None:
        angles_reg = 2 * np.pi * np.arange(8) / 8
        best_params = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)]).flatten()
    
    # Return as list of [x, y] pairs
    pts = best_params.reshape(n, 2)
    return [[float(pts[k, 0]), float(pts[k, 1])] for k in range(n)]
# EVOLVE-BLOCK-END
