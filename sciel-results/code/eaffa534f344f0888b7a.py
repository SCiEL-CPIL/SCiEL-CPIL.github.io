"""Program for covering the sphere with 130 equal caps using smooth soft-objective
optimization with analytical gradients and an exact convex-hull polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # safety margin before the 150 s hard limit

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def generate_grid(n_lat, n_lon):
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
        lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Pre-compute grids
    grid_coarse = generate_grid(40, 80)
    grid_med = generate_grid(80, 160)
    grid_fine = generate_grid(120, 240)

    def covering_radius(pts, grid):
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        return float(np.max(np.min(ang, axis=1)))

    def _covering_radius_from_hull(points):
        """Compute the exact covering radius from the convex hull."""
        try:
            hull = ConvexHull(points)
        except Exception:
            return np.pi
        
        max_radius = 0.0
        for simplex in hull.simplices:
            verts = points[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            normal = np.cross(b - a, c - a)
            norm_val = np.linalg.norm(normal)
            
            if norm_val < 1e-12:
                continue
            
            normal = normal / norm_val
            
            centroid = np.mean(verts, axis=0)
            if np.dot(normal, centroid) < 0:
                normal = -normal
            
            dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
            angles = np.arccos(dots)
            
            facet_radius = np.max(angles)
            max_radius = max(max_radius, facet_radius)
        
        return max_radius

    def soft_covering_radius_and_grad(points, grid_points, sigma=0.05):
        """
        Vectorized soft covering radius and its gradient using smooth soft-min/soft-max.
        Returns (value, grad) where grad has shape (N, 3).
        """
        dots = grid_points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # (M, N)
        
        # Soft-min: for each grid point, compute weighted average of angles
        # w_j = exp(-angle_j / sigma) / sum_k exp(-angle_k / sigma)
        neg_scaled = -angles / sigma  # (M, N)
        max_neg = np.max(neg_scaled, axis=1, keepdims=True)  # (M, 1)
        exp_neg = np.exp(neg_scaled - max_neg)  # (M, N)
        weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # (M, N)
        
        # Soft-min value: weighted sum of angles
        soft_min_vals = np.sum(weights * angles, axis=1)  # (M,)
        
        # Soft-max of soft-min values
        scaled_max = soft_min_vals / sigma
        max_scaled = np.max(scaled_max)
        exp_max = np.exp(scaled_max - max_scaled)
        max_weights = exp_max / np.sum(exp_max)  # (M,)
        soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))
        
        # Gradient computation
        denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)
        coeff = max_weights[:, np.newaxis] * weights  # (M, N)
        alpha = coeff / denom  # (M, N)
        
        # Term 1: -sum_i (alpha[i,j] / denom[i,j]) * g_i  -> -(alpha.T @ grid_points)
        term1 = -(alpha.T @ grid_points)  # (N, 3)
        
        # Term 2: sum_i alpha[i,j] * dots[i,j] * p_j
        beta = alpha * dots  # (M, N)
        beta_sum = np.sum(beta, axis=0)  # (N,)
        term2 = beta_sum[:, np.newaxis] * points  # (N, 3)
        
        grad = term1 + term2  # (N, 3)
        
        return soft_max_val, grad

    def lloyd_step(pts, grid, alpha=0.5, temp=2.0):
        """
        Perform one Lloyd's iteration step.
        """
        dots = np.dot(grid, pts.T)
        nearest = np.argmax(dots, axis=1)
        
        new_pts = np.zeros_like(pts)
        for i in range(N):
            mask = (nearest == i)
            if not np.any(mask):
                new_pts[i] = pts[i]
                continue
            
            cell_pts = grid[mask]
            d = np.dot(pts[i], cell_pts.T)
            d = np.clip(d, -1.0, 1.0)
            ang = np.arccos(d)
            
            w = np.exp(-ang * temp) 
            w /= np.sum(w)
            
            new_c = np.sum(cell_pts * w[:, None], axis=0)
            nrm = np.linalg.norm(new_c)
            if nrm < 1e-10:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = new_c / nrm
        
        current = (1 - alpha) * pts + alpha * new_pts
        current /= np.linalg.norm(current, axis=1, keepdims=True)
        return current

    def gap_fill_step(pts, grid, alpha=0.5):
        """
        Identify the worst point on the grid and move the nearest config point towards it.
        """
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        min_d = np.min(ang, axis=1)
        nearest_idx = np.argmin(ang, axis=1)
        
        worst_grid_idx = np.argmax(min_d)
        worst_grid_pt = grid[worst_grid_idx]
        nearest_config_idx = nearest_idx[worst_grid_idx]
        
        new_pts = pts.copy()
        p = pts[nearest_config_idx]
        v = worst_grid_pt - p
        v_tan = v - np.dot(v, p) * p
        new_p = p + alpha * v_tan
        new_p /= np.linalg.norm(new_p)
        new_pts[nearest_config_idx] = new_p
        
        return new_pts

    def optimize_smooth(pts, sigma, grid, max_iter=50):
        if time.time() > deadline - 10.0:
            return pts
        
        x0 = pts.flatten()
        
        def obj(x):
            if time.time() > deadline - 5.0:
                return 1e10
            val, _ = soft_covering_radius_and_grad(x.reshape(N, 3), grid, sigma)
            return val
        
        def grad(x):
            if time.time() > deadline - 5.0:
                return np.zeros_like(x)
            _, g = soft_covering_radius_and_grad(x.reshape(N, 3), grid, sigma)
            return g.flatten()
        
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            if res.fun < obj(x0):
                p = res.x.reshape(N, 3)
                p /= np.linalg.norm(p, axis=1, keepdims=True)
                return p
        except Exception:
            pass
        return pts

    def exact_local_optimize(points, grid_points, max_iters=100):
        """
        Final high-precision polish: iteratively move each point to reduce the
        maximum facet circumradius, using exact geometric computations.
        """
        pts = points.copy()
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            
            # Find the worst facet
            worst_facet = None
            worst_radius = -1
            worst_normal = None
            
            for simplex in hull.simplices:
                verts = pts[simplex]
                a, b, c = verts[0], verts[1], verts[2]
                normal = np.cross(b - a, c - a)
                norm_val = np.linalg.norm(normal)
                if norm_val < 1e-12:
                    continue
                normal = normal / norm_val
                centroid = np.mean(verts, axis=0)
                if np.dot(normal, centroid) < 0:
                    normal = -normal
                
                dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
                angles = np.arccos(dots)
                facet_radius = np.max(angles)
                
                if facet_radius > worst_radius:
                    worst_radius = facet_radius
                    worst_facet = simplex
                    worst_normal = normal
            
            if worst_facet is None:
                break
            
            # Move the vertices of the worst facet towards the normal direction
            # to reduce the circumradius
            step_size = 0.01
            for idx in worst_facet:
                p = pts[idx]
                # Move towards the normal
                v = worst_normal - p
                v_tan = v - np.dot(v, p) * p
                new_p = p + step_size * v_tan
                new_p /= np.linalg.norm(new_p)
                pts[idx] = new_p
            
            # Also perform a small Lloyd step to maintain balance
            pts = lloyd_step(pts, grid_points, alpha=0.1, temp=2.0)
            
            # Check convergence
            if worst_radius < 0.1974: # ~11.31 degrees
                break
        
        return pts

    best_pts = None
    best_rad = np.inf

    n_restarts = 10

    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        if r == 0:
            pts = fibonacci_spiral(N)
        elif r == 1:
            rng = np.random.RandomState(42)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 2:
            rng = np.random.RandomState(123)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 3:
            rng = np.random.RandomState(789)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r >= 4 and best_pts is not None:
            rng = np.random.RandomState(1000 + r)
            scale = 0.05 * (1.0 - r / (n_restarts + 1.0))
            pts = best_pts + scale * rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            pts = fibonacci_spiral(N)

        # Stage 1: Coarse Lloyd's
        for i in range(10):
            if time.time() > deadline - 25.0:
                break
            pts = lloyd_step(pts, grid_coarse, alpha=0.5, temp=2.0)

        # Stage 2: Medium grid Lloyd's + Gap filling
        for i in range(10):
            if time.time() > deadline - 20.0:
                break
            pts = lloyd_step(pts, grid_med, alpha=0.4, temp=3.0)
            if i % 5 == 0:
                pts = gap_fill_step(pts, grid_med, alpha=0.2)

        # Stage 3: Smooth optimization on Medium grid
        pts = optimize_smooth(pts, sigma=0.05, grid=grid_med, max_iter=30)

        # Stage 4: Fine grid Lloyd's + Gap filling
        for i in range(10):
            if time.time() > deadline - 15.0:
                break
            pts = lloyd_step(pts, grid_fine, alpha=0.3, temp=4.0)
            if i % 3 == 0:
                pts = gap_fill_step(pts, grid_fine, alpha=0.1)

        # Stage 5: Smooth optimization on Fine grid
        pts = optimize_smooth(pts, sigma=0.02, grid=grid_fine, max_iter=30)

        # Stage 6: Exact local optimization
        if time.time() < deadline - 10.0:
            pts = exact_local_optimize(pts, grid_fine, max_iters=50)

        rad = covering_radius(pts, grid_fine)
        
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()

    # Final polish
    if best_pts is not None and time.time() < deadline - 5.0:
        best_pts = exact_local_optimize(best_pts, grid_fine, max_iters=50)
        rad = covering_radius(best_pts, grid_fine)
        if rad < best_rad:
            best_rad = rad

    if best_pts is None:
        best_pts = fibonacci_spiral(N)

    return best_pts
