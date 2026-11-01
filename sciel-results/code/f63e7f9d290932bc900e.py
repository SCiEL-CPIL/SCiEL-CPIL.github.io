"""Program for covering the sphere with 130 equal caps using a differentiable
neural-network potential-field regression approach with exact hull-based refinement."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 142.0

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def generate_grid(n_lat, n_lon):
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    grid_main = generate_grid(120, 240)
    grid_fine = generate_grid(80, 160)

    def covering_radius(pts, grid=None):
        if grid is None:
            grid = grid_main
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        return float(np.max(np.min(ang, axis=1)))

    def exact_covering_radius(pts):
        """Compute exact covering radius from convex hull."""
        try:
            hull = ConvexHull(pts)
        except Exception:
            return covering_radius(pts)
        max_radius = 0.0
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
            max_radius = max(max_radius, facet_radius)
        return max_radius

    def neural_potential_loss_and_grad(points, grid_points, beta=50.0, lam=0.1):
        """
        Neural-network-style differentiable potential field loss.
        
        The key idea: instead of directly optimizing the smooth max of distances,
        we construct a learned potential field where each point contributes a 
        Gaussian-like bump, and we minimize the maximum of this field over the grid.
        
        The potential at grid point i is:
            V_i = -log(sum_j exp(-beta * d_ij^2))
        
        where d_ij is the angular distance. This is equivalent to a soft-min
        but with a quadratic kernel that gives better conditioning.
        
        The gradient pushes points to reduce the maximum potential value.
        
        Additionally, we add a repulsion term to prevent point clustering.
        """
        dots = grid_points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # (M, N)
        d2 = angles ** 2  # (M, N)
        
        # Soft-min via log-sum-exp: V_i = -log(sum_j exp(-beta * d_ij^2))
        neg_beta_d2 = -beta * d2  # (M, N)
        max_nb = np.max(neg_beta_d2, axis=1, keepdims=True)  # (M, 1)
        exp_nb = np.exp(neg_beta_d2 - max_nb)  # (M, N)
        log_sum = max_nb[:, 0] + np.log(np.sum(exp_nb, axis=1))  # (M,)
        potential = -log_sum  # (M,)
        
        # Smooth max of potential
        scaled = potential / 0.02
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        max_weights = exp_scaled / np.sum(exp_scaled)  # (M,)
        loss_val = 0.02 * (max_scaled + np.log(np.sum(exp_scaled)))
        
        # Gradient of loss w.r.t. points
        # d(potential_i)/d(points_j) = -d/d(points_j) log(sum_k exp(-beta * d_ik^2))
        # = (sum_k exp(-beta * d_ik^2) * (-beta) * 2 * d_ik * d(d_ik)/d(points_j)) / sum_k exp(-beta * d_ik^2)
        # = -2 * beta * sum_k w_ik * d_ik * d(d_ik)/d(points_j)
        # where w_ik = exp(-beta * d_ik^2) / sum_k' exp(-beta * d_ik'^2)
        
        # d_ik = angle between grid_i and points_k
        # d(d_ik)/d(points_k) = -grid_i / sin(d_ik) + (grid_i . points_k) / sin^2(d_ik) * points_k
        #                      = -grid_i / sin(d_ik) + dots[i,k] / sin^2(d_ik) * points_k
        # Projected to tangent: this is already tangent to the sphere at points_k
        
        # For j != k: d(d_ik)/d(points_j) = 0
        
        # So d(potential_i)/d(points_j) = -2 * beta * w_ij * d_ij * (-grid_i / sin(d_ij) + dots[i,j]/sin^2(d_ij) * points_j)
        # = 2 * beta * w_ij * (grid_i / 1 - dots[i,j] * points_j / sin(d_ij))
        # Wait, let me redo this more carefully.
        
        # d_ij = arccos(dots[i,j])
        # d(d_ij)/d(dots[i,j]) = -1/sqrt(1-dots^2) = -1/sin(d_ij)
        # d(dots[i,j])/d(points_j) = grid_i
        # So d(d_ij)/d(points_j) = -grid_i / sin(d_ij)
        # Projected to tangent plane at points_j:
        # tangent_grad = -grid_i / sin(d_ij) + (grid_i . points_j)/sin(d_ij) * points_j
        # = -grid_i / sin(d_ij) + dots[i,j]/sin(d_ij) * points_j
        
        # d(potential_i)/d(points_j) = -2 * beta * w_ij * d_ij * tangent_grad_ij
        # = -2 * beta * w_ij * d_ij * (-grid_i / sin(d_ij) + dots[i,j]/sin(d_ij) * points_j)
        # = 2 * beta * w_ij * d_ij / sin(d_ij) * (grid_i - dots[i,j] * points_j)
        
        sin_d = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)
        # tan_d = d_ij / sin(d_ij) = angles / sin_d
        tan_d = angles / sin_d  # (M, N)
        
        # w_ij from the log-sum-exp
        w = exp_nb / np.sum(exp_nb, axis=1, keepdims=True)  # (M, N)
        
        # coeff[i,j] = max_weights[i] * 2 * beta * w[i,j] * tan_d[i,j]
        coeff = max_weights[:, np.newaxis] * (2.0 * beta * w * tan_d)  # (M, N)
        
        # d(loss)/d(points_j) = sum_i coeff[i,j] * (grid_i - dots[i,j] * points_j)
        # = sum_i coeff[i,j] * grid_i - points_j * sum_i coeff[i,j] * dots[i,j]
        
        term1 = coeff.T @ grid_points  # (N, 3)
        term2_coeff = np.sum(coeff * dots, axis=0)  # (N,)
        term2 = term2_coeff[:, np.newaxis] * points  # (N, 3)
        
        grad_potential = term1 - term2  # (N, 3)
        
        # Add repulsion term to prevent clustering
        # Repulsion: for each pair (j, k), add a force pushing them apart
        # Use a simple pairwise repulsion with cutoff
        repulsion = np.zeros_like(points)
        # Compute pairwise distances
        dots_pair = points @ points.T  # (N, N)
        dots_pair = np.clip(dots_pair, -1.0, 1.0)
        angles_pair = np.arccos(dots_pair)  # (N, N)
        
        # Only consider pairs within a certain distance
        cutoff = 0.3  # ~17 degrees
        mask = (angles_pair < cutoff) & (np.arange(N)[:, None] != np.arange(N)[None, :])
        
        for j in range(N):
            neighbors = np.where(mask[j])[0]
            if len(neighbors) == 0:
                continue
            # Force away from neighbors
            for k in neighbors:
                direction = points[j] - points[k]
                # Project to tangent at points[j]
                proj = direction - np.dot(direction, points[j]) * points[j]
                repulsion[j] += proj / (np.linalg.norm(proj) + 1e-10)
        
        grad = grad_potential + lam * repulsion
        
        return loss_val, grad

    def lloyd_covering_step(pts, grid, n_iters=30, alpha=0.5):
        current = pts.copy()
        for it in range(n_iters):
            if time.time() > deadline - 15.0:
                break
            dots = np.dot(grid, current.T)
            nearest = np.argmax(dots, axis=1)
            new_pts = np.zeros_like(current)
            for i in range(N):
                mask = (nearest == i)
                if not np.any(mask):
                    new_pts[i] = current[i]
                    continue
                cell_pts = grid[mask]
                d = current[i] @ cell_pts.T
                d = np.clip(d, -1.0, 1.0)
                ang = np.arccos(d)
                temp = 3.0
                w = np.exp(ang * temp)
                w /= np.sum(w)
                new_c = np.sum(cell_pts * w[:, None], axis=0)
                nrm = np.linalg.norm(new_c)
                if nrm < 1e-10:
                    new_pts[i] = current[i]
                else:
                    new_pts[i] = new_c / nrm
            current = (1 - alpha) * current + alpha * new_pts
            current /= np.linalg.norm(current, axis=1, keepdims=True)
        return current

    def optimize_neural(pts, betas, max_iter_per_stage=20):
        """Multi-stage optimization using the neural potential field loss."""
        if time.time() > deadline - 15.0:
            return pts
        
        current = pts.copy()
        
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            
            def objective_and_grad(x_flat, beta=beta):
                pts_local = x_flat.reshape(N, 3)
                norms = np.linalg.norm(pts_local, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                pts_local = pts_local / norms
                val, grad = neural_potential_loss_and_grad(pts_local, grid_main, beta=beta, lam=0.05)
                return val, grad.flatten()
            
            x0 = current.flatten()
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter_per_stage, 'ftol': 1e-10, 'gtol': 1e-5}
                )
                if res.fun < objective_and_grad(x0)[0]:
                    p = res.x.reshape(N, 3)
                    p /= np.linalg.norm(p, axis=1, keepdims=True)
                    current = p
            except Exception:
                pass
        
        return current

    def exact_local_optimize(points, max_iters=50):
        """
        Final high-precision polish: iteratively move points to reduce the
        maximum facet circumradius, using exact geometric computations.
        """
        pts = points.copy()
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            
            worst_facet = None
            worst_radius = 0.0
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
            
            move_alpha = 0.01
            for idx in worst_facet:
                pts[idx] = pts[idx] + move_alpha * worst_normal
                pts[idx] /= np.linalg.norm(pts[idx])
        
        return pts

    best_pts = None
    best_rad = np.inf

    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 40.0:
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
            scale = max(0.01, 0.04 - 0.005 * (r - 4))
            pts = best_pts + scale * rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            pts = fibonacci_spiral(N)
        
        # Phase 1: Lloyd iterations for coarse optimization
        cand = lloyd_covering_step(pts, grid_main, n_iters=25, alpha=0.5)
        
        # Phase 2: Neural potential field optimization (the new structural approach)
        betas = [20.0, 50.0, 100.0, 200.0]
        cand = optimize_neural(cand, betas, max_iter_per_stage=15)
        
        # Phase 3: Final fine-tuning with very high beta
        if time.time() < deadline - 20.0:
            cand = optimize_neural(cand, [300.0, 500.0], max_iter_per_stage=10)
        
        # Phase 4: Exact hull-based local optimization
        if time.time() < deadline - 10.0:
            cand = exact_local_optimize(cand, max_iters=30)
        
        rad = covering_radius(cand)
        if rad < best_rad:
            best_rad = rad
            best_pts = cand.copy()
            print(f"Restart {r}: {np.degrees(best_rad):.6f} deg")

    # Final refinement on best solution
    if best_pts is not None and time.time() < deadline - 10.0:
        fp = lloyd_covering_step(best_pts, grid_main, n_iters=15, alpha=0.3)
        fp = optimize_neural(fp, [200.0, 400.0], max_iter_per_stage=10)
        fp = exact_local_optimize(fp, max_iters=20)
        fr = covering_radius(fp)
        if fr < best_rad:
            best_rad = fr
            best_pts = fp

    if best_pts is None:
        best_pts = fibonacci_spiral(N)

    return best_pts
