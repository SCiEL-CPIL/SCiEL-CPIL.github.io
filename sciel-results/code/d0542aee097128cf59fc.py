"""Program for covering the sphere with 130 equal caps using soft-attention gradient optimization
with exact hull-based local refinement and entropic dual smoothing."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s hard limit

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

    # Dense grid for accurate gradient and evaluation
    grid_main = generate_grid(120, 240)

    def lloyd_covering_step(pts, grid, n_iters=50, alpha=0.5):
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

    def soft_covering_radius_and_grad(points, grid_points, sigma=0.05):
        """
        Vectorized soft covering radius and its gradient using smooth soft-min over all points.
        For each grid point, we compute a weighted sum over all configuration points with
        weights from a soft-min of the angular distances. This gives a smooth gradient.
        Returns (value, grad) where grad has shape (N, 3).
        """
        # Compute angular distances: (M, N)
        dots = grid_points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # (M, N)
        
        # Soft-min: for each grid point, compute weighted average of angles
        # using softmax weights over the negative angles (to get min)
        # w_j = exp(-angle_j / sigma) / sum_k exp(-angle_k / sigma)
        neg_scaled = -angles / sigma  # (M, N)
        max_neg = np.max(neg_scaled, axis=1, keepdims=True)  # (M, 1)
        exp_neg = np.exp(neg_scaled - max_neg)  # (M, N)
        weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # (M, N)
        
        # Soft-min value: weighted sum of angles
        soft_min_vals = np.sum(weights * angles, axis=1)  # (M,)
        
        # Soft-max of soft-min values (to get covering radius approximation)
        scaled_max = soft_min_vals / sigma
        max_scaled = np.max(scaled_max)
        exp_max = np.exp(scaled_max - max_scaled)
        max_weights = exp_max / np.sum(exp_max)  # (M,)
        soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))
        
        # Gradient computation:
        # d(soft_min_i)/d(points_j) = sum_k weights_ik * d(angle_ik)/d(points_j)
        # d(angle_ik)/d(points_j) = 0 if j != k, else -g_i / sqrt(1 - dot^2) projected to tangent
        
        # Compute per (i,j) gradient contributions
        # denom: (M, N)
        denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)
        
        # coeff[i,j] = max_weights[i] * weights[i,j]
        coeff = max_weights[:, np.newaxis] * weights  # (M, N)
        
        # Term 1: sum_i coeff[i,j] * (-g_i / denom[i,j])
        # = -sum_i (coeff[i,j] / denom[i,j]) * g_i
        # Let alpha[i,j] = coeff[i,j] / denom[i,j]
        alpha = coeff / denom  # (M, N)
        
        # Term1_j = -sum_i alpha[i,j] * g_i
        # This is: -(alpha.T @ grid_points)  -> shape (N, 3)
        term1 = -(alpha.T @ grid_points)  # (N, 3)
        
        # Term 2: sum_i coeff[i,j] * dots[i,j] / denom[i,j] * p_j
        # = sum_i alpha[i,j] * dots[i,j] * p_j
        # For each j: p_j * sum_i alpha[i,j] * dots[i,j]
        beta = alpha * dots  # (M, N)
        beta_sum = np.sum(beta, axis=0)  # (N,)
        term2 = beta_sum[:, np.newaxis] * points  # (N, 3)
        
        grad = term1 + term2  # (N, 3)
        
        return soft_max_val, grad

    def entropic_dual_objective_and_grad(points, grid_points, eps=0.05):
        """
        Entropic optimal transport dual formulation.
        The configuration points act as a discrete source measure (uniform weights 1/N).
        The grid points approximate the uniform sphere as target measure (uniform weights 1/M).
        
        The entropic OT cost is:
        W_eps = min_{pi in U} sum_{i,j} pi_ij * c_ij + eps * H(pi)
        where c_ij = angular_distance(points_j, grid_i)^2 (squared for better conditioning)
        
        Dual formulation:
        W_eps = max_{f,g} <f, mu> + <g, nu> - eps * (sum exp((f_i + g_j - c_ij)/eps) - 1)
        
        For our purpose, we want to minimize the covering radius, not the OT cost directly.
        Instead, we use the dual potential f_j (associated with source points) as a 
        smooth surrogate for the "load" each point bears. Points with high dual potential 
        are in "dense" regions and should move.
        
        Actually, let's use a simpler but effective approach: the Sinkhorn divergence
        D_eps(mu, nu) = W_eps(mu, nu) - 0.5 * W_eps(mu, mu) - 0.5 * W_eps(nu, nu)
        But computing W_eps(mu, mu) is expensive.
        
        Alternative: use the dual potential gradient directly.
        In the entropic OT dual, the optimal transport plan is:
        pi_ij = mu_i * nu_j * exp((f_i + g_j - c_ij)/eps)
        
        The gradient of the dual objective w.r.t. f_j (source potential) is:
        df_j = mu_j - sum_i pi_ij
        
        We can use this as a "force" that pushes points: if a point has more mass
        assigned to it than its share (mu_j = 1/N), it should move away from dense regions.
        
        Let's implement a simplified version: compute the Sinkhorn iterates to get
        the dual potentials, then use the imbalance as gradient.
        """
        M = len(grid_points)
        # Cost matrix: squared angular distances
        dots = grid_points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # (M, N)
        C = angles ** 2  # (M, N) - squared for better conditioning
        
        # Uniform marginals
        mu = np.ones(N) / N  # source weights
        nu = np.ones(M) / M  # target weights
        
        # Sinkhorn iterations to find dual potentials
        # f: (N,), g: (M,)
        f = np.zeros(N)
        g = np.zeros(M)
        
        # Precompute exp(-C/eps) for efficiency
        K = np.exp(-C / eps)  # (M, N)
        
        # Run Sinkhorn iterations
        u = np.ones(N) / N
        v = np.ones(M) / M
        
        for _ in range(20):
            # Update v: v_j = nu_j / (K^T u)_j
            Ku = K.T @ u  # (M,)
            Ku = np.maximum(Ku, 1e-30)
            v = nu / Ku
            # Update u: u_i = mu_i / (K v)_i
            Kv = K @ v  # (N,)
            Kv = np.maximum(Kv, 1e-30)
            u = mu / Kv
        
        # The transport plan is pi_ij = u_i * K_ij * v_j
        # The "load" on source point j is: sum_i pi_ij = u_j * (K v)_j
        # Ideally this should equal mu_j = 1/N
        # The imbalance is: load_j - mu_j
        # We want to minimize the maximum load, so use a smooth approximation
        
        loads = u * (K @ v)  # (N,)
        mu_target = np.ones(N) / N
        
        # Smooth max of (loads - mu_target)
        # Use log-sum-exp for smooth max
        scaled = (loads - mu_target) / eps
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        soft_max_val = eps * (max_scaled + np.log(np.sum(exp_scaled)))
        
        # Gradient: d(soft_max)/d(points_j)
        # = (1/eps) * exp(scaled_j - max_scaled) / sum(exp(scaled_k - max_scaled)) * d(loads_j)/d(points_j)
        max_weights = exp_scaled / np.sum(exp_scaled)  # (N,)
        
        # d(loads_j)/d(points_j) requires chain rule through Sinkhorn
        # This is complex. Instead, use a simpler heuristic gradient:
        # Points with high load should move toward directions that reduce their load.
        # The load is high when many grid points are close to this point.
        # So the gradient should push the point away from the centroid of nearby grid points.
        
        # Compute weighted centroid of grid points for each source point
        # Using the transport plan weights
        pi = np.outer(u, K) * v[np.newaxis, :]  # (M, N) - transport plan
        # pi[i,j] is mass from source j to target i
        
        # For each source point j, the "center of mass" of its assigned targets is:
        # com_j = sum_i pi_ij * grid_i / sum_i pi_ij
        total_mass = np.sum(pi, axis=0)  # (N,)
        total_mass = np.maximum(total_mass, 1e-30)
        
        # Weighted centroid
        com = (pi.T @ grid_points) / total_mass[:, np.newaxis]  # (N, 3)
        
        # Direction to move: away from com (since high load means too many targets here)
        # But we need to project onto the tangent plane
        grad = np.zeros_like(points)
        for j in range(N):
            if max_weights[j] < 1e-10:
                continue
            # Move away from com_j
            direction = -com[j]  # away from the center of mass of assigned targets
            # Project onto tangent plane at points[j]
            proj = direction - np.dot(direction, points[j]) * points[j]
            grad[j] = max_weights[j] * proj
        
        return soft_max_val, grad

    def optimize_with_annealing(pts, sigmas, max_iter_per_stage=20, use_entropic=False):
        """Multi-stage gradient optimization with sigma annealing."""
        if time.time() > deadline - 15.0:
            return pts
        
        current = pts.copy()
        
        for sigma in sigmas:
            if time.time() > deadline - 10.0:
                break
            
            if use_entropic:
                def objective_and_grad(x_flat, sigma=sigma):
                    pts_local = x_flat.reshape(N, 3)
                    norms = np.linalg.norm(pts_local, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-12, 1.0, norms)
                    pts_local = pts_local / norms
                    
                    val, grad = entropic_dual_objective_and_grad(pts_local, grid_main, eps=sigma)
                    return val, grad.flatten()
            else:
                def objective_and_grad(x_flat, sigma=sigma):
                    pts_local = x_flat.reshape(N, 3)
                    norms = np.linalg.norm(pts_local, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-12, 1.0, norms)
                    pts_local = pts_local / norms
                    
                    val, grad = soft_covering_radius_and_grad(pts_local, grid_main, sigma=sigma)
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

    def exact_local_optimize(points, grid_points, max_iters=50):
        """
        Final high-precision polish: iteratively move each point to reduce the
        maximum facet circumradius, using exact geometric computations.
        """
        pts = points.copy()
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute current covering radius from hull
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            
            # Find the worst facet
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
            
            # Move the vertices of the worst facet slightly toward the facet normal
            # to reduce the circumradius
            move_alpha = 0.01
            for idx in worst_facet:
                pts[idx] = pts[idx] + move_alpha * worst_normal
                pts[idx] /= np.linalg.norm(pts[idx])
        
        return pts

    best_pts = None
    best_rad = np.inf

    n_restarts = 8
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
        cand = lloyd_covering_step(pts, grid_main, n_iters=30, alpha=0.5)
        
        # Phase 2: Multi-stage soft-attention gradient optimization
        sigmas = [0.15, 0.08, 0.04, 0.02]
        cand = optimize_with_annealing(cand, sigmas, max_iter_per_stage=15, use_entropic=False)
        
        # Phase 2b: Try entropic dual refinement
        if time.time() < deadline - 30.0:
            cand_ent = optimize_with_annealing(cand, [0.05, 0.03, 0.02], max_iter_per_stage=10, use_entropic=True)
            rad_ent = covering_radius(cand_ent)
            rad_soft = covering_radius(cand)
            if rad_ent < rad_soft:
                cand = cand_ent
        
        # Phase 3: Final fine-tuning with very small sigma
        if time.time() < deadline - 20.0:
            cand = optimize_with_annealing(cand, [0.01, 0.005], max_iter_per_stage=10, use_entropic=False)
        
        # Phase 4: Exact hull-based local optimization
        if time.time() < deadline - 10.0:
            cand = exact_local_optimize(cand, grid_main, max_iters=30)
        
        rad = covering_radius(cand)
        if rad < best_rad:
            best_rad = rad
            best_pts = cand.copy()
            print(f"Restart {r}: {np.degrees(best_rad):.6f} deg")

    # Final refinement on best solution
    if best_pts is not None and time.time() < deadline - 10.0:
        fp = lloyd_covering_step(best_pts, grid_main, n_iters=15, alpha=0.3)
        fp = optimize_with_annealing(fp, [0.01, 0.005], max_iter_per_stage=10, use_entropic=False)
        fp = exact_local_optimize(fp, grid_main, max_iters=20)
        fr = covering_radius(fp)
        if fr < best_rad:
            best_rad = fr
            best_pts = fp

    if best_pts is None:
        best_pts = fibonacci_spiral(N)

    return best_pts
