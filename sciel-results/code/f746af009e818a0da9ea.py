import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_EPS = 1e-12
_DEADLINE = 140.0 # Safety margin before 150s kill

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_grid(res=30):
    """
    Generate a high-resolution grid of points on the sphere.
    Using a Fibonacci lattice for uniformity.
    """
    n = res * res * 4 # ~3600 points
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _hull_radii_and_poles(pts):
    """
    Compute covering radii and pole directions for all hull facets.
    Returns (radii, poles, faces, valid_mask) where radii/holes are for valid facets only.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.array([np.pi]), np.zeros((0,3)), faces, valid
        
        normals = cross[valid].copy()
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return radii, normals, faces, valid
    except Exception:
        return np.array([np.pi]), np.zeros((0,3)), np.empty((0,3),dtype=int), np.zeros(0,dtype=bool)

def _soft_covering_objective(x, grid_pts, beta):
    """
    Differentiable surrogate for the covering radius.
    Minimizes the maximum over grid points of the 'soft-min' distance to the nearest site.
    We minimize the sum of squared soft-min distances, or simply the max soft-min distance.
    Using sum of squared distances for smoothness and vectorization.
    
    x: shape (N, 3) flattened
    grid_pts: shape (M, 3)
    """
    pts = x.reshape(N, 3)
    # Normalize points to unit sphere to handle numerical drift during optimization
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / (norms + _EPS)
    
    # Compute distances: dots[i, j] = pts[i] @ grid_pts[j]
    # Distance squared = 2 - 2 * dot
    dots = pts @ grid_pts.T # (N, M)
    dist_sq = 2.0 - 2.0 * dots # (N, M)
    
    # Soft-min of dist_sq for each grid point j
    # min_i dist_sq[i, j] approximated by -1/beta * log(sum(exp(-beta * dist_sq)))
    # To avoid overflow, subtract max (or min of neg)
    # Let u = -dist_sq. max u = -min dist_sq.
    # log-sum-exp trick: max_u + log(sum(exp(u - max_u)))
    # soft_min = - (max_u + log(sum(exp(u - max_u)))) / beta ? No.
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # Let v = -beta * dist_sq.
    # v_max = max(v) = -beta * min(dist_sq)
    # sum(exp(v - v_max))
    # soft_min = -1/beta * (v_max + log(sum(exp(v - v_max))))
    
    v = -beta * dist_sq # (N, M)
    v_max = np.max(v, axis=0, keepdims=True) # (1, M)
    # exp(v - v_max) is stable
    exp_v = np.exp(v - v_max)
    log_sum = np.log(np.sum(exp_v, axis=0)) # (M,)
    
    soft_min_dist_sq = -(v_max[0] + log_sum) / beta # (M,)
    
    # Objective: Sum of squared soft-min distances? Or just sum?
    # Minimizing sum of soft_min_dist_sq is a good surrogate for minimizing max distance.
    # To make it more sensitive to the maximum, we can square them or use a power.
    # Let's use sum of squared soft_min_dist_sq to penalize large errors more.
    obj = np.sum(soft_min_dist_sq ** 2)
    
    # Gradient calculation
    # d obj / d x_k
    # Let S_j = soft_min_dist_sq[j]
    # obj = sum_j S_j^2
    # d obj / d S_j = 2 S_j
    # S_j = -1/beta * (v_max_j + log(sum_i exp(v_ij - v_max_j)))
    # v_ij = -beta * dist_sq_ij = -beta * (2 - 2 * dot_ij) = 2 * beta * (dot_ij - 1)
    # d v_ij / d dot_ij = 2 * beta
    # d dot_ij / d x_k = grid_pts[j] (if k is component of point i)
    
    # Let w_ij = exp(v_ij - v_max_j) / sum_m exp(v_mj - v_max_j)
    # This is the softmax weight of site i for grid point j.
    w = exp_v / np.sum(exp_v, axis=0, keepdims=True) # (N, M)
    
    # d S_j / d v_ij = -1/beta * (delta(i, argmax) - w_ij) ?
    # Actually, derivative of log-sum-exp w.r.t v_i is w_i.
    # S_j = -1/beta * (v_max_j + L_j)
    # d S_j / d v_ij = -1/beta * (d v_max_j / d v_ij + d L_j / d v_ij)
    # d v_max_j / d v_ij = 1 if i is max, else 0. But softmax handles the "soft" max.
    # Standard result: d/dv_i ( -1/beta * log(sum(exp(-beta x))) ) ...
    # Let's use the chain rule carefully.
    # f(v) = -1/beta * log(sum(exp(v)))
    # df/dv_i = -1/beta * exp(v_i) / sum(exp(v)) = -1/beta * w_i
    # So d S_j / d v_ij = -1/beta * w_ij
    
    # d obj / d v_ij = 2 * S_j * (-1/beta * w_ij)
    # d v_ij / d x_k = 2 * beta * grid_pts[j] (for the corresponding point i)
    
    # d obj / d x_k = sum_j (d obj / d v_ij) * (d v_ij / d x_k)
    # = sum_j (2 * S_j * (-1/beta * w_ij)) * (2 * beta * grid_pts[j])
    # = sum_j (-4 * S_j * w_ij) * grid_pts[j]
    
    # This sum is for a specific point i.
    # grad_x[i, :] = sum_j (-4 * S_j * w_ij) * grid_pts[j]
    
    # Vectorized:
    # coeff[i, j] = -4 * S_j * w_ij
    # grad_x[i, k] = sum_j coeff[i, j] * grid_pts[j, k]
    
    coeff = -4.0 * (soft_min_dist_sq[None, :] * w) # (N, M)
    grad_x = coeff @ grid_pts # (N, 3)
    
    # Reshape to match x
    grad_flat = grad_x.flatten()
    
    return obj, grad_flat

def _local_polish(pts, deadline):
    """
    Use subgradient descent on the actual convex hull covering radius
    to polish the result from the smooth surrogate optimization.
    """
    pts = pts.copy()
    best_pts = pts.copy()
    best_r = _hull_radii_and_poles(pts)[0].max()
    
    step = 0.01
    no_improve = 0
    rng = np.random.default_rng(42)
    
    max_iters = 1000
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        radii, poles, faces, valid = _hull_radii_and_poles(pts)
        if len(radii) == 0:
            break
            
        curr_r = radii.max()
        
        if curr_r < best_r - 1e-10:
            best_r = curr_r
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                break
        
        # Find worst facet
        worst_idx = np.argmax(radii)
        worst_face = faces[valid][worst_idx]
        pole = poles[worst_idx]
        
        cand = pts.copy()
        for v in worst_face:
            tangent_dir = pole - pts[v]
            tangent_dir = tangent_dir - np.dot(tangent_dir, pts[v]) * pts[v]
            norm_t = np.linalg.norm(tangent_dir)
            if norm_t > _EPS:
                tangent_dir /= norm_t
            else:
                tangent_dir = rng.normal(size=3)
                tangent_dir = tangent_dir - np.dot(tangent_dir, pts[v]) * pts[v]
                norm_t = np.linalg.norm(tangent_dir)
                if norm_t > _EPS:
                    tangent_dir /= norm_t
                else:
                    continue
            cand[v] = pts[v] + step * tangent_dir
            cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
        
        new_r = _hull_radii_and_poles(cand)[0].max()
        if new_r < curr_r:
            pts = cand
            step *= 1.05 # Slightly increase step if improving
            if step > 0.05:
                step = 0.05
        else:
            step *= 0.8
            
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # 1. Generate a dense grid for the surrogate
    # 20x20x4 = 1600 points is a good balance between resolution and speed
    grid_pts = _generate_grid(res=20)
    
    best_pts = None
    best_r = np.pi
    
    # 2. Multi-restart optimization using L-BFGS-B on the soft surrogate
    n_restarts = 5
    time_per_restart = (_DEADLINE - 10) / n_restarts # Reserve 10s for polish
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        if r == 0:
            pts_init = _initial_fib()
        else:
            # Random initialization with some structure
            theta = np.random.uniform(0, 2 * np.pi, N)
            z = np.random.uniform(-1, 1, N)
            rho = np.sqrt(np.clip(1 - z**2, 0, None))
            pts_init = np.column_stack([rho * np.cos(theta), rho * np.sin(theta), z])
            # Normalize
            pts_init /= np.linalg.norm(pts_init, axis=1, keepdims=True)
        
        x0 = pts_init.flatten()
        
        # Define bounds? L-BFGS-B works better with bounds, but sphere constraint is handled by normalization in objective.
        # However, L-BFGS-B assumes Euclidean space. The normalization inside the objective makes the landscape non-Euclidean.
        # It's often better to optimize on the sphere directly or use a retraction.
        # Since we normalize inside, the gradient is projected. 
        # To help L-BFGS-B, we can provide a line search that respects the sphere? No, standard L-BFGS-B is fine if the objective is smooth enough.
        
        # Increase beta (sharpness of soft-min) over time? 
        # Start with low beta (smooth, global) and increase (sharper, local).
        beta_start = 50.0
        beta_end = 200.0
        
        # Simple strategy: Run L-BFGS-B with a fixed moderate beta, then polish.
        # Or run multiple L-BFGS-B calls with increasing beta.
        
        # Let's do a single L-BFGS-B call with a reasonably high beta to capture local structure,
        # relying on the random restarts for global search.
        beta = 100.0
        
        res = minimize(
            _soft_covering_objective, 
            x0, 
            args=(grid_pts, beta), 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 100, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < np.pi:
            pts_opt = res.x.reshape(N, 3)
            pts_opt /= np.linalg.norm(pts_opt, axis=1, keepdims=True)
            
            # Evaluate true covering radius
            radii, _, _, _ = _hull_radii_and_poles(pts_opt)
            if len(radii) > 0:
                curr_r = radii.max()
                if curr_r < best_r:
                    best_r = curr_r
                    best_pts = pts_opt.copy()
        
        # Check time
        if time.time() > deadline - 10:
            break
            
    # 3. Polish the best result with hull-based subgradient descent
    if best_pts is not None:
        remaining = deadline - time.time()
        if remaining > 2:
            best_pts, best_r = _local_polish(best_pts, deadline - 1)
    else:
        # Fallback
        best_pts = _initial_fib()
    
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / (norms + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
