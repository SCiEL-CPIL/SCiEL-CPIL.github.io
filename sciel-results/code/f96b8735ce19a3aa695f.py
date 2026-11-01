import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings

N = 130

# Global time tracking
_start_time = 0.0
_total_time_limit = 145.0  # 5s safety margin before 150s kill

def _covering_radius_from_hull(points):
    """
    Compute the exact covering radius from the convex hull.
    For each facet (triangle) of the convex hull, the covering radius
    contribution is the angular distance from the outward normal (cap center)
    to the farthest vertex.
    """
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
        
        # Ensure normal points outward (away from origin)
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
        angles = np.arccos(dots)
        
        facet_radius = np.max(angles)
        max_radius = max(max_radius, facet_radius)
    
    return max_radius

def _soft_covering_radius_and_grad(points, grid_points, sigma=0.03):
    """
    Vectorized soft covering radius and its gradient using smooth soft-min/soft-max.
    Returns (value, grad) where grad has shape (N, 3).
    """
    # Compute angular distances: (M, N)
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # Soft-min over points for each grid point
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
    
    # Gradient computation
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)
    
    coeff = max_weights[:, np.newaxis] * weights  # (M, N)
    alpha = coeff / denom  # (M, N)
    
    # Term 1: -sum_i alpha[i,j] * g_i
    term1 = -(alpha.T @ grid_points)  # (N, 3)
    
    # Term 2: sum_i alpha[i,j] * dots[i,j] * p_j
    beta = alpha * dots  # (M, N)
    beta_sum = np.sum(beta, axis=0)  # (N,)
    term2 = beta_sum[:, np.newaxis] * points  # (N, 3)
    
    grad = term1 + term2  # (N, 3)
    
    return soft_max_val, grad

def _lloyd_step(points, grid_points, n_iters=5):
    """
    Perform Lloyd's algorithm iterations using grid-based Voronoi.
    """
    pts = points.copy()
    for _ in range(n_iters):
        if time.time() - _start_time > _total_time_limit - 3.0:
            break
        
        dots = grid_points @ pts.T
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        nearest = np.argmin(angles, axis=1)
        
        new_points = np.zeros_like(pts)
        
        for j in range(N):
            mask = (nearest == j)
            count = np.sum(mask)
            if count > 0:
                new_points[j] = np.sum(grid_points[mask], axis=0) / count
            else:
                new_points[j] = pts[j]
        
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        new_points = new_points / norms
        
        diff = np.max(np.linalg.norm(new_points - pts, axis=1))
        pts = new_points
        
        if diff < 1e-7:
            break
    
    return pts

def _sinkhorn_dual_optimize(points, grid_points, time_limit=30.0):
    """
    Optimize using a Sinkhorn/entropic optimal transport dual formulation.
    The configuration points act as a discrete source; the uniform sphere grid
    is the target. We minimize the entropic OT cost, which smooths the max-norm
    objective and provides stable gradients via the dual potentials.
    
    The dual potential u (on source points) is updated via Sinkhorn iterations,
    and the gradient of the OT cost w.r.t. source positions is computed.
    """
    start_time = time.time()
    pts = points.copy()
    n = len(pts)
    m = len(grid_points)
    
    # Initial learning rate
    lr = 0.01
    entropy_reg = 0.05  # Regularization parameter for entropic OT
    
    for it in range(300):
        if time.time() - start_time > time_limit:
            break
        
        # Compute cost matrix: C[i,j] = angular distance between pts[i] and grid_points[j]
        dots = grid_points @ pts.T  # (m, n)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # (m, n)
        
        # Sinkhorn iterations to find dual potentials
        # We want to minimize sum_{i,j} u_i * v_j * exp(-C[i,j]/eps)
        # where u is on source (n), v is on target (m)
        # Standard Sinkhorn: alternate updates
        
        eps = entropy_reg
        K = np.exp(-angles / eps)  # (m, n) kernel matrix
        
        # Initialize dual potentials
        u = np.ones(n) / n  # source potential
        v = np.ones(m) / m  # target potential
        
        # Sinkhorn iterations
        for _ in range(50):
            # Update v: v_j = 1 / sum_i u_i * K[i,j]
            Kt_u = K.T @ u  # (n,) -> actually (n,) since K is (m,n), K.T is (n,m)
            # Wait: K is (m, n). K.T is (n, m).
            # We want for each target j: v_j = 1 / sum_i u_i * K[i,j]
            # sum_i u_i * K[i,j] = (u @ K)[j]  -> shape (m,)
            row_sums = u @ K  # (m,)
            row_sums = np.maximum(row_sums, 1e-15)
            v = 1.0 / row_sums  # (m,)
            
            # Update u: u_i = 1 / sum_j v_j * K[i,j]
            col_sums = v @ K.T  # (n,)
            col_sums = np.maximum(col_sums, 1e-15)
            u = 1.0 / col_sums  # (n,)
        
        # The OT cost is sum_{i,j} u_i * v_j * C[i,j]
        # Gradient w.r.t. pts: 
        # d(cost)/d(pts_k) = sum_j u_k * v_j * d(C[k,j])/d(pts_k)
        # d(C[k,j])/d(pts_k) = -g_j / sqrt(1 - dots[k,j]^2) + dots[k,j]/sqrt(1-dots[k,j]^2) * pts_k
        # (projected to tangent space)
        
        # Compute gradient
        denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (m, n)
        
        # coeff[i,j] = u_j * v_i  (note: u is source index, v is target index)
        # Wait, let me be careful with indices.
        # C[i,j] where i is target (grid), j is source (pts)
        # u_j is source potential, v_i is target potential
        # Gradient for source point k:
        # grad_k = sum_i v_i * u_k * d(C[i,k])/d(pts_k)
        # d(C[i,k])/d(pts_k) = -g_i / denom[i,k] + dots[i,k]/denom[i,k] * pts_k
        
        # coeff[i,k] = v_i * u_k
        coeff = v[:, np.newaxis] * u[np.newaxis, :]  # (m, n)
        alpha = coeff / denom  # (m, n)
        
        # Term 1: -sum_i alpha[i,k] * g_i
        term1 = -(alpha.T @ grid_points)  # (n, 3)
        
        # Term 2: sum_i alpha[i,k] * dots[i,k] * pts_k
        beta = alpha * dots  # (m, n)
        beta_sum = np.sum(beta, axis=0)  # (n,)
        term2 = beta_sum[:, np.newaxis] * pts  # (n, 3)
        
        grad = term1 + term2  # (n, 3)
        
        # Project gradient to tangent space
        for k in range(n):
            radial = np.dot(grad[k], pts[k])
            grad[k] -= radial * pts[k]
        
        # Update
        pts = pts + lr * grad
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Decay learning rate
        lr *= 0.995
    
    return pts

def _gradient_descent_covering(points, grid_points, time_limit=15.0):
    """
    Gradient descent on the soft covering radius.
    """
    start_time = time.time()
    current = points.copy()
    
    lr = 0.005
    sigma = 0.03
    
    for it in range(200):
        if time.time() - start_time > time_limit:
            break
        
        val, grad = _soft_covering_radius_and_grad(current, grid_points, sigma)
        
        # Project gradient to tangent space
        for i in range(N):
            radial = np.dot(grad[i], current[i])
            grad[i] -= radial * current[i]
        
        # Update
        current = current + lr * grad
        current /= np.linalg.norm(current, axis=1, keepdims=True)
        
        # Decay learning rate and sigma
        lr *= 0.99
        if it % 20 == 19:
            sigma *= 0.8
    
    return current

def _exact_local_optimize(points, max_iters=100):
    """
    Final high-precision polish: iteratively move each point to reduce the
    maximum facet circumradius using exact geometric computations.
    """
    pts = points.copy()
    lr = 0.01
    
    for it in range(max_iters):
        if time.time() - _start_time > _total_time_limit - 3.0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        forces = np.zeros((N, 3))
        
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
            
            # For each vertex, compute the direction to move
            # to reduce the facet radius
            for idx, vert_idx in enumerate(simplex):
                # The vertex should move toward the cap center to reduce radius
                # Direction: from vertex to cap center, projected to tangent
                diff = normal - pts[vert_idx]
                radial = np.dot(diff, pts[vert_idx])
                tangent = diff - radial * pts[vert_idx]
                
                # Weight by how much this facet contributes
                weight = facet_radius / (np.pi / 2)
                forces[vert_idx] += weight * tangent
        
        # Update
        pts = pts + lr * forces
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Decay learning rate
        lr *= 0.98
    
    return pts

def _fibonacci_spiral(n):
    """Generate points on a Fibonacci spiral on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    global _start_time, _total_time_limit
    _start_time = time.time()
    _total_time_limit = 145.0
    
    best_score = np.inf
    best_points = None
    
    # Create grid points for optimization
    n_grid = 30000
    i_grid = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(np.clip(1.0 - z_grid * z_grid, 0, 1))
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), 
                                    r_grid * np.sin(phi_grid), 
                                    z_grid])
    
    # Generate multiple starting configurations
    rng = np.random.default_rng(12345)
    
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    
    # 2. Several random starts
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 1000 + 42)
        rp = rng_local.normal(size=(N, 3))
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        starts.append(rp)
    
    # 3. Icosahedron-based
    t = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_expanded = []
    rng_ico = np.random.default_rng(777)
    for v in ico_verts:
        for _ in range(11):
            jittered = v + rng_ico.normal(scale=0.15, size=3)
            jittered /= np.linalg.norm(jittered)
            ico_expanded.append(jittered)
    starts.append(np.array(ico_expanded[:N]))
    
    # Evaluate each start with optimization pipeline
    for idx, start_pts in enumerate(starts):
        if time.time() - _start_time > _total_time_limit - 10.0:
            break
        
        # Evaluate initial covering radius
        score = _covering_radius_from_hull(start_pts)
        if score < best_score:
            best_score = score
            best_points = start_pts.copy()
        
        # Step 1: Lloyd's algorithm
        lloyd_points = _lloyd_step(start_pts, grid_points, n_iters=10)
        score = _covering_radius_from_hull(lloyd_points)
        if score < best_score:
            best_score = score
            best_points = lloyd_points.copy()
        
        # Step 2: Sinkhorn dual optimization
        if time.time() - _start_time < _total_time_limit - 20.0:
            sinkhorn_points = _sinkhorn_dual_optimize(lloyd_points, grid_points, time_limit=25.0)
            score = _covering_radius_from_hull(sinkhorn_points)
            if score < best_score:
                best_score = score
                best_points = sinkhorn_points.copy()
        
        # Step 3: Gradient descent on soft covering radius
        if time.time() - _start_time < _total_time_limit - 15.0:
            gd_points = _gradient_descent_covering(best_points, grid_points, time_limit=15.0)
            score = _covering_radius_from_hull(gd_points)
            if score < best_score:
                best_score = score
                best_points = gd_points.copy()
        
        # Step 4: Exact local optimization
        if time.time() - _start_time < _total_time_limit - 5.0:
            exact_points = _exact_local_optimize(best_points, max_iters=50)
            score = _covering_radius_from_hull(exact_points)
            if score < best_score:
                best_score = score
                best_points = exact_points.copy()
    
    # Fallback
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    # Ensure all points are normalized
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
