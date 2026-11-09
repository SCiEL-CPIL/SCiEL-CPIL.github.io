import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200
DEADLINE = 140.0  # Safety margin for 150s limit

def compute_radius(points):
    """Compute the maximum common radius for given circle centers."""
    if points is None:
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    
    # Calculate min pairwise distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    
    # Calculate min wall distance
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return float(r)

def get_soft_min_value_grad(pts, lam):
    """
    Compute value and gradient of the negative soft-min radius.
    We minimize this function to maximize the soft-min radius.
    
    Soft-min of a set of values v_k:
    S(v) = -1/lam * log( sum exp(-lam * v_k) )
    
    We want to minimize F = -S(v) = 1/lam * log( sum exp(-lam * v_k) )
    
    The values v_k are:
    1. d_ij / 2 for all i < j
    2. dist_to_wall_i for all i
    """
    pts = np.asarray(pts, dtype=np.float64)
    N = pts.shape[0]
    
    # 1. Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 1e-10) # Avoid div by 0 if needed, though we use sqrt
    dist = np.sqrt(dist_sq)
    
    # We only need i < j
    iu = np.triu_indices(N, k=1)
    d_pairs = dist[iu] / 2.0
    
    # 2. Wall distances
    x = pts[:, 0]
    y = pts[:, 1]
    d_walls = np.minimum(
        np.minimum(x + 0.5, 0.5 - x),
        np.minimum(y + 0.5, 0.5 - y)
    )
    
    # Combine all constraints
    # v is a vector of all distances
    v = np.concatenate([d_pairs, d_walls])
    
    # Compute soft-min value
    # To avoid overflow in exp, subtract max? No, we have exp(-lam * v).
    # If v is small, -lam*v is large positive? No, v is positive. -lam*v is negative.
    # So exp(-lam*v) is small.
    # If v is large, exp(-lam*v) is very small.
    # The term with smallest v will dominate the sum.
    
    # For numerical stability, let's shift v by min(v)
    v_min = np.min(v)
    # S(v) = -1/lam * log( sum exp(-lam * (v - v_min) + lam * v_min) )
    #      = -1/lam * ( lam * v_min + log( sum exp(-lam * (v - v_min)) ) )
    #      = -v_min - 1/lam * log( sum exp(-lam * (v - v_min)) )
    
    shifted = v - v_min
    exp_terms = np.exp(-lam * shifted)
    sum_exp = np.sum(exp_terms)
    
    # Value of F = -S(v) = v_min + 1/lam * log(sum_exp)
    value = v_min + (1.0 / lam) * np.log(sum_exp)
    
    # Gradient of F w.r.t v
    # dF/dv_k = d/dv_k [ v_min + 1/lam * log(sum_exp) ]
    # Note: v_min depends on v, but in the soft-min approximation, we treat the log-sum-exp as the primary term.
    # Actually, the derivative of the log-sum-exp term is:
    # d/dv_k (1/lam * log(sum exp(-lam v_j))) = (1/lam) * (1/sum_exp) * exp(-lam v_k) * (-lam)
    # = - exp(-lam v_k) / sum_exp
    # 
    # So dF/dv_k = - p_k, where p_k is the probability weight of constraint k.
    # p_k = exp(-lam v_k) / sum_exp
    
    p = exp_terms / sum_exp
    
    # Now, chain rule to get gradient w.r.t pts
    # dF/dpts = sum_k (dF/dv_k) * (dv_k/dpts)
    # = - sum_k p_k * (dv_k/dpts)
    
    grad = np.zeros_like(pts)
    
    # Pairwise part
    # v_pair_idx corresponds to d_ij/2
    # d(d_ij/2)/d(pts_i) = (pts_i - pts_j) / d_ij
    # d(d_ij/2)/d(pts_j) = (pts_j - pts_i) / d_ij
    
    # p_pairs = p[:len(d_pairs)]
    p_pairs = p[:len(d_pairs)]
    
    # We need to sum over all pairs (i,j)
    # grad_i += - p_{ij} * (pts_i - pts_j) / d_ij
    # grad_j += - p_{ij} * (pts_j - pts_i) / d_ij
    
    # Efficient calculation:
    # diff is (N, N, 2). dist is (N, N).
    # We need to mask i < j.
    
    # Create a matrix of weights for the force
    # Force on i from j is proportional to p_{ij} * (pts_i - pts_j) / d_ij
    # Let's use the full distance matrix and mask
    
    # p_matrix is (N, N)
    p_matrix = np.zeros((N, N))
    p_matrix[iu] = p_pairs
    p_matrix[iu[1], iu[0]] = p_pairs # Symmetric
    
    # dist is (N, N)
    # Avoid division by zero
    safe_dist = np.where(dist > 1e-10, dist, 1e-10)
    
    # Force term: p_matrix * (1/safe_dist)
    # This is (N, N)
    force_weights = p_matrix / safe_dist
    
    # grad_pairs = - sum_j force_weights[i,j] * (pts[i] - pts[j])
    #             = - sum_j force_weights[i,j] * pts[i] + sum_j force_weights[i,j] * pts[j]
    #             = - pts[i] * sum_j force_weights[i,j] + sum_j force_weights[i,j] * pts[j]
    
    row_sums = np.sum(force_weights, axis=1)
    
    # sum_j force_weights[i,j] * pts[j] is matrix-vector product
    weighted_pts = force_weights.dot(pts)
    
    grad_pairs = - (pts * row_sums[:, np.newaxis]) + weighted_pts
    
    # Wall part
    # v_wall_i = min( x_i+0.5, 0.5-x_i, y_i+0.5, 0.5-y_i )
    # The gradient of the soft-min w.r.t wall distances is -p_wall_i
    # Then we need dv_wall_i / dpts_i.
    
    p_walls = p[len(d_pairs):]
    
    # For each point i, determine which wall is the "active" one in the soft-min sense?
    # No, the soft-min smooths the min.
    # v_wall_i is the min of 4 distances.
    # Let w_x_l = x_i + 0.5, w_x_r = 0.5 - x_i, w_y_b = y_i + 0.5, w_y_t = 0.5 - y_i
    # v_wall_i = min(w_x_l, w_x_r, w_y_b, w_y_t)
    #
    # The derivative of the soft-min w.r.t the *minimum* value is handled by the log-sum-exp.
    # But here v_wall_i is itself a min of 4 values.
    # This creates a nested min.
    # 
    # Alternative: Treat the 4 wall distances as separate constraints in the main sum.
    # This is more robust.
    
    # Let's restart the constraint list with 4 wall distances per point.
    # Total constraints: N*(N-1)/2 + 4*N
    
    # I will rewrite the function to use 4 separate wall constraints.
    # This simplifies the gradient calculation.
    
    return value, grad  # Placeholder, will rewrite below

def get_soft_min_value_grad_v2(pts, lam):
    """
    Compute value and gradient of the negative soft-min radius.
    Constraints:
    1. d_ij / 2 for i < j
    2. x_i + 0.5
    3. 0.5 - x_i
    4. y_i + 0.5
    5. 0.5 - y_i
    """
    pts = np.asarray(pts, dtype=np.float64)
    N = pts.shape[0]
    
    # 1. Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 1e-10)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    d_pairs = dist[iu] / 2.0
    
    # 2. Wall distances (4 per point)
    x = pts[:, 0]
    y = pts[:, 1]
    
    w_x_l = x + 0.5
    w_x_r = 0.5 - x
    w_y_b = y + 0.5
    w_y_t = 0.5 - y
    
    # Flatten all constraints
    v = np.concatenate([d_pairs, w_x_l, w_x_r, w_y_b, w_y_t])
    
    # Soft-min value
    v_min = np.min(v)
    shifted = v - v_min
    exp_terms = np.exp(-lam * shifted)
    sum_exp = np.sum(exp_terms)
    
    value = v_min + (1.0 / lam) * np.log(sum_exp)
    
    # Probabilities
    p = exp_terms / sum_exp
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # Pairwise part
    p_pairs = p[:len(d_pairs)]
    
    p_matrix = np.zeros((N, N))
    p_matrix[iu] = p_pairs
    p_matrix[iu[1], iu[0]] = p_pairs
    
    safe_dist = np.where(dist > 1e-10, dist, 1e-10)
    force_weights = p_matrix / safe_dist
    
    row_sums = np.sum(force_weights, axis=1)
    weighted_pts = force_weights.dot(pts)
    
    grad_pairs = - (pts * row_sums[:, np.newaxis]) + weighted_pts
    
    # Wall part
    # p_walls has length 4*N
    p_walls = p[len(d_pairs):]
    
    # Indices for walls
    # 0..N-1: x_l
    # N..2N-1: x_r
    # 2N..3N-1: y_b
    # 3N..4N-1: y_t
    
    p_x_l = p_walls[0:N]
    p_x_r = p_walls[N:2*N]
    p_y_b = p_walls[2*N:3*N]
    p_y_t = p_walls[3*N:4*N]
    
    # d(w_x_l)/dx = 1, d(w_x_r)/dx = -1, d(w_y_b)/dy = 1, d(w_y_t)/dy = -1
    # dF/dx_i = - p_x_l[i] * 1 - p_x_r[i] * (-1) = -p_x_l[i] + p_x_r[i]
    # dF/dy_i = - p_y_b[i] * 1 - p_y_t[i] * (-1) = -p_y_b[i] + p_y_t[i]
    
    grad_walls_x = -p_x_l + p_x_r
    grad_walls_y = -p_y_b + p_y_t
    
    grad[:, 0] += grad_walls_x
    grad[:, 1] += grad_walls_y
    
    total_grad = grad_pairs + grad_walls_x[:, np.newaxis] * 0 + grad_walls_y[:, np.newaxis] * 0 # Just to keep shapes? No, already added.
    # Wait, I added to grad directly.
    
    return value, (grad_pairs + np.column_stack([grad_walls_x, grad_walls_y]))

def hex_init(n_rows):
    """Generate a hexagonal lattice initialization with n_rows rows."""
    c_base = N // n_rows
    rem = N % n_rows
    counts = [c_base] * n_rows
    for i in range(rem):
        counts[i] += 1
    
    pts = []
    y = 0.0
    for i, c in enumerate(counts):
        xs = np.arange(c) - (c - 1) / 2.0
        if i % 2 == 1:
            xs += 0.5
        for x in xs:
            pts.append([x, y])
        y += np.sqrt(3) / 2.0
    
    pts = np.array(pts, dtype=np.float64)
    # Center and scale to fit in [-0.5, 0.5]
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    max_extent = max(np.max(np.abs(pts[:, 0])), np.max(np.abs(pts[:, 1])))
    if max_extent > 0:
        scale = 0.48 / max_extent
        pts *= scale
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_r = 0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(20):
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
        
    # 2. Hexagonal
    for n_rows in range(12, 19):
        pts = hex_init(n_rows)
        inits.append(pts)
        
    # 3. Grid
    m = 14
    grid_pts = []
    for i in range(m):
        for j in range(m):
            if len(grid_pts) < N:
                grid_pts.append([i/m - 0.5 + 0.5/m, j/m - 0.5 + 0.5/m])
    if len(grid_pts) >= N:
        grid_pts = np.array(grid_pts[:N])
        grid_pts += np.random.randn(N, 2) * 0.01
        inits.append(grid_pts)
    
    # Optimization loop
    for init in inits:
        if time.time() > deadline:
            break
            
        pts = init.copy()
        
        # Define objective for L-BFGS-B
        def obj(x):
            x = x.reshape(N, 2)
            val, grad = get_soft_min_value_grad_v2(x, lam=50.0)
            return val, grad.flatten()
        
        # L-BFGS-B
        try:
            res = minimize(obj, pts.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                pts_opt = res.x.reshape(N, 2)
                r = compute_radius(pts_opt)
                if r > best_r:
                    best_r = r
                    best_points = pts_opt.copy()
        except Exception:
            pass
            
    # Final polish with higher lambda if time remains
    if best_points is not None and time.time() < deadline - 10:
        def obj_polish(x):
            x = x.reshape(N, 2)
            val, grad = get_soft_min_value_grad_v2(x, lam=200.0)
            return val, grad.flatten()
            
        try:
            res = minimize(obj_polish, best_points.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10})
            if res.success:
                pts_polished = res.x.reshape(N, 2)
                r_polished = compute_radius(pts_polished)
                if r_polished > best_r:
                    best_r = r_polished
                    best_points = pts_polished.copy()
        except Exception:
            pass
            
    if best_points is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
