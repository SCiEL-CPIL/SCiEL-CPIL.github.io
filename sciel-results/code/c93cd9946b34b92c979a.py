"""Pack 200 equal circles in a unit square to maximize common radius.

Uses a smooth soft-min surrogate objective optimized with L-BFGS-B.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, leave margin before 150s kill


def _compute_radius(pts):
    """Compute the achievable radius for a set of points inside [-0.5, 0.5]^2."""
    if len(pts) < N:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    if np.any(~np.isfinite(pts)):
        return 0.0
    # Check inside square
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0
    # Min distance to wall
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    # Min pairwise distance
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2.0)
    return max(r, 0.0)


def _generate_hexagonal_grid(n):
    """Generate a hexagonal (close-packing) grid arrangement for n points."""
    # Approximate number of rows
    # In hex packing, density is pi / (2*sqrt(3)) * (1/r)^2 per unit area?
    # Area of unit square is 1. Number of points N ~ Area / (Area per point).
    # Area per point in hex lattice with spacing d is d^2 * sqrt(3)/2.
    # d ~ sqrt(2 * Area / (N * sqrt(3))) = sqrt(2 / (200 * 1.732)) ~ sqrt(1/173) ~ 0.076
    # Let's just iterate rows and fill.
    
    pts_list = []
    # Estimate spacing
    # If we have R rows, each row has approx C columns.
    # C * R = N.
    # Spacing x: dx, Spacing y: dy = dx * sqrt(3)/2.
    # Total width ~ C*dx, Total height ~ R*dy.
    # We want C*dx <= 1 and R*dy <= 1.
    
    # Try to find R, C such that R*C >= N and fits.
    # Let's start with a square grid of 14x15 or similar and offset.
    # Better: iterate row by row.
    
    # Let's guess R ~ sqrt(N * sqrt(3) / 2) ~ sqrt(200 * 0.866) ~ sqrt(173) ~ 13.
    # So roughly 13-14 rows.
    
    # We will adjust spacing to fit exactly N points in the square.
    # Let's try R rows.
    # Points in row i: if i even, C, if i odd, C-1 (or vice versa).
    # Total N = R*C - floor(R/2).
    # Solve for C: C ~ (N + R/2) / R.
    
    best_pts = None
    best_r = -1
    
    for R in range(10, 20):
        # Calculate C
        # N = R*C - floor(R/2)  => C = (N + floor(R/2)) / R
        C = int(np.ceil((N + R // 2) / R))
        
        # Generate points
        pts = []
        # Spacing
        # Width used: (C-1)*dx + offset. Offset is dx/2 for odd rows.
        # Max x extent: (C-1)*dx + dx/2 = (C - 0.5)*dx.
        # We need (C - 0.5)*dx <= 1. So dx <= 1 / (C - 0.5).
        # Height used: (R-1)*dy. dy = dx * sqrt(3)/2.
        # We need (R-1)*dy <= 1. So dx <= 1 / ((R-1)*sqrt(3)/2).
        
        dx_x = 1.0 / (C - 0.5)
        dx_y = 1.0 / ((R - 1) * np.sqrt(3) / 2) if R > 1 else 1.0
        dx = min(dx_x, dx_y)
        dy = dx * np.sqrt(3) / 2
        
        # Center the grid
        # Total width = (C-1)*dx + dx/2 (for offset) = (C - 0.5)*dx
        # Total height = (R-1)*dy
        # Start x: -0.5 + (1 - (C - 0.5)*dx) / 2 ? No, just center the bounding box.
        # Bounding box width: (C - 0.5) * dx
        # Bounding box height: (R - 1) * dy
        
        start_x = -0.5 + (1.0 - (C - 0.5) * dx) / 2.0
        start_y = -0.5 + (1.0 - (R - 1) * dy) / 2.0
        
        for r in range(R):
            y = start_y + r * dy
            if r % 2 == 0:
                x_start = start_x
                n_pts = C
            else:
                x_start = start_x + dx / 2.0
                n_pts = C - 1
            
            for c in range(n_pts):
                x = x_start + c * dx
                pts.append([x, y])
                if len(pts) == N:
                    break
            if len(pts) == N:
                break
        
        if len(pts) == N:
            pts = np.array(pts)
            r_val = _compute_radius(pts)
            if r_val > best_r:
                best_r = r_val
                best_pts = pts
    
    if best_pts is None:
        # Fallback to simple grid
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
        
    return best_pts


def _soft_min_obj_and_grad(x, beta):
    """
    Compute the soft-min objective (to be minimized) and its gradient.
    Objective: (1/beta) * log(sum(exp(-beta * c_i)))
    This is an approximation of -min(c_i). Minimizing this maximizes min(c_i).
    """
    pts = x.reshape(N, 2)
    n = N
    
    # Compute pairwise distances
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, 1e10)
    dist = np.sqrt(dist_sq)
    
    # Pairwise constraints: c_ij = dist_ij / 2
    # We only need upper triangle to avoid double counting, but for vectorization
    # it's easier to compute all and mask, or just sum over all i<j.
    # Let's use the full matrix and divide weights by 2 later? 
    # No, the sum in the log is over all distinct constraints.
    # Constraints:
    # 1. Pairwise: i < j, c = dist/2
    # 2. Walls: 4 per point.
    
    # To be efficient, we compute gradients directly.
    
    # Weights for pairwise constraints
    # w_ij = exp(-beta * dist_ij / 2)
    # Note: dist is symmetric.
    
    # Wall constraints
    # c_wall_k = min(pts[k,0]+0.5, 0.5-pts[k,0], pts[k,1]+0.5, 0.5-pts[k,1])
    # Actually, the soft min sums exp(-beta * c).
    # So we sum over all 4 wall distances for each point.
    
    wall_dists = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pairwise distances (upper triangle)
    iu = np.triu_indices(n, 1)
    pair_dists = dist[iu] / 2.0
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Calculate weights
    # To avoid overflow/underflow, subtract max
    max_c = np.max(all_dists)
    exps = np.exp(-beta * (all_dists - max_c))
    sum_exps = np.sum(exps)
    
    # Objective value
    # L = (1/beta) * log(sum_exps) - max_c
    # Wait: log(sum exp(-beta c_i)) = log(sum exp(-beta (c_i - max_c) * exp(-beta max_c)))
    # = -beta max_c + log(sum exp(-beta (c_i - max_c)))
    # So L = (1/beta) * (-beta max_c + log(sum exps)) = -max_c + (1/beta) log(sum_exps)
    obj = -max_c + (1.0 / beta) * np.log(sum_exps)
    
    # Gradient
    # dL/dx_k = - sum_i (w_i / sum_w) * (d c_i / d x_k)
    # where w_i = exp(-beta c_i).
    # Note: the weights in the gradient formula are normalized probabilities p_i = w_i / sum_w.
    # p_i = exps[i] / sum_exps
    
    grads = np.zeros_like(pts)
    
    # Pairwise gradients
    # For pair (i, j), c = ||pi - pj|| / 2
    # dc/dpi = (pi - pj) / (2 * ||pi - pj||)
    # dc/dpj = (pj - pi) / (2 * ||pi - pj||)
    
    # We have weights for each pair in exps[:len(iu[0])]
    pair_weights = exps[:len(iu[0])] / sum_exps
    
    # Vectorized pairwise gradient accumulation
    # diff[i, j] is pi - pj
    # We need to sum over j for each i.
    # grad_i += sum_j p_ij * (pi - pj) / (2 * dist_ij)
    
    # dist is the full matrix.
    # p_matrix is zero on diagonal and symmetric?
    # No, exps are for i<j.
    # Let's construct a weight matrix W where W[i,j] = p_ij if i<j, p_ji if i>j, 0 if i=j.
    # Actually, since the sum is over i<j, and the contribution to grad_i from pair (i,j) is p_ij * (pi-pj)/(2*dist),
    # and contribution to grad_j is p_ij * (pj-pi)/(2*dist).
    
    # Let's create a matrix of weights for all i,j
    P = np.zeros((n, n))
    P[iu[0], iu[1]] = pair_weights
    P[iu[1], iu[0]] = pair_weights
    
    # Now, grad_i = sum_j P[i,j] * (pi - pj) / (2 * dist_ij)
    # Let V[i,j] = (pi - pj) / (2 * dist_ij)
    # V is antisymmetric: V[j,i] = -V[i,j]
    
    # We can compute this as a matrix multiplication if we arrange it correctly.
    # grad_i = sum_j P[i,j] * V[i,j]
    
    # Compute V
    # dist is n x n.
    # diff is n x n x 2.
    # V_x[i,j] = diff[i,j,0] / (2 * dist[i,j])
    # V_y[i,j] = diff[i,j,1] / (2 * dist[i,j])
    
    # Avoid division by zero (diagonal)
    # dist is inf on diagonal, so 1/dist is 0.
    inv_dist_2 = 1.0 / (2.0 * dist)
    # Set diagonal to 0 explicitly to be safe
    np.fill_diagonal(inv_dist_2, 0.0)
    
    V_x = diff[:, :, 0] * inv_dist_2
    V_y = diff[:, :, 1] * inv_dist_2
    
    grads[:, 0] = np.sum(P * V_x, axis=1)
    grads[:, 1] = np.sum(P * V_y, axis=1)
    
    # Wall gradients
    # Wall constraints are indexed from len(iu[0]) to end.
    wall_weights = exps[len(iu[0]):] / sum_exps
    
    # Structure of wall_dists:
    # 0..n-1: pts[:,0] + 0.5  (grad [1, 0])
    # n..2n-1: 0.5 - pts[:,0] (grad [-1, 0])
    # 2n..3n-1: pts[:,1] + 0.5 (grad [0, 1])
    # 3n..4n-1: 0.5 - pts[:,1] (grad [0, -1])
    
    w_x_pos = wall_weights[0:n]
    w_x_neg = wall_weights[n:2*n]
    w_y_pos = wall_weights[2*n:3*n]
    w_y_neg = wall_weights[3*n:4*n]
    
    grads[:, 0] += w_x_pos - w_x_neg
    grads[:, 1] += w_y_pos - w_y_neg
    
    # The objective function returns the value to minimize.
    # The gradient is the gradient of the objective to minimize.
    # In the derivation: dL/dx = - sum p_i dc_i/dx.
    # So we return -grads?
    # Let's check.
    # L = -max_c + ...
    # dL/dx_k = - sum_i p_i dc_i/dx_k.
    # So the gradient of the objective is - (sum p_i dc_i/dx_k).
    # My `grads` variable currently holds sum p_i dc_i/dx_k.
    # So I should return -grads.
    
    return obj, -grads


def construct():
    t0 = time.time()
    
    # Generate initial configuration
    pts_init = _generate_hexagonal_grid(N)
    x0 = pts_init.flatten()
    
    best_pts = pts_init
    best_r = _compute_radius(pts_init)
    
    # Optimization
    # Use L-BFGS-B with soft-min objective
    # Beta controls the sharpness of the min approximation.
    # Smaller beta -> smoother, easier to optimize, but less accurate.
    # Larger beta -> sharper, closer to true min, but harder (nonsmooth-like).
    # We can anneal beta: start small, increase.
    
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() - t0 > _TIME_LIMIT - 10:
            break
            
        try:
            # Bounds: x in [-0.5, 0.5]
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            res = minimize(
                _soft_min_obj_and_grad,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            x_opt = res.x.reshape(N, 2)
            r_opt = _compute_radius(x_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = x_opt
                x0 = x_opt.copy()
                
        except Exception:
            # If optimization fails, continue with current best
            pass
            
    # Final polish: try to push the minimum distance further
    # Identify the active constraints and move points apart
    # This is a simple local search on the active set
    
    # Compute distances
    D = cdist(best_pts, best_pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    wall = np.min([
        np.min(best_pts[:, 0]) + 0.5,
        0.5 - np.max(best_pts[:, 0]),
        np.min(best_pts[:, 1]) + 0.5,
        0.5 - np.max(best_pts[:, 1]),
    ])
    
    current_r = min(wall, min_pair / 2.0)
    
    # If we have time, do a few more iterations with higher beta
    if time.time() - t0 < _TIME_LIMIT - 5:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                x0,
                args=(200.0,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            x_opt = res.x.reshape(N, 2)
            r_opt = _compute_radius(x_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_pts = x_opt
        except Exception:
            pass
            
    return best_pts
