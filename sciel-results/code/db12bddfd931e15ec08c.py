"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_lattice_init(n):
    """Generate n points on a hexagonal lattice inside unit disk."""
    best_spacing = None
    best_points = None
    # Try multiple spacings
    for trial in range(50):
        spacing = 0.12 - trial * 0.0003
        if spacing < 0.04:
            break
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        r = np.hypot(points[:, 0], points[:, 1])
        mask = r <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            # Sort by distance from center, take closest n
            idx = np.argsort(r[mask], kind='stable')[:n]
            pts = points[idx]
            # Compute current score
            d = np.array([[np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])
                          for j in range(n)] for i in range(n)])
            np.fill_diagonal(d, np.inf)
            min_pair = d.min() / 2.0
            min_boundary = (1.0 - np.hypot(pts[:,0], pts[:,1])).min()
            score = min(min_pair, min_boundary)
            if best_points is None or score > (best_spacing if best_spacing is not None else 0):
                best_spacing = score
                best_points = pts
    if best_points is None:
        # Fallback: uniform random
        rng = np.random.default_rng(42)
        best_points = rng.uniform(-0.9, 0.9, size=(n, 2))
    return best_points


def _compute_score(points):
    """Compute the packing score: min(min pair distance/2, min boundary distance)."""
    n = len(points)
    # Pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    min_boundary = (1.0 - np.hypot(points[:, 0], points[:, 1])).min()
    return min(min_pair, min_boundary)


def _soft_min_objective(x, beta):
    """
    Smooth approximation of -min(constraints) using log-sum-exp.
    Constraints: for each pair (i,j): d_ij - 2r >= 0, and for each i: 1 - |c_i| - r >= 0.
    We maximize r, so minimize -r.
    """
    n = len(x) // 2
    pts = x[:2*n].reshape(n, 2)
    r = x[2*n]
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    
    # Constraint violations (want all >= 0)
    # Pair constraints: dists - 2r
    pair_viol = (dists - 2 * r)
    # Boundary constraints: 1 - |c_i| - r
    boundary_viol = 1.0 - np.hypot(pts[:, 0], pts[:, 1]) - r
    
    all_viol = np.concatenate([pair_viol.ravel(), boundary_viol])
    
    # Soft min: -1/beta * log(sum(exp(-beta * v))) ≈ min(v) for large beta
    # We want to maximize min(v), so minimize -soft_min
    # soft_min = -1/beta * log(sum(exp(-beta * v)))
    max_v = all_viol.max()
    if not np.isfinite(max_v):
        return -r
    exp_terms = np.exp(-beta * (all_viol - max_v))
    soft_min_val = max_v - np.log(exp_terms.sum()) / beta
    
    return -soft_min_val  # We want to maximize soft_min, so minimize its negative


def _soft_min_grad(x, beta):
    """Gradient of the soft-min objective."""
    n = len(x) // 2
    pts = x[:2*n].reshape(n, 2)
    r = x[2*n]
    
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    
    pair_viol = (dists - 2 * r)
    boundary_viol = 1.0 - np.hypot(pts[:, 0], pts[:, 1]) - r
    
    all_viol = np.concatenate([pair_viol.ravel(), boundary_viol])
    
    max_v = all_viol.max()
    if not np.isfinite(max_v):
        grad = np.zeros_like(x)
        grad[2*n] = -1.0
        return grad
    
    exp_terms = np.exp(-beta * (all_viol - max_v))
    weights = exp_terms / exp_terms.sum()
    
    # d(soft_min)/d(all_viol) = weights
    # soft_min = max_v - log(sum(exp(-beta*(v - max_v))))/beta
    # d(soft_min)/dv_i = weights[i]
    
    grad_viol = weights
    
    # Now backprop to x
    grad_x = np.zeros_like(x)
    
    # Pair violations: v_ij = dist_ij - 2r
    # d(dist_ij)/d(pts_i) = (pts_i - pts_j) / dist_ij
    # d(dist_ij)/d(pts_j) = (pts_j - pts_i) / dist_ij
    # d(-2r)/dr = -2
    
    # Boundary violations: v_i = 1 - |c_i| - r
    # d(v_i)/d(pts_i) = -pts_i / |c_i|
    # d(v_i)/dr = -1
    
    n_pairs = n * (n - 1)
    pair_w = grad_viol[:n_pairs]
    bound_w = grad_viol[n_pairs:]
    
    # Reshape pair weights to (n, n)
    pair_w_mat = pair_w.reshape(n, n)
    
    # Gradient for points from pair constraints
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = pair_w_mat[i, j]
            d_ij = dists[i, j]
            if d_ij > 1e-12:
                grad_x[2*i] += w * (pts[i, 0] - pts[j, 0]) / d_ij
                grad_x[2*i+1] += w * (pts[i, 1] - pts[j, 1]) / d_ij
                grad_x[2*j] += w * (pts[j, 0] - pts[i, 0]) / d_ij
                grad_x[2*j+1] += w * (pts[j, 1] - pts[i, 1]) / d_ij
            grad_x[2*n] += w * (-2.0)
    
    # Gradient for points from boundary constraints
    for i in range(n):
        w = bound_w[i]
        r_i = np.hypot(pts[i, 0], pts[i, 1])
        if r_i > 1e-12:
            grad_x[2*i] += w * (-pts[i, 0] / r_i)
            grad_x[2*i+1] += w * (-pts[i, 1] / r_i)
        grad_x[2*n] += w * (-1.0)
    
    # The objective is -soft_min, so negate the gradient
    return -grad_x


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = N
    # Initial guess
    pts0 = _hex_lattice_init(n)
    r0 = _compute_score(pts0)
    
    x0 = np.concatenate([pts0.ravel(), [r0 * 0.95]])
    
    best_x = x0.copy()
    best_score = r0
    
    # Adaptive beta schedule: start large (smooth), decrease (sharper)
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        def obj(x):
            return _soft_min_objective(x, beta)
        
        def grad(x):
            return _soft_min_grad(x, beta)
        
        # Bounds: points in [-1,1]^2, r in [0.01, 0.1]
        bounds = [(-1.0, 1.0)] * (2 * n) + [(0.01, 0.1)]
        
        try:
            result = minimize(
                obj, current_x, jac=grad,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.success or result.fun < obj(current_x):
                current_x = result.x
                score = _compute_score(current_x[:2*n].reshape(n, 2))
                if score > best_score:
                    best_score = score
                    best_x = current_x.copy()
        except Exception:
            pass
    
    # Final polish: use the best points found
    final_pts = best_x[:2*n].reshape(n, 2).copy()
    
    if not np.all(np.isfinite(final_pts)):
        return pts0

    # Ensure all points are strictly inside the unit circle.
    # The evaluator requires min_i (1 - |c_i|) to be part of the score.
    # If |c_i| >= 1, the score is 0 or invalid.
    
    dists_from_origin = np.hypot(final_pts[:, 0], final_pts[:, 1])
    
    # If any point is on or outside the boundary, project it inward.
    # We use a small epsilon to ensure strict inequality.
    eps = 1e-9
    outside_mask = dists_from_origin >= (1.0 - eps)
    
    if np.any(outside_mask):
        for i in np.where(outside_mask)[0]:
            d = dists_from_origin[i]
            if d > 0:
                # Project to radius (1 - eps)
                final_pts[i] *= (1.0 - eps) / d
            else:
                # If at origin, no change needed, but mask shouldn't trigger
                pass
        
        # Recalculate distances after projection
        dists_from_origin = np.hypot(final_pts[:, 0], final_pts[:, 1])
        
        # If projection caused issues (e.g. points moved too far apart or close?), 
        # we rely on the fact that moving inward generally helps boundary constraint 
        # but might hurt pair constraint if points were close to each other near boundary.
        # However, validity (being inside) is a hard constraint for a non-zero score.
        
    # Final sanity check: Are any points still outside?
    if np.any(dists_from_origin >= 1.0):
        # Fallback to the initial safe configuration
        return pts0
        
    # Optional: If the score of the repaired points is significantly worse than pts0, 
    # maybe return pts0? But usually optimization improves it.
    # Let's just return the repaired points.
    
    return final_pts

# EVOLVE-BLOCK-END
