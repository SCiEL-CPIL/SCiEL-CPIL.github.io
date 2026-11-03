"""Rewritten program for packing 50 equal circles in a unit circle."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Place 50 equal, non-overlapping circles inside the unit circle.
    Returns an array of shape (N, 2) with the centres.
    """
    # Initialize with a hexagonal lattice pattern
    # We want points roughly spread out, so start with a reasonable density
    # Hexagonal packing in a disk: use concentric rings or a lattice cut
    
    # Let's use a simple initialization: points on concentric rings
    # or a grid that we then optimize
    
    # Better: use a hexagonal lattice clipped to the disk, then take N points
    # But since we'll optimize, let's just use a reasonable initial spread
    
    # Initialize on a hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                rows.append([x, y])
    points_init = np.array(rows)
    
    if len(points_init) < N:
        # Fallback: random initialization
        points_init = np.random.rand(N, 2) * 2 - 1
        mask = np.hypot(points_init[:, 0], points_init[:, 1]) < 1.0
        points_init = points_init[mask]
        if len(points_init) < N:
            points_init = np.random.rand(N, 2) * 0.8
    
    # Take the first N points (or all if fewer)
    if len(points_init) >= N:
        points_init = points_init[:N]
    else:
        # Pad with random points
        extra = np.random.rand(N - len(points_init), 2) * 0.8
        points_init = np.vstack([points_init, extra])
    
    # Define the objective function to minimize
    # We want to maximize the minimum distance, so we minimize the negative
    # of a smooth approximation of the minimum pairwise distance and boundary distance
    
    def objective(x):
        """
        x: flattened array of shape (2*N,)
        Returns a scalar to minimize.
        """
        pts = x.reshape(-1, 2)
        
        # Pairwise distances
        # Compute all pairwise distances
        dx = pts[:, 0, np.newaxis] - pts[:, 0, np.newaxis].T
        dy = pts[:, 1, np.newaxis] - pts[:, 1, np.newaxis].T
        dists = np.sqrt(dx * dx + dy * dy)
        
        # Set diagonal to inf so it doesn't affect min
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances: 1 - |p_i|
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dists = 1.0 - radii
        
        # We want to maximize the minimum of:
        #   - half of each pairwise distance (since radius = dist/2 for touching)
        #   - boundary distance
        # So minimize the negative of a soft-min of these quantities
        
        # Soft-min approximation using log-sum-exp with small temperature
        # Actually, we want to maximize min, so minimize -min
        # Use a smooth approximation: -log-sum-exp(-values / T) * T
        
        # Combine all constraints into one vector
        # Pairwise: dists / 2 (the radius if circles touch)
        # Boundary: boundary_dists
        
        # We need to be careful: the actual radius is min(min_pairwise/2, min_boundary)
        # So we want to maximize this value.
        
        # Let's define values = [dists[i,j]/2 for i<j, boundary_dists[i] for i]
        # Then maximize min(values), i.e., minimize -min(values)
        
        # Use soft-min: -T * log(sum(exp(-v/T)))
        T = 0.01  # Temperature for soft-min
        
        # Get upper triangle of dists
        iu = np.triu_indices_from(dists, k=1)
        pairwise_vals = dists[iu] / 2.0
        boundary_vals = boundary_dists
        
        all_vals = np.concatenate([pairwise_vals, boundary_vals])
        
        # Soft-min
        neg_vals = -all_vals / T
        # Clip for numerical stability
        neg_vals = np.clip(neg_vals, -100, 100)
        log_sum = np.log(np.sum(np.exp(neg_vals)))
        soft_min = -T * log_sum
        
        # We want to maximize soft_min, so minimize -soft_min
        return -soft_min
    
    def objective_and_grad(x):
        """
        Compute objective and gradient.
        """
        pts = x.reshape(-1, 2)
        
        # Pairwise distances
        dx = pts[:, 0, np.newaxis] - pts[:, 0, np.newaxis].T
        dy = pts[:, 1, np.newaxis] - pts[:, 1, np.newaxis].T
        dists = np.sqrt(dx * dx + dy * dy + 1e-10)  # Small epsilon for stability
        
        # Set diagonal to large value
        np.fill_diagonal(dists, 1e10)
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dists = 1.0 - radii
        
        # Get upper triangle indices
        iu = np.triu_indices_from(dists, k=1)
        pairwise_vals = dists[iu] / 2.0
        boundary_vals = boundary_dists
        
        all_vals = np.concatenate([pairwise_vals, boundary_vals])
        
        T = 0.01
        neg_vals = -all_vals / T
        neg_vals = np.clip(neg_vals, -100, 100)
        exp_vals = np.exp(neg_vals)
        sum_exp = np.sum(exp_vals)
        log_sum = np.log(sum_exp + 1e-10)
        soft_min = -T * log_sum
        obj = -soft_min
        
        # Gradient computation
        # d(soft_min)/d(all_vals) = exp(neg_vals) / sum_exp = softmax weights
        weights = exp_vals / (sum_exp + 1e-10)
        
        # Gradient of soft_min w.r.t. all_vals
        # soft_min = -T * log(sum(exp(-v/T)))
        # d(soft_min)/dv_i = -T * (1/sum_exp) * exp(-v_i/T) * (-1/T) = exp(-v_i/T) / sum_exp = weights[i]
        # So d(obj)/dv_i = -weights[i]
        
        grad_all = -weights
        
        # Now propagate gradient back to x
        grad_pts = np.zeros_like(pts)
        
        # Pairwise part
        n_pairs = len(iu)
        pair_grads = grad_all[:n_pairs]
        
        # For each pair (i,j) with i<j, the value is dists[i,j]/2
        # d(dist/2)/d(pts[i]) = -0.5 * (pts[i] - pts[j]) / dist
        # d(dist/2)/d(pts[j]) = 0.5 * (pts[i] - pts[j]) / dist
        
        for idx in range(n_pairs):
            i, j = iu[0][idx], iu[1][idx]
            d = dists[i, j]
            if d < 1e-6:
                continue
            diff = pts[i] - pts[j]
            grad_pts[i] += -0.5 * diff / d * pair_grads[idx]
            grad_pts[j] += 0.5 * diff / d * pair_grads[idx]
        
        # Boundary part
        n_boundary = len(boundary_vals)
        boundary_grads = grad_all[n_pairs:n_pairs + n_boundary]
        
        # boundary_dists = 1 - |p_i|
        # d(boundary_dist)/d(pts[i]) = -pts[i] / |p_i|
        for i in range(N):
            r = radii[i]
            if r < 1e-6:
                continue
            grad_pts[i] += -pts[i] / r * boundary_grads[i]
        
        # Flatten gradient
        grad = grad_pts.flatten()
        
        return obj, grad
    
    # Use L-BFGS-B optimizer with gradient
    x0 = points_init.flatten()
    
    # Multiple restarts with different initializations
    best_result = None
    best_score = -np.inf
    
    for restart in range(5):
        if restart == 0:
            x_start = x0.copy()
        else:
            # Add some random perturbation
            x_start = x0.copy() + np.random.normal(0, 0.1, size=2 * N)
            # Project back into the unit disk approximately
            pts = x_start.reshape(-1, 2)
            radii = np.hypot(pts[:, 0], pts[:, 1])
            mask = radii > 0.95
            if np.any(mask):
                pts[mask] *= 0.95 / radii[mask, np.newaxis]
            x_start = pts.flatten()
        
        try:
            result = minimize(
                objective_and_grad,
                x_start,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            pts = result.x.reshape(-1, 2)
            
            # Compute the actual score
            dx = pts[:, 0, np.newaxis] - pts[:, 0, np.newaxis].T
            dy = pts[:, 1, np.newaxis] - pts[:, 1, np.newaxis].T
            dists = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(dists, np.inf)
            min_pairwise = np.min(dists) / 2.0
            
            radii = np.hypot(pts[:, 0], pts[:, 1])
            min_boundary = np.min(1.0 - radii)
            
            score = min(min_pairwise, min_boundary)
            
            if score > best_score:
                best_score = score
                best_result = pts.copy()
                
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to initial points
        return points_init
    
    # Ensure all points are inside the unit circle
    radii = np.hypot(best_result[:, 0], best_result[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        best_result[mask] *= 0.999 / radii[mask, np.newaxis]
    
    return best_result
# EVOLVE-BLOCK-END
