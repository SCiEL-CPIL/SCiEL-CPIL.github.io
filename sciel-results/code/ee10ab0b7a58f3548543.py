"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _compute_score(points):
    """Compute the packing score: min pair distance / 2, min distance to boundary."""
    n = len(points)
    radii = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - radii)
    if min_boundary < 0:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = np.min(dists[:, 1])
    return min(min_pair / 2.0, min_boundary)


def _soft_min_obj_and_grad(x, n, beta, k_nn=5):
    """
    Soft-min objective using log-sum-exp for smoothness.
    Uses k-nearest neighbors to provide a more robust gradient than 1-NN.
    """
    points = x.reshape(n, 2)
    radii = np.linalg.norm(points, axis=1)
    
    # Check validity
    if np.any(radii > 1.0):
        penalty = np.sum(np.maximum(0, radii - 1.0)) ** 2
        return 1e6 + penalty, np.zeros_like(x)
    
    # Pairwise distances via KDTree for efficiency
    tree = cKDTree(points)
    # Query k_nn nearest neighbors (including self at index 0)
    # We want the distances to the k_nn nearest neighbors (excluding self)
    # So query k_nn + 1
    dists, idxs = tree.query(points, k=k_nn + 1)
    
    # dists[:, 0] is 0 (self)
    # dists[:, 1:k_nn+1] are distances to k_nn neighbors
    # idxs[:, 1:k_nn+1] are indices of k_nn neighbors
    
    # We will create a list of constraints:
    # For each point i, and each of its k_nn neighbors j:
    #   constraint value = dist(i, j) / 2
    # Note: This double counts pairs. That's fine for soft-min, it just changes the weights.
    # However, to keep the magnitude similar, we might need to adjust beta or accept the change.
    # Actually, double counting makes the soft-min "sharper" effectively if beta is fixed.
    # Let's stick to unique pairs if possible? No, KDTree doesn't give unique pairs easily.
    # We will use the k_nn neighbors for each point.
    
    # Collect all pair distances and indices
    pair_dists = np.zeros((n, k_nn))
    pair_idxs = np.zeros((n, k_nn), dtype=int)
    
    for i in range(n):
        # idxs[i, 0] is i itself
        # We want idxs[i, 1:k_nn+1]
        pair_idxs[i, :] = idxs[i, 1:k_nn+1]
        pair_dists[i, :] = dists[i, 1:k_nn+1]
        
    # Boundary distances
    boundary_dists = 1.0 - radii
    
    # Flatten constraints
    # Total constraints: n * k_nn (pairs) + n (boundary)
    # Note: Pairs are duplicated (i,j) and (j,i). 
    # This is acceptable for a surrogate.
    
    all_pair_dists = pair_dists.flatten() / 2.0
    all_constraints = np.concatenate([all_pair_dists, boundary_dists])
    
    # Filter out any non-finite or zero distances (shouldn't happen with unique points)
    # If points coincide, dist is 0.
    valid_mask = np.isfinite(all_constraints) & (all_constraints > 1e-12)
    
    if not np.any(valid_mask):
         return 1e6, np.zeros_like(x)
         
    a = all_constraints[valid_mask]
    
    # Soft-min via log-sum-exp
    a_max = np.max(a)
    if a_max < -1e10:
        return 1e6, np.zeros_like(x)
    
    exp_terms = np.exp(-beta * (a - a_max))
    sum_exp = np.sum(exp_terms)
    if sum_exp == 0 or not np.isfinite(sum_exp):
        return 1e6, np.zeros_like(x)
        
    log_sum = np.log(sum_exp)
    t_soft = a_max - log_sum / beta
    
    # Weights for each valid constraint
    weights = exp_terms / sum_exp
    
    # Map weights back to original arrays
    # We need to assign weights to the specific (i, j) pairs and boundary terms
    # This is tricky because we flattened and masked.
    # Let's compute gradients directly without flattening if possible, or use a more direct mapping.
    
    # Alternative: Compute gradient by looping over i and its neighbors
    # This is O(N * k_nn) which is fine.
    
    grad = np.zeros_like(points)
    
    # We need the weights for each (i, j) pair and each boundary i.
    # Let's reconstruct the weights array in the original shape before masking?
    # Easier: Just re-calculate the "importance" or use the fact that the gradient is a weighted sum of unit vectors.
    
    # Let's go back to the unmasked arrays and compute weights there, handling zeros.
    # If dist is 0, weight is exp(0)=1? No, if dist is 0, a=0.
    # The soft-min is dominated by the smallest a.
    
    # Let's use the masked weights and map them back.
    # This is complex. Let's use a simpler approach for the gradient:
    # The gradient of the soft-min is the weighted sum of the gradients of the individual terms.
    # term_grad[i, j] = (p_i - p_j) / (2 * dist_ij)
    
    # We will iterate over all i and its k_nn neighbors.
    # We need the weight associated with the constraint (i, j).
    # The constraint value was a = dist/2.
    # The weight was exp(-beta * (a - a_max)) / sum.
    
    # To avoid the mapping headache, let's compute the weights in the loop?
    # No, beta is large, so only a few terms have non-negligible weight.
    # We can approximate: Only compute gradient for the "active" constraints (those with small a).
    # Threshold: a < a_max + log(k_total) / beta ?
    
    # Let's just use the full weighted sum but compute it efficiently.
    # We'll create an array of weights for the flattened pair_dists.
    
    # Re-calculate weights for the full array (with zeros handled)
    # If dist is 0, the gradient is undefined. We assume points don't exactly coincide.
    
    full_pair_dists = pair_dists / 2.0
    full_all_constraints = np.concatenate([full_pair_dists.flatten(), boundary_dists])
    
    # Recalculate weights for the full set to avoid mapping issues
    a_full = full_all_constraints
    a_max_full = np.max(a_full)
    exp_terms_full = np.exp(-beta * (a_full - a_max_full))
    sum_exp_full = np.sum(exp_terms_full)
    weights_full = exp_terms_full / sum_exp_full
    
    # Reshape weights for pairs
    weights_pairs = weights_full[:n * k_nn].reshape(n, k_nn)
    weights_boundary = weights_full[n * k_nn:]
    
    # Compute gradient for pairs
    for i in range(n):
        for k in range(k_nn):
            j = pair_idxs[i, k]
            d = points[i] - points[j]
            norm_d = np.linalg.norm(d)
            if norm_d > 1e-12:
                w = weights_pairs[i, k]
                grad[i] += w * d / (2.0 * norm_d)
                grad[j] -= w * d / (2.0 * norm_d)
                
    # Compute gradient for boundary
    for i in range(n):
        r = radii[i]
        if r > 1e-12:
            w = weights_boundary[i]
            grad[i] -= w * points[i] / r
            
    obj = -t_soft
    return obj, (-grad).flatten()


def _generate_hex_lattice(n):
    """Generate hexagonal lattice points inside unit disk."""
    # Optimal hex spacing for n points in unit disk
    # Area per hex cell = s^2 * sqrt(3)/2
    # n * s^2 * sqrt(3)/2 <= pi
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0))) * 0.97
    
    k = int(2.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        y = j * s * np.sqrt(3.0) / 2.0
        if abs(y) > 1.0:
            continue
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            if np.hypot(x, y) <= 1.0 - s / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        # Fallback: add more points via spiral
        theta = np.linspace(0, 2 * np.pi * 20, n - len(points))
        r = np.linspace(0.1, 0.95, n - len(points))
        extra = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        points = np.vstack([points, extra])
    else:
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')
        points = points[idx[:n]]
    
    return points[:n]


def _generate_ring_config(n):
    """Generate concentric ring configuration."""
    # Center point
    points = [[0.0, 0.0]]
    remaining = n - 1
    
    ring_radius = 0.15
    ring_count = 0
    while remaining > 0:
        # Number of points on this ring
        circumference = 2 * np.pi * ring_radius
        # Spacing ~ 2*r where r is target radius
        # Target r ~ 0.053
        target_spacing = 0.106
        count = max(1, int(circumference / target_spacing))
        count = min(count, remaining)
        
        if count > 0:
            theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
            ring_pts = np.column_stack([ring_radius * np.cos(theta), 
                                         ring_radius * np.sin(theta)])
            points.append(ring_pts)
            remaining -= count
        
        ring_radius += 0.08
        ring_count += 1
        if ring_radius > 0.9:
            break
    
    points = np.vstack(points)
    if len(points) > n:
        points = points[:n]
    elif len(points) < n:
        # Add more
        theta = np.linspace(0, 2 * np.pi, n - len(points))
        r = 0.95 * np.ones(n - len(points))
        extra = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        points = np.vstack([points, extra])
    
    return points[:n]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initial configurations
    candidates = []
    
    # Hexagonal lattice
    try:
        hex_pts = _generate_hex_lattice(N)
        if len(hex_pts) == N and np.all(np.isfinite(hex_pts)):
            candidates.append(hex_pts)
    except Exception:
        pass
    
    # Ring configuration
    try:
        ring_pts = _generate_ring_config(N)
        if len(ring_pts) == N and np.all(np.isfinite(ring_pts)):
            candidates.append(ring_pts)
    except Exception:
        pass
    
    # Random initialization
    try:
        np.random.seed(42)
        r = np.sqrt(np.random.rand(N)) * 0.95
        theta = np.random.rand(N) * 2 * np.pi
        rand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        candidates.append(rand_pts)
    except Exception:
        pass
    
    if not candidates:
        # Last resort
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r = np.linspace(0, 0.95, N)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Evaluate all candidates
    best_score = -np.inf
    best_points = None
    
    for cand in candidates:
        try:
            score = _compute_score(cand)
            if score > best_score:
                best_score = score
                best_points = cand.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = candidates[0]
        best_score = _compute_score(best_points)
    
    # Local optimization using L-BFGS-B with soft-min objective
    # Use adaptive beta: start with lower beta (smoother) and increase
    
    beta_schedule = [20.0, 50.0, 100.0, 200.0, 500.0]
    k_nn = 5
    
    for beta in beta_schedule:
        if time.time() > deadline - 10.0:
            break
        
        x0 = best_points.flatten()
        
        try:
            # Get initial objective and gradient
            obj, grad = _soft_min_obj_and_grad(x0, N, beta, k_nn)
            
            if not np.isfinite(obj) or not np.all(np.isfinite(grad)):
                continue
            
            result = minimize(
                _soft_min_obj_and_grad,
                x0,
                args=(N, beta, k_nn),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            if result.x is not None:
                new_points = result.x.reshape(N, 2)
                if np.all(np.isfinite(new_points)):
                    new_score = _compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
        except Exception:
            pass
    
    # Try a few random restarts if time permits
    restart_count = 0
    while time.time() < deadline - 15.0 and restart_count < 5:
        # Perturb best points
        noise_scale = 0.005
        x0 = best_points.flatten() + np.random.normal(0, noise_scale, size=N * 2)
        
        beta = 100.0
        try:
            result = minimize(
                _soft_min_obj_and_grad,
                x0,
                args=(N, beta, k_nn),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            if result.x is not None:
                new_points = result.x.reshape(N, 2)
                if np.all(np.isfinite(new_points)):
                    new_score = _compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
        except Exception:
            pass
        
        restart_count += 1
    
    # Final validation
    radii = np.linalg.norm(best_points, axis=1)
    if np.any(radii > 1.0):
        mask = radii > 1.0
        best_points[mask] *= (1.0 - 1e-9) / radii[mask, np.newaxis]
    
    # Ensure non-finite values are handled
    if not np.all(np.isfinite(best_points)):
        # Fallback
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r = np.linspace(0, 0.95, N)
        best_points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    return best_points
# EVOLVE-BLOCK-END
