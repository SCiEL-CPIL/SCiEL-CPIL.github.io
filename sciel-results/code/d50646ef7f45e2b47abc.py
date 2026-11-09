"""Combined program: hex lattice init + soft-min L-BFGS-B optimization with ring-based polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the packing score: min pair distance / 2, min distance to boundary."""
    points = np.asarray(points)
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    n = len(points)
    # Pairwise distances using KDTree for efficiency
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = np.min(dists[:, 1])
    
    # Distance to boundary
    radii = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair / 2.0, min_boundary)


def _initial_hex_lattice(n):
    """Generate a hexagonal lattice of points inside the unit disk, return first n."""
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0))) * 0.95
    
    max_r = 1.0
    k = int(2.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        y = j * s * np.sqrt(3.0) / 2.0
        if abs(y) > 1.0:
            continue
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            r = np.hypot(x, y)
            if r <= 1.0 - s / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        theta = np.linspace(0, 2 * np.pi * 10, n)
        r = np.linspace(0, 1, n)
        points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    else:
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')
        points = points[idx[:n]]
    return points


def _softmin_obj_and_grad(x, n, beta=100.0):
    """
    Smooth surrogate objective using log-sum-exp for min pair distance and min boundary.
    Returns (objective_value, gradient).
    """
    points = x.reshape(n, 2)
    radii = np.linalg.norm(points, axis=1)
    
    # Boundary distances
    boundary_dists = 1.0 - radii
    
    # Pair distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    pair_vals = dist / 2.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_vals_flat = pair_vals[i_idx, j_idx]
    
    boundary_vals = boundary_dists
    
    all_vals = np.concatenate([pair_vals_flat, boundary_vals])
    
    # Soft-min computation
    m = np.min(all_vals)
    shifted = all_vals - m
    
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        sum_exp = 1e-300
    
    log_sum_exp = np.log(sum_exp)
    
    soft_min_val = m - log_sum_exp / beta
    
    obj = -soft_min_val
    
    weights = exp_terms / sum_exp
    
    grad_points = np.zeros_like(points)
    
    # Gradient from boundary
    b_weights = weights[len(pair_vals_flat):]
    safe_radii = np.where(radii > 1e-12, radii, 1.0)
    grad_boundary = (b_weights / safe_radii)[:, np.newaxis] * points
    grad_points += grad_boundary
    
    # Gradient from pairs
    p_weights = weights[:len(pair_vals_flat)]
    diff_vec = points[i_idx] - points[j_idx]
    dist_vec = np.linalg.norm(diff_vec, axis=1)
    safe_dist = np.where(dist_vec > 1e-12, dist_vec, 1.0)
    
    grad_pair_vec = (diff_vec / (2.0 * safe_dist))[:, np.newaxis] * p_weights[:, np.newaxis]
    
    np.add.at(grad_points, i_idx, -grad_pair_vec)
    np.add.at(grad_points, j_idx, grad_pair_vec)
    
    return obj, grad_points.flatten()


def _objective_soft(x, n, beta=100.0):
    """Objective function for optimization: negative soft-min score."""
    obj, _ = _softmin_obj_and_grad(x, n, beta)
    return obj


def _objective_soft_grad(x, n, beta=100.0):
    """Gradient of the objective."""
    _, grad = _softmin_obj_and_grad(x, n, beta)
    return grad


def _ring_polish(points, time_limit=10.0):
    """
    Polish the packing using a ring-based parameterization.
    Group points into concentric rings and optimize radii and angular offsets.
    This exploits the radial symmetry to potentially find better local optima.
    """
    start = time.time()
    points = points.copy()
    n = len(points)
    
    # Compute current radii and angles
    radii = np.linalg.norm(points, axis=1)
    angles = np.arctan2(points[:, 1], points[:, 0])
    
    # Sort points by radius
    order = np.argsort(radii, kind='stable')
    radii_sorted = radii[order]
    angles_sorted = angles[order]
    points_sorted = points[order]
    
    # Determine ring boundaries: points with similar radii are in the same ring
    # Use a tolerance based on the expected ring spacing
    # For ~300 points in unit circle, rings are roughly at r ~ k * 0.06
    # Let's cluster by radius with a tolerance
    
    # Simple approach: define rings by quantiles or by spacing
    # Better: use the fact that in hex packing, rings have specific counts
    # For now, use a heuristic: group points whose radii are within 0.02 of each other
    
    ring_tolerance = 0.015
    rings = []
    current_ring = [0]
    
    for i in range(1, n):
        if radii_sorted[i] - radii_sorted[i-1] > ring_tolerance:
            rings.append(current_ring)
            current_ring = [i]
        else:
            current_ring.append(i)
    if current_ring:
        rings.append(current_ring)
    
    # For each ring, we optimize:
    # - The common radius of the ring (r_k)
    # - The angular offset of the ring (phi_k)
    # - Optionally, individual angular deviations if the ring is not perfectly uniform
    
    # To keep it simple and fast, we'll optimize ring radius and angular offset
    # and assume points within a ring are equally spaced with a common offset
    
    # Build a parameter vector: for each ring, (r_k, phi_k)
    # But we also need to handle the center point separately
    
    # Let's separate the center point (if any) from the rings
    center_mask = radii_sorted < 0.01
    if np.any(center_mask):
        center_idx = np.where(center_mask)[0][0]
        ring_indices = [list(r) for r in rings if not any(idx == center_idx for idx in r)]
    else:
        ring_indices = [list(r) for r in rings]
        center_idx = None
    
    n_rings = len(ring_indices)
    
    # Parameters: for each ring, [r_k, phi_k]
    # Plus center position if exists: [cx, cy]
    n_params = 2 * n_rings + (2 if center_idx is not None else 0)
    
    # Initialize parameters from current configuration
    params = np.zeros(n_params)
    ring_params_offset = 0
    for k, ring in enumerate(ring_indices):
        r_k = np.mean(radii_sorted[ring])
        # Compute mean angle
        if len(ring) > 0:
            # Use circular mean
            angles_ring = angles_sorted[ring]
            sin_mean = np.mean(np.sin(angles_ring))
            cos_mean = np.mean(np.cos(angles_ring))
            phi_k = np.arctan2(sin_mean, cos_mean)
        else:
            phi_k = 0.0
        params[ring_params_offset] = r_k
        params[ring_params_offset + 1] = phi_k
        ring_params_offset += 2
    
    if center_idx is not None:
        params[-2] = points_sorted[center_idx, 0]
        params[-1] = points_sorted[center_idx, 1]
    
    def _ring_objective(p):
        """Objective for ring-based optimization."""
        elapsed = time.time() - start
        if elapsed > time_limit:
            return 0.0  # Will be handled by caller
        
        # Reconstruct points from parameters
        pts = np.zeros((n, 2))
        
        # Fill rings
        ring_params_offset = 0
        for k, ring in enumerate(ring_indices):
            r_k = p[ring_params_offset]
            phi_k = p[ring_params_offset + 1]
            ring_params_offset += 2
            
            # Ensure r_k is in valid range
            r_k = np.clip(r_k, 0.0, 0.99)
            
            m = len(ring)
            # Points in this ring are equally spaced with offset phi_k
            base_angles = phi_k + 2 * np.pi * np.arange(m) / m
            for idx_i, global_idx in enumerate(ring):
                theta = base_angles[idx_i]
                pts[global_idx, 0] = r_k * np.cos(theta)
                pts[global_idx, 1] = r_k * np.sin(theta)
        
        # Fill center
        if center_idx is not None:
            pts[center_idx, 0] = p[-2]
            pts[center_idx, 1] = p[-1]
        
        # Map back to original order
        pts_original = pts[np.argsort(order)]
        
        # Compute score
        score = _compute_score(pts_original)
        return -score
    
    def _ring_gradient(p):
        """Numerical gradient for ring-based optimization."""
        elapsed = time.time() - start
        if elapsed > time_limit:
            return np.zeros(n_params)
        
        grad = np.zeros(n_params)
        eps = 1e-7
        f0 = _ring_objective(p)
        
        for i in range(n_params):
            p_plus = p.copy()
            p_plus[i] += eps
            f_plus = _ring_objective(p_plus)
            grad[i] = (f_plus - f0) / eps
            
            if time.time() - start > time_limit:
                break
        
        return grad
    
    # Optimize using L-BFGS-B
    try:
        result = minimize(
            _ring_objective,
            params,
            method='L-BFGS-B',
            jac=_ring_gradient,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 500}
        )
        
        # Reconstruct final points
        p_opt = result.x
        pts = np.zeros((n, 2))
        
        ring_params_offset = 0
        for k, ring in enumerate(ring_indices):
            r_k = p_opt[ring_params_offset]
            phi_k = p_opt[ring_params_offset + 1]
            ring_params_offset += 2
            
            r_k = np.clip(r_k, 0.0, 0.99)
            
            m = len(ring)
            base_angles = phi_k + 2 * np.pi * np.arange(m) / m
            for idx_i, global_idx in enumerate(ring):
                theta = base_angles[idx_i]
                pts[global_idx, 0] = r_k * np.cos(theta)
                pts[global_idx, 1] = r_k * np.sin(theta)
        
        if center_idx is not None:
            pts[center_idx, 0] = p_opt[-2]
            pts[center_idx, 1] = p_opt[-1]
        
        pts_original = pts[np.argsort(order)]
        
        # Validate and return
        score = _compute_score(pts_original)
        if score > 0:
            return pts_original
        else:
            return points
    
    except Exception:
        return points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    np.random.seed(42)
    
    # Step 1: Generate initial hexagonal lattice
    points = _initial_hex_lattice(N)
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Step 2: Use L-BFGS-B to optimize with soft-min objective
    x0 = points.flatten()
    
    # Try different beta values
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 30.0:
            break
            
        try:
            result = minimize(
                _objective_soft,
                x0,
                args=(N, beta),
                method='L-BFGS-B',
                jac=_objective_soft_grad,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_points = result.x.reshape(N, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                x0 = new_points.flatten()
        except Exception:
            pass
    
    # Step 3: Ring-based polish
    if time.time() < deadline - 15.0:
        ring_time = min(15.0, deadline - time.time() - 5.0)
        polished_points = _ring_polish(best_points, time_limit=ring_time)
        polished_score = _compute_score(polished_points)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points.copy()
    
    # Step 4: Final local optimization pass with higher beta if time remains
    if time.time() < deadline - 5.0:
        x0 = best_points.flatten()
        beta = 300.0
        try:
            result = minimize(
                _objective_soft,
                x0,
                args=(N, beta),
                method='L-BFGS-B',
                jac=_objective_soft_grad,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            new_points = result.x.reshape(N, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        except Exception:
            pass
    
    # Final validation: ensure all points are inside and non-overlapping
    radii = np.linalg.norm(best_points, axis=1)
    if np.any(radii > 1.0):
        mask = radii > 1.0
        best_points[mask] *= (1.0 - 1e-9) / radii[mask, np.newaxis]
    
    return best_points

# EVOLVE-BLOCK-END
