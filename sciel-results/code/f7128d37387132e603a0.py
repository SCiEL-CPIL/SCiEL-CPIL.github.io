"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    # Using broadcasting is memory intensive for N=300 (300x300 is fine, but let's be careful)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init():
    """Generate initial points using a hexagonal lattice."""
    # Estimate spacing
    # Area of unit circle is pi. N circles of area pi*r^2.
    # r ~ sqrt(pi / (N * 1.1)) roughly
    r_est = np.sqrt(np.pi / (N * 1.2))
    spacing = 2 * r_est
    
    while True:
        k = int(1.0 / spacing) + 2
        # Hex lattice points
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Filter points inside the circle, keeping a margin
        margin = spacing / 2.0
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            # Sort by radius to pick the most central ones? 
            # Actually, for packing, we want to fill the space. 
            # Picking the first N might be biased. 
            # A better heuristic: sort by distance to center and take N.
            indices = np.argsort(radius, kind="stable")[:N]
            points = points[indices]
            break
        spacing *= 0.95
        
    return points

def _radial_ring_init():
    """
    Generate initial points using concentric rings.
    This often provides a better starting point for circular containers.
    """
    # Determine number of rings
    # For N=300, typical ring counts are around 15-20
    # Let's try to estimate optimal ring count
    # Approximation: rings are spaced by 2r.
    # Total radius 1. Number of rings ~ 1/(2r).
    # r ~ 0.05, so ~ 10 rings? 
    # Actually, for N=300, best packing has ~18-20 rings.
    
    best_points = None
    best_score = 0.0
    
    # Try a range of ring counts
    for num_rings in range(10, 25):
        # Radii of rings
        # The outer ring is near 1-r. Inner rings are spaced.
        # Simple heuristic: uniform spacing of ring radii
        r_max = 1.0
        # We need to fit N points.
        # Points on ring k: n_k
        # Sum n_k = N
        # Optimal n_k is proportional to radius r_k
        # n_k ~ r_k * C
        
        # Let's just use a simple distribution first and refine
        radii = np.linspace(0.05, 0.95, num_rings)
        
        # Distribute N points among rings proportional to radius
        weights = radii
        weights /= np.sum(weights)
        counts = np.round(weights * N).astype(int)
        
        # Adjust counts to sum to N
        diff = N - np.sum(counts)
        # Add/remove from rings with largest/smallest fractional parts or just largest rings
        fracs = weights * N - counts
        indices = np.argsort(-fracs)[:diff] if diff > 0 else np.argsort(fracs)[:abs(diff)]
        for idx in indices:
            counts[idx] += 1 if diff > 0 else -1
            
        points = []
        for k, r in enumerate(radii):
            n_k = counts[k]
            if n_k == 0:
                continue
            # Angular offset to stagger rings
            offset = (k % 2) * (np.pi / n_k)
            angles = np.linspace(0, 2 * np.pi, n_k, endpoint=False) + offset
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            points.append(np.column_stack((x, y)))
            
        if len(points) == 0:
            continue
            
        points_array = np.vstack(points)
        
        # Basic check: are they inside?
        norms = np.linalg.norm(points_array, axis=1)
        if np.any(norms > 1.0):
            continue
            
        score = _compute_score(points_array)
        if score > best_score:
            best_score = score
            best_points = points_array
            
    if best_points is None:
        return _hex_lattice_init()
        
    return best_points

def _soft_min_obj_and_grad(x_flat, beta):
    """
    Objective: -soft_min(pairwise_dists/2, boundary_dists)
    Gradient: computed analytically.
    """
    x = x_flat.reshape(-1, 2)
    n = len(x)
    
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # We only need i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[mask] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(x, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min
    y_min = np.min(all_dists)
    # Numerical stability
    exps = np.exp(-beta * (all_dists - y_min))
    sum_exps = np.sum(exps)
    soft_min_val = y_min - (1.0 / beta) * np.log(sum_exps)
    
    # Objective is -soft_min_val
    obj = -soft_min_val
    
    # Gradient
    probs = exps / sum_exps
    
    grad_flat = np.zeros_like(x_flat)
    
    # Gradient of pairwise distances
    safe_dists = np.where(dists < 1e-8, 1e-8, dists)
    
    # We need to map pair index back to (i, j) to build the gradient matrix.
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    A = np.zeros((n, n))
    p_pairs = probs[:len(pair_dists)]
    d_pairs = dists[i_idx, j_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1e-8, d_pairs)
    weights = p_pairs / (2.0 * safe_d_pairs)
    
    A[i_idx, j_idx] = weights
    A[j_idx, i_idx] = weights
    
    row_sums = A.sum(axis=1)
    Ax = A @ x
    
    grad_pairwise = x * row_sums[:, np.newaxis] - Ax
    
    # Gradient of boundary distances
    p_bound = probs[len(pair_dists):]
    safe_norms = np.where(norms < 1e-8, 1e-8, norms)
    grad_bound = -p_bound[:, np.newaxis] * x / safe_norms[:, np.newaxis]
    
    total_grad = grad_pairwise + grad_bound
    
    # Objective is -soft_min, so gradient is -total_grad
    return obj, -total_grad.flatten()

def _polish_active_constraints(points, iterations=50, lr=0.001):
    """
    Polish the solution by focusing on the active constraints (closest pairs and boundary).
    This is a simple gradient step on the min distance, using only the top K constraints.
    """
    pts = points.copy()
    n = len(pts)
    
    for _ in range(iterations):
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pairs
        flat_dists = dists.flatten()
        # Get indices of smallest distances
        # We want the smallest distances, so argsort
        # But we need to ignore diagonal
        # Let's just find the min pair distance and boundary distance
        min_pair_dist = np.min(flat_dists)
        min_pair_idx = np.unravel_index(np.argmin(flat_dists), dists.shape)
        
        # Check boundary
        norms = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - norms
        min_bound_dist = np.min(bound_dists)
        min_bound_idx = np.argmin(bound_dists)
        
        # Determine which constraint is active
        # We want to increase the minimum of these.
        # Gradient ascent on the min value.
        
        # Let's use a soft-min on just the top 10 closest pairs and top 5 boundary points
        # This is faster than full N^2
        top_k = 10
        # Get top K smallest pair distances
        # This is O(N^2 log K) or just sort. N=300, N^2=90000. Sort is fine.
        flat_dists_sorted = np.sort(flat_dists)
        # Actually, we need the indices.
        # Let's just use the full soft-min gradient but with a very high beta?
        # Or just use the full gradient from _soft_min_obj_and_grad with high beta.
        
        # Let's stick to the full gradient for simplicity and correctness in this polish step,
        # but limit iterations.
        x_flat = pts.flatten()
        _, grad = _soft_min_obj_and_grad(x_flat, beta=1000.0)
        
        # Gradient step
        pts += lr * grad.reshape(-1, 2)
        
        # Project back to feasible region? 
        # The objective handles it, but if we go outside, score drops.
        # Let's clip to unit circle roughly
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask, np.newaxis]
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Initialization
    # Try both and pick the better one
    p_hex = _hex_lattice_init()
    p_ring = _radial_ring_init()
    
    s_hex = _compute_score(p_hex)
    s_ring = _compute_score(p_ring)
    
    if s_ring > s_hex:
        points = p_ring
    else:
        points = p_hex
        
    # 2. L-BFGS Optimization with Soft Min
    # Start with low beta and increase
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    x_current = points.flatten()
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
            
        def obj_grad(x):
            return _soft_min_obj_and_grad(x, beta)
        
        try:
            result = minimize(
                lambda x: obj_grad(x)[0],
                x_current,
                jac=lambda x: obj_grad(x)[1],
                method='L-BFGS-B',
                bounds=[(-1.1, 1.1)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success:
                x_current = result.x
            else:
                # Even if not success, if it improved, take it
                if result.fun < obj_grad(x_current)[0]:
                    x_current = result.x
        except Exception:
            pass
            
    points_final = x_current.reshape(-1, 2)
    
    best_score = _compute_score(points_final)
    best_points = points_final.copy()
    
    # 3. Final Polish with Active Constraints / High Beta Gradient Descent
    if time.time() < deadline - 5.0:
        # Perform a few steps of gradient descent with high beta
        # This helps escape local minima of the soft-min approximation
        lr = 0.001
        for i in range(100):
            if time.time() > deadline - 2.0:
                break
            x_flat = best_points.flatten()
            _, grad = _soft_min_obj_and_grad(x_flat, beta=5000.0)
            
            # Adaptive learning rate
            # If gradient is large, step smaller
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-6:
                break
                
            new_x = x_flat + lr * grad
            new_points = new_x.reshape(-1, 2)
            
            # Check if it improved
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
            else:
                # Reduce learning rate
                lr *= 0.5
                
    return best_points
# EVOLVE-BLOCK-END
