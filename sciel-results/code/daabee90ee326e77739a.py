"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 120.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return max(0.0, DEADLINE - _elapsed() - 5.0)  # 5s safety margin


def _score(points):
    """Compute the packing score: min of pairwise distances/2 and boundary distances."""
    n = len(points)
    if n == 0:
        return 0.0
    # Boundary constraint
    radii = np.linalg.norm(points, axis=1)
    boundary_min = 1.0 - radii
    if np.any(boundary_min < -1e-9):
        return -np.inf
    # Pairwise distances
    # For efficiency, compute in chunks if needed, but N=300 is small
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pairwise_min = np.min(dists) / 2.0
    return min(boundary_min.min(), pairwise_min)


def _hex_lattice_init():
    """Generate a hexagonal lattice configuration inside the unit disk."""
    # Estimate spacing
    # Area of disk = pi. Area per point ~ pi/300. Spacing ~ sqrt(pi/300 / (sqrt(3)/2))
    # But let's just try a few spacings
    for spacing in [0.11, 0.10, 0.09, 0.08, 0.07]:
        points = []
        # Generate hex grid points
        # x = i * spacing + (j % 2) * spacing / 2
        # y = j * spacing * sqrt(3) / 2
        max_i = int(2.0 / spacing) + 2
        max_j = int(2.0 / spacing) + 2
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = i * spacing + (j % 2) * spacing / 2.0
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x ** 2 + y ** 2 <= (1.0 - spacing / 2.0) ** 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            # Sort by radius to pick central points? Or just take first N?
            # Taking first N might bias towards origin if generated that way.
            # Let's sort by distance to origin and take N closest to ensure compactness?
            # Actually, any N points from the lattice work.
            # To be safe, let's take the N points closest to the origin to ensure they fit well.
            radii = np.linalg.norm(points, axis=1)
            idx = np.argsort(radii)[:N]
            return points[idx]
    # Fallback: random
    return np.random.uniform(-0.9, 0.9, (N, 2))


def _random_init():
    """Generate random points inside the unit disk."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def _optimize_from(points):
    """Run L-BFGS-B optimization starting from given points."""
    if _remaining() < 1.0:
        return points, _score(points)
    
    def objective_and_grad(x):
        if _remaining() < 0.1:
            return 0.0, np.zeros_like(x)
        
        pts = x.reshape(-1, 2)
        
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        radii = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - radii
        
        # We want to maximize min(dists/2, boundary)
        # Equivalent to minimizing -min(...)
        # Or maximizing the minimum.
        # Let's define a smooth approximation or use the exact min.
        # For gradient descent, we need differentiable objective.
        # Min is not differentiable.
        # Common trick: minimize sum of exp(-k * dist) or similar.
        # Or use a barrier method.
        # Let's try minimizing: -min_val + penalty for violations.
        # Actually, let's use a simpler approach:
        # Minimize: sum(1/dists^2) + sum(1/boundary^2) ? No, that minimizes distances.
        # We want to MAXIMIZE distances.
        # So Minimize: sum(1/dists) is bad.
        # Let's use: -log(min_val) is bad.
        
        # Let's use a soft-min approximation:
        # soft_min = -1/k * log(sum(exp(-k * values)))
        # values = [dists/2, boundary]
        # This is smooth.
        
        # However, computing all pairwise dists is O(N^2).
        # N=300, N^2=90000. This is fast.
        
        # Let's construct the vector of all constraints values
        # Pairwise: dists/2
        # Boundary: boundary
        
        # We want to maximize the minimum of these.
        # Let's use the log-sum-exp trick for the minimum.
        # min(x) = -1/k * log(sum(exp(-k*x)))
        
        k = 20.0  # Sharpness of min approximation
        
        # Flatten dists (excluding diagonal)
        # Create mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pairwise_vals = dists[mask] / 2.0
        
        # Boundary vals
        boundary_vals = boundary
        
        # Combine
        all_vals = np.concatenate([pairwise_vals, boundary_vals])
        
        # Numerical stability
        min_val = np.min(all_vals)
        if min_val < 1e-6:
            # Avoid overflow in exp
            # If min is very small, objective should be large (bad)
            # Return a large penalty
            return 1000.0, np.zeros_like(x)
            
        # Compute soft min
        # To avoid underflow/overflow, shift by min_val
        shifted = all_vals - min_val
        exp_vals = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_vals))
        soft_min = min_val - (1.0 / k) * log_sum
        
        # Objective is -soft_min
        obj = -soft_min
        
        # Gradient
        # d(soft_min)/dx = (1/k) * sum( (k * exp(-k*shifted_i) / sum(exp(-k*shifted))) * d(shifted_i)/dx )
        # Note: shifted_i = val_i - min_val. d(min_val)/dx is tricky.
        # Actually, the gradient of the soft min is the weighted average of the gradients of the individual values,
        # where weights are proportional to exp(-k * (val_i - min_val)).
        # Wait, the derivative of -1/k log(sum(exp(-k v_i))) w.r.t v_j is:
        # -1/k * (1/sum(exp)) * (-k exp(-k v_j)) = exp(-k v_j) / sum(exp(-k v_i))
        # So grad soft_min = sum_j (w_j * grad v_j) where w_j = exp(-k (v_j - min_val)) / sum(exp(-k (v_i - min_val)))
        
        weights = exp_vals / np.sum(exp_vals)
        
        # Now we need to backpropagate weights to x.
        # v_i are either pairwise distances/2 or boundary distances.
        
        grad_x = np.zeros_like(x)
        
        # Boundary part
        # v_b_i = 1 - ||p_i||
        # grad v_b_i / grad p_i = -p_i / ||p_i||
        # Weight for boundary i is weights[N_pairs + i]
        # N_pairs = len(pairwise_vals)
        n_pairs = len(pairwise_vals)
        w_boundary = weights[n_pairs:]
        
        # grad_boundary = -w_boundary[:, np.newaxis] * pts / radii[:, np.newaxis]
        # Handle radii == 0
        safe_radii = np.where(radii < 1e-8, 1.0, radii)
        grad_boundary = -w_boundary[:, np.newaxis] * pts / safe_radii[:, np.newaxis]
        grad_x += grad_boundary
        
        # Pairwise part
        # v_p_k = ||p_i - p_j|| / 2
        # grad v_p_k / grad p_i = (p_i - p_j) / (2 * ||p_i - p_j||)
        # grad v_p_k / grad p_j = (p_j - p_i) / (2 * ||p_i - p_j||)
        
        # We have weights for each pair.
        # We need to map weights back to pairs (i, j).
        # This is expensive if done naively.
        # Let's reconstruct the pairwise distance matrix gradient.
        
        # Instead of iterating over all pairs, we can compute the gradient matrix.
        # G[i, j] = w_{ij} * (p_i - p_j) / (2 * d_{ij})
        # Then grad p_i = sum_j G[i, j]
        
        # But we only have the flattened weights.
        # Let's create a zero matrix and fill it.
        # This is O(N^2) memory and time. N=300, 90000 entries. Fine.
        
        grad_mat = np.zeros((N, N, 2))
        
        # Get indices of upper triangle
        i_idx, j_idx = np.where(mask)
        
        # For each pair (i, j), get weight w
        w_pairs = weights[:n_pairs]
        
        # dists for these pairs
        d_pairs = dists[i_idx, j_idx]
        
        # diff vectors
        diff_vecs = pts[i_idx] - pts[j_idx]
        
        # Gradient contribution to p_i: w * diff_vec / (2 * d)
        # Gradient contribution to p_j: w * -diff_vec / (2 * d)
        
        # Normalize
        safe_d = np.where(d_pairs < 1e-8, 1.0, d_pairs)
        factors = w_pairs / (2.0 * safe_d)
        
        # Accumulate
        np.add.at(grad_x, (i_idx, 0), factors * diff_vecs[:, 0])
        np.add.at(grad_x, (i_idx, 1), factors * diff_vecs[:, 1])
        
        np.add.at(grad_x, (j_idx, 0), -factors * diff_vecs[:, 0])
        np.add.at(grad_x, (j_idx, 1), -factors * diff_vecs[:, 1])
        
        # Total gradient is -grad_soft_min
        return obj, -grad_x.flatten()
    
    # Initial point
    x0 = points.flatten()
    
    # Bounds: keep points inside disk with some margin?
    # Actually, the objective handles boundaries. But to help optimizer,
    # we can bound x, y to [-1, 1].
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        res = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.success or res.x is not None:
            new_points = res.x.reshape(-1, 2)
            return new_points, _score(new_points)
    except Exception:
        pass
    
    return points, _score(points)


def construct():
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Hex lattice
    if _remaining() > 5.0:
        pts = _hex_lattice_init()
        pts, sc = _optimize_from(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    # Strategy 2: Random restarts
    # Use remaining time for random restarts
    while _remaining() > 10.0:
        pts = _random_init()
        pts, sc = _optimize_from(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _hex_lattice_init()
    
    # Final check
    final_score = _score(best_points)
    if final_score < 0:
        # Fallback to hex lattice if optimization failed badly
        hex_pts = _hex_lattice_init()
        if _score(hex_pts) > final_score:
            best_points = hex_pts
    
    return best_points
# EVOLVE-BLOCK-END
