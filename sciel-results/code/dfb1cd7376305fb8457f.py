import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

def _hilbert_curve_3d(n, depth=4):
    """Generate points along a 3D Hilbert curve.
    
    Args:
        n: Number of points to generate
        depth: Depth of the Hilbert curve (2^depth points total)
    
    Returns:
        Array of shape (n, 3) with coordinates in [0, 1]^3
    """
    # Generate full Hilbert curve
    total_points = 2 ** depth
    # Use a simple recursive Hilbert curve generator
    def hilbert_3d(order, x, y, z):
        """Recursive Hilbert curve generator for 3D."""
        if order == 0:
            return [(x, y, z)]
        
        points = []
        # 8 sub-cubes in 3D Hilbert curve
        # The order follows the standard 3D Hilbert curve pattern
        sub_order = order - 1
        sub_size = 2 ** sub_order
        
        # Define the 8 sub-cubes in Hilbert order
        # This is a simplified version; true Hilbert curve in 3D is complex
        # For simplicity, use a space-filling curve that visits all sub-cubes
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Visit sub-cube (i, j, k)
                    sub_points = hilbert_3d(sub_order, 
                                            x + i * sub_size, 
                                            y + j * sub_size, 
                                            z + k * sub_size)
                    points.extend(sub_points)
        
        return points
    
    full_curve = hilbert_3d(depth, 0, 0, 0)
    total = len(full_curve)
    
    # Normalize to [0, 1]^3
    max_val = 2 ** depth - 1
    normalized = np.array(full_curve) / max_val
    
    # Sample n points uniformly along the curve
    indices = np.linspace(0, total - 1, n, dtype=int)
    sampled = normalized[indices]
    
    return sampled

def _soft_min_score(centers, alpha=20.0):
    """Smooth approximation of the minimum radius using soft-min."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Combine all distances
    all_dists = np.concatenate([dist[np.triu_indices_from(dist, k=1)], wall_dists.ravel()])
    
    # Soft minimum: -log(sum(exp(-alpha * d))) / alpha
    # This approximates min(d) as alpha -> inf
    log_sum = np.log(np.sum(np.exp(-alpha * all_dists)))
    soft_min = -log_sum / alpha
    
    return soft_min

def _objective_flat(x, alpha=20.0):
    """Objective function for L-BFGS-B: minimize negative soft-min radius."""
    centers = x.reshape(N, 3)
    return -_soft_min_score(centers, alpha)

def _objective_grad_flat(x, alpha=20.0):
    """Objective function and gradient for L-BFGS-B."""
    centers = x.reshape(N, 3)
    
    # Compute distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    wall_dists = 0.5 - np.abs(centers)
    
    all_dists = np.concatenate([dist[np.triu_indices_from(dist, k=1)], wall_dists.ravel()])
    
    # Soft min value
    exp_vals = np.exp(-alpha * all_dists)
    log_sum = np.log(np.sum(exp_vals))
    soft_min = -log_sum / alpha
    
    # Gradient of soft_min w.r.t. all_dists
    # d(soft_min)/d(all_dists[i]) = (exp(-alpha * all_dists[i])) / (sum(exp(-alpha * all_dists)))
    grad_all_dists = exp_vals / np.sum(exp_vals)
    
    # Gradient w.r.t. centers
    grad_centers = np.zeros_like(centers)
    
    # Pairwise distances gradient
    # For each pair (i, j), d/d(centers[i]) |c_i - c_j| = (c_i - c_j) / |c_i - c_j|
    mask = dist > 1e-8
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=mask)
    
    # Gradient contribution from pairwise distances
    # grad_centers[i] += sum_j (grad_all_dists[pair_idx] * (c_i - c_j) / dist[i,j])
    for i in range(N):
        for j in range(i + 1, N):
            if mask[i, j]:
                # Find the index of this pair in all_dists
                # Upper triangle indices: i < j
                # The index in the flattened upper triangle
                pair_idx = i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
                weight = grad_all_dists[pair_idx]
                diff_ij = centers[i] - centers[j]
                grad_centers[i] += weight * diff_ij * inv_dist[i, j]
                grad_centers[j] -= weight * diff_ij * inv_dist[i, j]
    
    # Wall distances gradient
    # d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
    for i in range(N):
        for k in range(3):
            wall_idx = i * 3 + k
            weight = grad_all_dists[wall_idx]
            grad_centers[i, k] -= weight * np.sign(centers[i, k])
    
    # Gradient of objective = -gradient of soft_min
    grad = -grad_centers.ravel()
    
    return -soft_min, grad

def _repulsive_dynamics(centers, steps=100, lr=0.005):
    """Force-directed relaxation with repulsive forces."""
    centers = centers.copy()
    
    for _ in range(steps):
        # Pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        mask = dist > 1e-8
        inv_dist_cubed = np.zeros_like(dist)
        np.divide(1.0, dist**3, out=inv_dist_cubed, where=mask)
        
        forces = diff * inv_dist_cubed[:, :, np.newaxis]
        total_force = np.sum(forces, axis=1)
        
        # Wall repulsion
        abs_centers = np.abs(centers)
        wall_dist = 0.5 - abs_centers
        threshold = 0.15
        wall_mask = wall_dist < threshold
        
        wall_force_mag = np.zeros_like(wall_dist)
        np.divide(1.0, (wall_dist**2 + 1e-6), out=wall_force_mag, where=wall_mask)
        
        signs = np.sign(centers)
        wall_force = -signs * wall_force_mag
        
        total_force += wall_force
        
        centers += lr * total_force
        centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    
    return centers

def _local_search(centers, max_iter=500, tol=1e-10):
    """Local search using gradient-based optimization."""
    x0 = centers.ravel()
    bounds = [(-0.5 + 1e-6, 0.5 - 1e-6)] * (N * 3)
    
    try:
        result = minimize(
            _objective_flat,
            x0,
            jac=_objective_grad_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
        )
        if result.success:
            return result.x.reshape(N, 3)
    except Exception:
        pass
    
    return centers

def _compute_score(centers):
    """Compute the exact minimum radius for the given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def construct():
    """Construct optimal sphere packing using Hilbert curve init + gradient optimization."""
    start_time = time.time()
    time_limit = 130.0  # Leave 20s margin for safety
    
    best_score = 0.0
    best_centers = None
    
    # Phase 1: Multiple restarts with Hilbert curve initialization
    num_restarts = 8
    
    for i in range(num_restarts):
        # Check time
        if time.time() - start_time > time_limit - 30:
            break
        
        # Generate initial points using Hilbert curve with different depths/offsets
        if i == 0:
            # Standard Hilbert curve
            centers = _hilbert_curve_3d(N, depth=4)
            # Map from [0,1]^3 to [-0.45, 0.45]^3
            centers = centers * 0.9 - 0.45
        elif i == 1:
            # Hilbert curve with slight random perturbation
            centers = _hilbert_curve_3d(N, depth=5)
            centers = centers * 0.9 - 0.45
            rng = np.random.default_rng(42 + i)
            centers += rng.normal(0, 0.02, (N, 3))
            centers = np.clip(centers, -0.45, 0.45)
        elif i == 2:
            # Different Hilbert curve orientation
            centers = _hilbert_curve_3d(N, depth=4)
            centers = centers * 0.9 - 0.45
            # Rotate
            rng = np.random.default_rng(100 + i)
            angle = rng.uniform(0, np.pi / 4)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            centers = centers @ rot.T
            centers = np.clip(centers, -0.45, 0.45)
        elif i == 3:
            # Grid initialization
            grid = np.array(np.meshgrid(
                np.linspace(-0.4, 0.4, 4),
                np.linspace(-0.4, 0.4, 4),
                np.linspace(-0.4, 0.4, 4)
            )).T.reshape(-1, 3)
            rng = np.random.default_rng(200 + i)
            centers = grid[rng.choice(len(grid), N, replace=False)]
        elif i == 4:
            # BCC-like lattice
            centers = []
            for x in [-0.3, 0.0, 0.3]:
                for y in [-0.3, 0.0, 0.3]:
                    for z in [-0.3, 0.0, 0.3]:
                        centers.append([x, y, z])
            # Add face centers
            for x in [-0.15, 0.15]:
                centers.append([x, -0.15, 0.0])
                centers.append([x, 0.15, 0.0])
                centers.append([x, 0.0, -0.15])
                centers.append([x, 0.0, 0.15])
            centers = np.array(centers[:N])
        else:
            # Random initialization
            rng = np.random.default_rng(300 + i)
            centers = rng.uniform(-0.4, 0.4, (N, 3))
        
        # Step 1: Repulsive dynamics for quick improvement
        relaxed = _repulsive_dynamics(centers, steps=50, lr=0.005)
        
        # Step 2: Gradient-based optimization with soft-min objective
        if time.time() - start_time < time_limit - 20:
            optimized = _local_search(relaxed, max_iter=200)
        else:
            optimized = relaxed
        
        # Step 3: Final repulsive dynamics polish
        if time.time() - start_time < time_limit - 10:
            polished = _repulsive_dynamics(optimized, steps=30, lr=0.002)
        else:
            polished = optimized
        
        # Calculate score
        score = _compute_score(polished)
        
        if score > best_score:
            best_score = score
            best_centers = polished.copy()
    
    # Phase 2: Intensive local optimization on best solution
    if best_centers is not None and time.time() - start_time < time_limit - 5:
        # Multiple local search passes with different alpha values
        for alpha in [15.0, 20.0, 30.0]:
            if time.time() - start_time > time_limit - 5:
                break
            
            x0 = best_centers.ravel()
            bounds = [(-0.5 + 1e-6, 0.5 - 1e-6)] * (N * 3)
            
            try:
                result = minimize(
                    _objective_flat,
                    x0,
                    jac=_objective_grad_flat,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'alpha': alpha}
                )
                if result.success:
                    candidate = result.x.reshape(N, 3)
                    cand_score = _compute_score(candidate)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_centers = candidate.copy()
            except Exception:
                pass
        
        # Final repulsive dynamics
        if time.time() - start_time < time_limit - 2:
            final = _repulsive_dynamics(best_centers, steps=50, lr=0.001)
            final_score = _compute_score(final)
            if final_score > best_score:
                best_centers = final.copy()
                best_score = final_score
    
    if best_centers is None:
        # Fallback: Simple grid
        grid = np.array(np.meshgrid(
            np.linspace(-0.4, 0.4, 4),
            np.linspace(-0.4, 0.4, 4),
            np.linspace(-0.4, 0.4, 4)
        )).T.reshape(-1, 3)
        return grid[:50]
    
    return best_centers
