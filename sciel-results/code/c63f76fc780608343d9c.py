"""Improved packing of 300 equal circles in a unit circle using local optimization."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def initial_hex(n):
        """Generate initial hexagonal lattice points."""
        s_est = math.sqrt(2 * math.pi * 0.95**2 / (n * math.sqrt(3)))
        
        best_points = None
        best_score = -1
        
        for trial in range(20):
            spacing = s_est * (1.0 + trial * 0.005)
            points = []
            r_max = int(1.0 / (spacing * math.sqrt(3) / 2)) + 2
            for j in range(-r_max, r_max + 1):
                for i in range(-r_max, r_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * math.sqrt(3) / 2
                    if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                        points.append((x, y))
            
            if len(points) >= n:
                points_arr = np.array(points)
                dists = np.hypot(points_arr[:, 0], points_arr[:, 1])
                idx = np.argsort(dists)[:n]
                candidate = points_arr[idx]
                
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate
            
            if time.time() > deadline - 5:
                break
        
        if best_points is None:
            best_points = np.random.randn(n, 2) * 0.8
            max_r = np.max(np.hypot(best_points[:, 0], best_points[:, 1]))
            if max_r > 0.9:
                best_points *= 0.9 / max_r
        return best_points

    def compute_score(points):
        """Compute the minimum radius for the entire configuration (O(N^2))."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        radii = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - radii)
        
        return min(min_dist / 2.0, min_boundary)
    
    def smooth_min_radius(points, beta=100.0):
        """
        Smooth surrogate for min(d_ij/2, 1-r_i) using log-sum-exp.
        This is differentiable and helps gradient optimization.
        """
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Only consider upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        
        # Boundary distances
        radii = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - radii
        
        # All constraints: d_ij/2 and 1-r_i
        # We want to maximize min of all these
        # Smooth min: -log(sum(exp(-beta * x))) / beta
        all_vals = np.concatenate([pair_dists / 2.0, bound_dists])
        
        # Clip to avoid overflow
        all_vals = np.clip(all_vals, 1e-10, 1.0)
        
        # Smooth minimum
        max_val = np.max(all_vals)
        exp_vals = np.exp(-beta * (all_vals - max_val))
        log_sum = np.log(np.sum(exp_vals))
        smooth_min = max_val - np.log(np.sum(exp_vals)) / beta
        
        return smooth_min

    def get_bottlenecks(points, r_target):
        """Find points that are limiting the score (within r_target of others or boundary)."""
        tree = cKDTree(points)
        bottleneck_idx = []
        
        for i in range(N):
            # Check boundary
            r_i = np.linalg.norm(points[i])
            if 1.0 - r_i <= r_target * 1.01:
                bottleneck_idx.append(i)
                continue
            
            # Check neighbors within 2*r_target
            neighbors = tree.query_ball_point(points[i], 2.0 * r_target)
            for j in neighbors:
                if j != i:
                    d = np.linalg.norm(points[i] - points[j])
                    if d <= 2.0 * r_target * 1.01:
                        bottleneck_idx.append(i)
                        break
        
        return list(set(bottleneck_idx))

    def gradient_optimize(points, max_iter=200):
        """
        Vectorized gradient-based optimization using a smooth surrogate objective.
        Uses full pairwise distances for N=300 which is fast in NumPy.
        """
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.005
        no_improve = 0
        softness = 0.05  # Softness parameter for the surrogate
        
        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            
            # Compute pairwise distances and vectors
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Target distance is 2 * current_score
            target_d = 2.0 * current_score
            
            grads = np.zeros_like(points)
            
            # 1. Boundary forces
            radii = np.linalg.norm(points, axis=1)
            # If close to boundary, push inward
            boundary_mask = radii > (1.0 - current_score)
            if np.any(boundary_mask):
                # Force direction is -points[i] / r_i
                # Magnitude proportional to violation
                violations = radii[boundary_mask] - (1.0 - current_score)
                # Normalize and scale
                # We want to move inward
                safe_radii = np.where(radii[boundary_mask] < 1e-10, 1e-10, radii[boundary_mask])
                inward_vecs = -points[boundary_mask] / safe_radii[:, np.newaxis]
                # Scale by violation magnitude, but keep it smooth
                # Using a simple linear scale
                scale_factor = np.clip(violations / (current_score + 1e-10), 0.0, 1.0)
                grads[boundary_mask] += inward_vecs * scale_factor[:, np.newaxis]
            
            # 2. Pairwise repulsive forces
            # We want to push i away from j if d_ij < target_d
            # Gradient of d_ij w.r.t p_i is (p_i - p_j) / d_ij
            # We want to increase d_ij, so force is + (p_i - p_j) / d_ij * weight
            
            # To avoid division by zero, mask diagonal
            dists_safe = np.where(dists < 1e-10, 1e-10, dists)
            
            # Calculate forces
            # Force on i from j: (p_i - p_j) / d_ij * weight_ij
            # weight_ij = 1 / (1 + exp((d_ij - target_d) / softness))
            # This is 0.5 at d=target_d, ~1 for d<<target_d, ~0 for d>>target_d
            
            # Let's identify the "active" pairs: those with d_ij < 2 * current_score * 1.1
            active_mask = dists < (2.0 * current_score * 1.1)
            # Exclude diagonal
            np.fill_diagonal(active_mask, False)
            
            if np.any(active_mask):
                # Normalize diff vectors
                # diff[i,j] = p_i - p_j
                # unit_vec[i,j] = diff[i,j] / d_ij
                unit_vecs = diff / dists_safe[:, :, np.newaxis]
                
                # Weights using smooth sigmoid
                # w = 1 / (1 + exp((d - target_d) / softness))
                # For d << target_d, w ~ 1; for d >> target_d, w ~ 0
                exponent = (dists - target_d) / softness
                # Clip to avoid overflow
                exponent = np.clip(exponent, -50, 50)
                w = 1.0 / (1.0 + np.exp(exponent))
                # Zero out inactive
                w[~active_mask] = 0.0
                
                # Force on i is sum over j of w[i,j] * unit_vec[i,j]
                # Note: unit_vec[i,j] points from j to i. So adding it pushes i away from j. Correct.
                force_sum = np.sum(w[:, :, np.newaxis] * unit_vecs, axis=1)
                
                grads += force_sum
            
            # Normalize gradients to prevent large steps
            grad_norms = np.linalg.norm(grads, axis=1)
            max_grad = np.max(grad_norms)
            if max_grad > 1e-10:
                grads /= max_grad
            
            # Update points
            points += lr * grads
            
            # Project back to unit circle
            radii = np.linalg.norm(points, axis=1)
            mask = radii > 0.999
            if np.any(mask):
                points[mask] *= 0.999 / radii[mask]
            
            # Compute score
            new_score = compute_score(points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
                no_improve = 0
            else:
                no_improve += 1
                lr *= 0.95
                # If stuck, try a small random jitter to escape local minima
                if no_improve > 5:
                    points += np.random.randn(N, 2) * 0.002
                    # Project back after jitter
                    radii = np.linalg.norm(points, axis=1)
                    mask = radii > 0.999
                    if np.any(mask):
                        points[mask] *= 0.999 / radii[mask]
            
            if no_improve > 30 or lr < 1e-6:
                break
        
        return best_points, best_score

    def local_optimize(points, max_iter=500):
        """Fast local optimization using O(N) score for single point updates."""
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.01
        no_improve_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline - 2.0:
                    break
                
                for _ in range(2):
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = lr * math.cos(angle)
                    dy = lr * math.sin(angle)
                    
                    old_x, old_y = points[i]
                    new_x, new_y = old_x + dx, old_y + dy
                    
                    r_new = math.hypot(new_x, new_y)
                    if r_new > 0.999:
                        scale = 0.999 / r_new
                        new_x *= scale
                        new_y *= scale
                    
                    # Compute distance to closest neighbor for new pos
                    diffs = points - np.array([new_x, new_y])
                    dists = np.linalg.norm(diffs, axis=1)
                    dists[i] = np.inf
                    closest_dist = np.min(dists)
                    
                    r_new = math.hypot(new_x, new_y)
                    bound_dist = 1.0 - r_new
                    
                    new_local_r = min(closest_dist / 2.0, bound_dist)
                    
                    if new_local_r > current_score:
                        points[i] = [new_x, new_y]
                        improved = True
                        break
            
            if it % 50 == 0 or not improved:
                current_score = compute_score(points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    lr *= 0.95
                    if no_improve_count > 10 or lr < 1e-5:
                        break
            else:
                no_improve_count = 0
        
        final_score = compute_score(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()
            
        return best_points, best_score

    # Generate initial configuration
    points = initial_hex(N)
    
    # Phase 1: Gradient-based optimization
    points, score = gradient_optimize(points, max_iter=200)
    
    # Phase 2: Local random search (to escape any local minima)
    points, score2 = local_optimize(points, max_iter=150)
    if score2 > score:
        points, score = points, score2
    
    # Phase 3: Fine gradient polish
    if time.time() < deadline - 25:
        points, score3 = gradient_optimize(points, max_iter=150)
        if score3 > score:
            points, score = points, score3
    
    # Phase 4: Final local polish
    if time.time() < deadline - 10:
        points, score4 = local_optimize(points, max_iter=100)
        if score4 > score:
            points, score = points, score4
    
    # Ensure we're within the unit circle (final safety check)
    radii = np.linalg.norm(points, axis=1)
    mask = radii > 0.9999
    if np.any(mask):
        points[mask] *= 0.9999 / radii[mask]
    
    return points
# EVOLVE-BLOCK-END
