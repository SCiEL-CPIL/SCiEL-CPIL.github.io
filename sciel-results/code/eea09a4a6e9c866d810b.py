"""Improved packing of 300 equal circles in a unit circle using local optimization."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def initial_hex(n):
        """Generate initial hexagonal lattice points."""
        # Estimate spacing
        s_est = math.sqrt(2 * math.pi * 0.95**2 / (n * math.sqrt(3)))
        
        best_points = None
        best_score = -1
        
        # Try a few spacings around the estimate
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
                
                # Quick score check
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate
            
            if time.time() > deadline - 5:
                break
        
        if best_points is None:
            # Fallback: random
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

    def get_bottlenecks(points, score):
        """Find points that are limiting the score (bottlenecks)."""
        n = len(points)
        bottlenecks = set()
        
        # Check pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Pairs with distance close to 2*score
        threshold = 2.0 * score * 1.05
        mask = dists < threshold
        for i in range(n):
            for j in np.where(mask[i])[0]:
                if i != j:
                    bottlenecks.add(i)
                    bottlenecks.add(j)
        
        # Check boundary
        radii = np.linalg.norm(points, axis=1)
        boundary_threshold = 1.0 - score * 1.05
        for i in range(n):
            if radii[i] > boundary_threshold:
                bottlenecks.add(i)
        
        return bottlenecks

    def local_optimize(points, max_iter=500):
        """Fast local optimization using O(N) score for single point updates."""
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.01
        no_improve_count = 0
        rng = np.random.default_rng(42)
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            indices = rng.permutation(N)
            
            for i in indices:
                if time.time() > deadline - 2.0:
                    break
                
                # Try moving point i in a random direction
                for _ in range(2):
                    angle = rng.uniform(0, 2 * np.pi)
                    dx = lr * math.cos(angle)
                    dy = lr * math.sin(angle)
                    
                    old_x, old_y = points[i]
                    new_x, new_y = old_x + dx, old_y + dy
                    
                    # Check bounds
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
                    
                    # Boundary distance
                    r_new = math.hypot(new_x, new_y)
                    bound_dist = 1.0 - r_new
                    
                    new_local_r = min(closest_dist / 2.0, bound_dist)
                    
                    # Accept if it improves the current global score
                    if new_local_r > current_score:
                        points[i] = [new_x, new_y]
                        improved = True
                        break
            
            # Recompute global score periodically
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
        
        # Final full score computation
        final_score = compute_score(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()
            
        return best_points, best_score

    def gradient_optimize(points, max_time=30.0):
        """Gradient-based optimization using subgradient method."""
        start = time.time()
        n = len(points)
        pts = points.astype(np.float64).copy()
        best_score = compute_score(pts)
        best_pts = pts.copy()
        lr = 0.005
        decay = 0.995
        
        while time.time() - start < max_time:
            # Compute pairwise distances
            diff = pts[:, None, :] - pts[None, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Find minimum pair distance
            min_idx = np.argmin(dists)
            i, j = np.unravel_index(min_idx, (n, n))
            min_pair_dist = dists[i, j] / 2.0
            
            # Compute boundary margins
            norms = np.sqrt(np.sum(pts ** 2, axis=1))
            min_boundary_idx = np.argmin(1.0 - norms)
            min_boundary = 1.0 - norms[min_boundary_idx]
            
            current_score = min(min_pair_dist, min_boundary)
            
            # If we improved, update best
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            
            # Compute gradient
            grad = np.zeros_like(pts)
            
            if min_pair_dist <= min_boundary:
                # Gradient from pair constraint
                d = pts[i] - pts[j]
                norm_d = np.linalg.norm(d)
                if norm_d > 1e-10:
                    grad[i] += d / norm_d
                    grad[j] -= d / norm_d
            else:
                # Gradient from boundary constraint
                p = pts[min_boundary_idx]
                norm_p = np.linalg.norm(p)
                if norm_p > 1e-10:
                    grad[min_boundary_idx] -= p / norm_p
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad /= grad_norm
            
            # Update
            pts += lr * grad
            
            # Project back: ensure points stay inside unit circle
            norms = np.sqrt(np.sum(pts ** 2, axis=1))
            mask = norms > 0.999
            if np.any(mask):
                pts[mask] *= 0.999 / norms[mask]
            
            # Decay learning rate
            lr *= decay
            
            # Check for convergence
            if lr < 1e-8:
                break
        
        return best_pts, best_score

    def pairwise_repulsion_optimize(points, max_time=20.0):
        """Pairwise repulsion optimization using KDTree."""
        start = time.time()
        n = len(points)
        pts = points.astype(np.float64).copy()
        best_score = compute_score(pts)
        best_pts = pts.copy()
        rng = np.random.default_rng(42)
        
        # Initial step size based on minimum distance
        tree = cKDTree(pts)
        dists, indices = tree.query(pts, k=2)
        min_dist = dists[:, 1].min()
        step_size = min_dist * 0.1
        
        for step in range(10000):
            if time.time() - start > max_time:
                break
            
            # Find the pair with minimum distance
            tree = cKDTree(pts)
            dists, indices = tree.query(pts, k=2)
            min_idx = np.argmin(dists[:, 1])
            min_d = dists[min_idx, 1]
            
            # Move points in this pair apart
            i, j = min_idx, indices[min_idx, 1]
            direction = pts[i] - pts[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                direction = rng.normal(size=2)
                norm = np.linalg.norm(direction)
            direction = direction / norm
            
            # Move each point by half the step in opposite directions
            delta = step_size / 2.0 * direction
            pts[i] += delta
            pts[j] -= delta
            
            # Clamp to unit circle
            for idx in [i, j]:
                r = np.linalg.norm(pts[idx])
                if r > 1.0:
                    pts[idx] = pts[idx] / r * 0.999
            
            # Check if score improved
            new_score = compute_score(pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
                step_size *= 0.99
            else:
                # Revert
                pts[i] -= delta
                pts[j] += delta
                step_size *= 0.95
            
            # Occasionally try random perturbations
            if step % 100 == 0:
                idx = rng.integers(n)
                perturbation = rng.normal(scale=step_size, size=2)
                old_pos = pts[idx].copy()
                pts[idx] += perturbation
                r = np.linalg.norm(pts[idx])
                if r > 0.999:
                    pts[idx] *= 0.999 / r
                new_score = compute_score(pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = pts.copy()
                else:
                    pts[idx] = old_pos
        
        return best_pts, best_score

    # Generate initial configuration
    points = initial_hex(N)
    initial_score = compute_score(points)
    
    # Phase 1: Local optimization
    points, score = local_optimize(points, max_iter=200)
    
    # Phase 2: Gradient optimization
    if time.time() < deadline - 60:
        points, score_grad = gradient_optimize(points, max_time=30.0)
        if score_grad > score:
            points = points
            score = score_grad
    
    # Phase 3: Pairwise repulsion optimization
    if time.time() < deadline - 40:
        points, score_rep = pairwise_repulsion_optimize(points, max_time=20.0)
        if score_rep > score:
            points = points
            score = score_rep
    
    # Phase 4: Final local polish
    if time.time() < deadline - 10:
        points, score_final = local_optimize(points, max_iter=100)
        if score_final > score:
            points = points
            score = score_final
    
    return points
# EVOLVE-BLOCK-END
