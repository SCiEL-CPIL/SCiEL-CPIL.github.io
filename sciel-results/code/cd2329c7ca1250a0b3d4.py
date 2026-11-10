# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Uses Sequential Convex Programming with bisection on the minimum distance constraint.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2

    # Initial configuration: regular hexagonal-like grid or random
    # Start with a reasonable initial guess - a square grid perturbed
    np.random.seed(42)
    
    # Create a 4x4 grid as initial guess, then optimize
    # Actually, let's use a better initial configuration
    # Use a hexagonal lattice pattern for 16 points
    # Or start with random and refine
    
    # Better: use a known good configuration and refine
    # Start with points on a circle or grid
    # Let's use a 4x4 grid scaled appropriately
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Add small random perturbation to break symmetry
    points += 0.01 * np.random.randn(n, d)
    
    def compute_score(pts):
        """Compute R = (d_min / d_max)^2"""
        if len(pts) < 2:
            return 0.0
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    def optimize_with_scp(points_init, max_iters=200, tol=1e-8):
        """
        Sequential Convex Programming:
        - Fix target max distance D_max
        - Minimize D_max subject to all pairwise distances >= d_min_target
        - Use bisection on d_min_target to find optimal packing
        
        Actually, let's frame it as:
        - For a given minimum distance constraint d_min_req, find the configuration
          that minimizes the maximum pairwise distance
        - Bisection on d_min_req to maximize the ratio
        
        Simpler approach: directly optimize using iterative projection
        """
        pts = points_init.copy()
        
        # Normalize initial points
        center = np.mean(pts, axis=0)
        pts = pts - center
        scale = np.max(pdist(pts))
        if scale > 0:
            pts = pts / scale
        
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        for it in range(max_iters):
            # Compute current pairwise distances
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max == 0:
                break
                
            current_ratio = d_min / d_max
            current_score = current_ratio ** 2
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            
            # Gradient-based step: 
            # We want to maximize min(d_ij) / max(d_ij)
            # Use a smooth approximation or direct optimization
            
            # For each point, compute gradient of a proxy objective
            # Let's use: maximize log(d_min) - log(d_max) approximately
            # Or use the approach of pushing apart close pairs and pulling together far pairs
            
            # Compute gradients
            gradients = np.zeros_like(pts)
            
            # Get indices of min and max distance pairs
            min_idx = np.argmin(dists)
            max_idx = np.argmax(dists)
            
            # Convert pdist index to (i, j)
            def idx_to_pair(idx, n):
                # pdist uses upper triangular indexing
                i = 0
                count = 0
                while count < idx:
                    i += 1
                    count += (n - i - 1)
                j = i + 1 + (idx - count)
                return i, j
            
            min_i, min_j = idx_to_pair(min_idx, n)
            max_i, max_j = idx_to_pair(max_idx, n)
            
            # Push apart the closest pair
            diff_min = pts[min_i] - pts[min_j]
            norm_min = np.linalg.norm(diff_min)
            if norm_min > 1e-10:
                unit_min = diff_min / norm_min
                gradients[min_i] += unit_min * 0.1
                gradients[min_j] -= unit_min * 0.1
            
            # Pull together the farthest pair
            diff_max = pts[max_i] - pts[max_j]
            norm_max = np.linalg.norm(diff_max)
            if norm_max > 1e-10:
                unit_max = diff_max / norm_max
                gradients[max_i] -= unit_max * 0.1
                gradients[max_j] += unit_max * 0.1
            
            # Also consider all pairs for a more robust gradient
            # Use a weighted sum: for each pair, if distance < target, push apart
            # if distance > target, pull together
            target_dist = d_min  # or some geometric mean
            
            # For simplicity, just use the min/max approach above
            # Add a small regularization to keep points centered
            center = np.mean(pts, axis=0)
            gradients -= 0.01 * (pts - center)
            
            # Update points
            step_size = 0.05
            pts_new = pts + step_size * gradients
            
            # Re-center
            center_new = np.mean(pts_new, axis=0)
            pts_new = pts_new - center_new
            
            # Check if score improved
            score_new = compute_score(pts_new)
            if score_new > current_score:
                pts = pts_new
            else:
                # Try smaller step
                step_size *= 0.5
                pts_new = pts + step_size * gradients
                center_new = np.mean(pts_new, axis=0)
                pts_new = pts_new - center_new
                score_new = compute_score(pts_new)
                if score_new > current_score:
                    pts = pts_new
                else:
                    # No improvement, break
                    if step_size < 1e-6:
                        break
            
            # Convergence check
            if it > 0 and abs(score_new - best_score) < tol:
                break
        
        return best_pts, best_score
    
    # Try multiple initial configurations and keep the best
    best_points = None
    best_score = -1
    
    # Configuration 1: 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    X, Y = np.meshgrid(x, y)
    pts1 = np.column_stack([X.ravel(), Y.ravel()]) + 0.01 * np.random.randn(n, d)
    p1, s1 = optimize_with_scp(pts1)
    if s1 > best_score:
        best_score = s1
        best_points = p1
    
    # Configuration 2: Hexagonal-like
    np.random.seed(123)
    pts2 = np.random.randn(n, d)
    p2, s2 = optimize_with_scp(pts2)
    if s2 > best_score:
        best_score = s2
        best_points = p2
    
    # Configuration 3: Circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts3 = np.column_stack([np.cos(angles), np.sin(angles)])
    p3, s3 = optimize_with_scp(pts3)
    if s3 > best_score:
        best_score = s3
        best_points = p3
    
    # Final normalization: scale so that d_max = 1 (or any convenient scale)
    if best_points is not None:
        dists = pdist(best_points)
        d_max = np.max(dists)
        if d_max > 0:
            best_points = best_points / d_max
    
    return best_points


# EVOLVE-BLOCK-END
