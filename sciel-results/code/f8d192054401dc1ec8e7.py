import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    def _compute_radius(centers):
        """Compute the maximum valid radius for given centers."""
        if centers.shape[0] != N:
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check inside cube
        if np.any(centers < -0.5) or np.any(centers > 0.5):
            return 0.0
        # Minimum distance to boundary
        dist_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_boundary = np.min(dist_boundary)
        # Minimum pairwise distance
        n = len(centers)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        if min_pair == 0.0:
            return 0.0
        return min(min_pair / 2.0, min_boundary)

    def _soft_objective(params):
        """
        Objective to maximize: Sum of log(distances) for pairs and boundaries.
        We minimize the negative of this.
        """
        centers = params.reshape(N, 3)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e-10) # Avoid log(0)
        
        # Sum of logs of pairwise distances (only upper triangle to avoid double counting, or all is fine for gradient)
        # Using all is easier for vectorization, just divide by 2 if needed for scale, but gradient direction is same.
        pair_score = np.sum(np.log(dists))
        
        # Boundary distances
        dist_boundary = 0.5 - np.abs(centers)
        min_dist_per_point = np.min(dist_boundary, axis=1)
        min_dist_per_point = np.maximum(min_dist_per_point, 1e-10)
        
        boundary_score = np.sum(np.log(min_dist_per_point))
        
        return -(pair_score + boundary_score)

    def _soft_gradient(params):
        """
        Gradient of the negative soft objective.
        """
        centers = params.reshape(N, 3)
        
        # Pairwise gradient
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e-10)
        
        # grad log(d_ij) w.r.t c_i is (c_i - c_j) / d_ij^2
        # grad log(d_ij) w.r.t c_j is (c_j - c_i) / d_ij^2
        # Summing over all j != i for c_i: sum_j (c_i - c_j) / d_ij^2
        
        # Vectorized:
        # inv_dists_sq = 1 / dists^2
        # grad_pair = sum_j (c_i - c_j) * (1/d_ij^2)
        
        inv_dists_sq = 1.0 / (dists ** 2)
        np.fill_diagonal(inv_dists_sq, 0.0)
        
        # diff is (i - j)
        # grad_pair[i] = sum_j diff[i,j] * inv_dists_sq[i,j]
        grad_pair = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
        
        # Boundary gradient
        # dist_boundary = 0.5 - |c|
        # grad log(0.5 - |c_k|) w.r.t c_k is -1 / (0.5 - |c_k|) * sign(c_k)
        # We need the min component for each point.
        
        # Find the index of the min distance for each point
        # dist_boundary shape (N, 3)
        min_idx = np.argmin(dist_boundary, axis=1)
        
        grad_boundary = np.zeros_like(centers)
        for i in range(N):
            k = min_idx[i]
            d = dist_boundary[i, k]
            d = max(d, 1e-10)
            # Derivative of log(d) is 1/d. Derivative of d w.r.t c_ik is -sign(c_ik).
            grad_boundary[i, k] = -1.0 / d * np.sign(centers[i, k])
            # If center is exactly 0, sign is 0, but it's a singular point. 
            # In practice, centers won't be exactly 0 if we start from BCC.
            # To be safe, if sign is 0, we can pick a small random direction or 0.
            if np.sign(centers[i, k]) == 0:
                 grad_boundary[i, k] = 0.0

        return -(grad_pair + grad_boundary).flatten()

    def _construct_bcc_init():
        """
        Generate a BCC lattice initialization.
        BCC unit cell has points at (0,0,0) and (0.5, 0.5, 0.5).
        We want ~50 points.
        A 3x3x3 grid of cells gives 27 cells -> 54 points.
        This is perfect.
        """
        # Lattice constant a.
        # Total extent needs to fit in [-0.5, 0.5].
        # Grid indices -1, 0, 1.
        # Points at i*a and (i+0.5)*a.
        # Range: -a to 1.5a (if i=-1,0,1).
        # Center of mass shift will center it.
        # Let's pick a such that the bounding box is roughly 0.8 to leave room for optimization.
        a = 0.3
        
        pts = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    pts.append([i*a, j*a, k*a])
                    pts.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
        
        pts = np.array(pts)
        
        # We have 54 points. We need 50.
        # Calculate distance from origin for each point.
        # Keep the 50 closest to the origin? Or the ones that form the most regular shape?
        # Actually, the 4 corners of the 3x3x3 grid are the furthest.
        # The "corners" of the superlattice are at (-a,-a,-a), (a,a,a) etc.
        # Let's sort by distance to center (0,0,0) and keep the 50 smallest?
        # This might bias towards the center. 
        # Better: Just take the first 50? The BCC grid is symmetric.
        # Removing 4 points breaks symmetry. 
        # A better strategy: Start with 54, optimize, then remove the 4 with smallest "slack" (distance to nearest neighbor/boundary)?
        # For simplicity and speed, let's just take 50. The optimizer will fix the asymmetry.
        # To be slightly more symmetric, let's remove the 4 points furthest from the origin.
        dists_origin = np.linalg.norm(pts, axis=1)
        sort_idx = np.argsort(dists_origin)
        pts = pts[sort_idx[:50]]
        
        # Center the configuration
        pts -= np.mean(pts, axis=0)
        
        # Scale to fit well inside. The current max coord is roughly 0.45.
        # Let's scale so max abs coord is 0.4.
        max_coord = np.max(np.abs(pts))
        if max_coord > 0:
             pts *= 0.4 / max_coord
             
        return pts

    def _optimize_soft(centers_start, time_limit=30.0):
        """Run gradient ascent on the soft objective."""
        start_time_opt = time.time()
        params = centers_start.flatten()
        
        # Use L-BFGS-B with gradient
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        best_params = params.copy()
        # Initial score for comparison
        # We don't strictly need to compare soft scores, just take the result.
        
        try:
            result = minimize(
                _soft_objective, 
                params, 
                jac=_soft_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.success:
                new_centers = result.x.reshape(N, 3)
                # Clip to ensure validity
                new_centers = np.clip(new_centers, -0.499999, 0.499999)
                return new_centers
        except Exception:
            pass
        
        return centers_start

    # Main logic
    global_start = time.time()
    deadline = time_limit
    
    # 1. Initialize with BCC
    centers_bcc = _construct_bcc_init()
    best_centers = centers_bcc.copy()
    best_r = _compute_radius(best_centers)
    
    # 2. Optimize using Soft Objective
    time_remaining = deadline - (time.time() - global_start)
    if time_remaining > 10:
        opt_centers = _optimize_soft(centers_bcc, time_limit=min(60.0, time_remaining - 5))
        opt_r = _compute_radius(opt_centers)
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # 3. Local Refinement using Exact Min-Radius Objective
    # The soft objective is a proxy. We can do a few steps of optimizing the exact min-radius
    # or just trust the soft result. 
    # Let's try a quick L-BFGS on the exact radius (negative) to polish.
    # This might be slow if the landscape is rough, but we are close.
    
    def _exact_objective(params):
        centers = params.reshape(N, 3)
        r = _compute_radius(centers)
        return -r

    def _exact_gradient(params):
        # Numerical gradient is safer and faster to implement correctly for non-smooth min.
        # Or we can skip gradient and let L-BFGS-B use numerical.
        # With N=50, 150 variables. Numerical grad takes 150 evals.
        # Eval is O(N^2) = 2500 ops. 150*2500 = 375k ops. Fast.
        centers = params.reshape(N, 3)
        r = _compute_radius(centers)
        
        # Find the limiting distance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        dist_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_bound = np.min(dist_boundary)
        
        r_val = min(min_pair_dist / 2.0, min_bound)
        
        grad = np.zeros_like(params)
        
        # If pair distance is limiting
        if min_pair_dist / 2.0 < min_bound + 1e-9:
            # Find the pair
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            # Gradient of dist(i,j) / 2
            # d/dc_i (||c_i - c_j|| / 2) = (c_i - c_j) / (2 * ||c_i - c_j||)
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                grad[i*3:(i+1)*3] = vec / (2.0 * norm)
                grad[j*3:(j+1)*3] = -vec / (2.0 * norm)
        else:
            # Boundary is limiting
            # Find the point and face
            min_point_idx = np.argmin(dist_boundary)
            min_face_idx = np.argmin(0.5 - np.abs(centers[min_point_idx]))
            # Gradient of (0.5 - |c_k|) is -sign(c_k)
            c_val = centers[min_point_idx, min_face_idx]
            if c_val > 0:
                grad[min_point_idx*3 + min_face_idx] = -1.0
            elif c_val < 0:
                grad[min_point_idx*3 + min_face_idx] = 1.0
            else:
                grad[min_point_idx*3 + min_face_idx] = 0.0
                
        return -grad # We are minimizing -r, so gradient of -r is -grad_r

    time_remaining = deadline - (time.time() - global_start)
    if time_remaining > 10:
        params = best_centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        try:
            result = minimize(
                _exact_objective, 
                params, 
                jac=_exact_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.success:
                new_centers = result.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -0.499999, 0.499999)
                new_r = _compute_radius(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
        except Exception:
            pass

    # Final validity check
    final_centers = np.clip(best_centers, -0.5, 0.5)
    final_r = _compute_radius(final_centers)
    
    if final_r <= 0.0:
        # Fallback
        final_centers = _construct_bcc_init()
        
    return final_centers

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    r = _compute_radius(res)
    print(f"Radius: {r}")
