"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Target structure based on known optimal packings for N=50:
    # Center (1), Ring 1 (6), Ring 2 (12), Ring 3 (21), Ring 4 (10)
    # Total = 1 + 6 + 12 + 21 + 10 = 50
    # Note: The exact distribution might vary slightly, but this is a strong heuristic.
    # Let's try a standard multi-ring setup.
    
    # We will optimize the radius of each ring and the angular offset of each ring.
    # Variables: [r1, r2, r3, r4, theta1, theta2, theta3, theta4]
    # r_i is the distance from center to the i-th ring.
    # theta_i is the angular rotation of the i-th ring.
    
    ring_counts = [1, 6, 12, 21, 10]
    num_rings = len(ring_counts) - 1 # Exclude center ring for optimization vars (fixed at 0)
    
    def get_positions(params):
        """
        params: [r1, r2, r3, r4, theta1, theta2, theta3, theta4]
        Returns (N, 2) array of centers.
        """
        centers = []
        # Center circle
        centers.append([0.0, 0.0])
        
        for i in range(num_rings):
            r = params[i]
            theta_offset = params[num_rings + i]
            n = ring_counts[i+1]
            # Generate angles
            angles = theta_offset + 2 * np.pi * np.arange(n) / n
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            centers.extend(zip(xs, ys))
            
        return np.array(centers)

    def energy(params, beta=50.0):
        """
        Minimize the negative of the minimum distance constraint.
        We want to maximize r_min = min(min_pairwise_dist/2, min_dist_to_boundary)
        Equivalent to minimizing -r_min.
        To make it differentiable, we use a soft-min approximation.
        
        Constraints:
        1. All centers must be inside unit circle: |c_i| + r <= 1  => r <= 1 - |c_i|
        2. All pairs must be non-overlapping: |c_i - c_j| >= 2r => r <= |c_i - c_j|/2
        
        So r_max = min( min_i(1-|c_i|), min_{i<j}(|c_i-c_j|/2) )
        We want to maximize r_max.
        
        We can define an objective function that penalizes violations of a target radius R.
        Or simply minimize -soft_min(constraints).
        
        Let's use a direct approach: 
        Objective = - soft_min( [1-|c_i| for all i, |c_i-c_j|/2 for all i<j] )
        
        However, calculating all pairwise distances is O(N^2) = 2500 ops, which is fast.
        N=50 is small.
        """
        if time.time() > deadline:
            return 1e10 # Penalty if we run out of time during eval (shouldn't happen usually)
            
        centers = get_positions(params)
        
        # Boundary constraints: 1 - |c_i|
        dists_to_boundary = 1.0 - np.linalg.norm(centers, axis=1)
        
        # Pairwise constraints: |c_i - c_j| / 2
        # We only need the minimum, but soft-min needs all values or a good approximation.
        # For N=50, we can compute all pairwise distances.
        # To save time, we can subsample or just compute all. 50*49/2 = 1225 pairs.
        
        # Vectorized pairwise distance calculation
        # centers shape (50, 2)
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=-1)
        # This creates a 50x50x2 array -> 50x50 distance matrix.
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        # Zero out diagonal
        np.fill_diagonal(dists_sq, 1e10)
        
        min_pairwise_dist = np.sqrt(np.min(dists_sq)) / 2.0
        
        # We want to maximize r = min(min_boundary, min_pairwise)
        # Let's use a soft minimum of the two global minima? 
        # Actually, the binding constraint is the minimum of ALL constraints.
        # Let's collect all constraint values:
        # boundary_vals = 1 - |c_i|
        # pairwise_vals = |c_i - c_j| / 2
        
        # Instead of soft-min over 1275 values, we can just use the hard min if we use a 
        # penalty method or if we optimize the parameters to increase the minimum.
        # A common trick: minimize -min(constraints) is non-smooth.
        # Use: minimize - (sum(constraints^k) / len(constraints))^(1/k) for large k?
        # Or simpler: minimize the violation of a target radius R.
        
        # Let's try a different objective: 
        # Maximize r such that all constraints >= r.
        # This is equivalent to minimizing the maximum violation.
        
        # Let's use a simple gradient-based approach on the parameters to maximize the minimum distance.
        # Since N is small, we can compute the exact min distance and use subgradients or 
        # just rely on the fact that L-BFGS-B can handle non-smooth functions reasonably well 
        # if we smooth it slightly.
        
        # Smoothed min: -soft_min(constraints)
        # soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
        # We have two groups: boundary and pairwise.
        
        # To keep it efficient, let's just use the hard min and add a tiny bit of noise? 
        # No, let's use the soft min on the combined set of critical distances.
        
        # Critical distances are the smallest few. 
        # For simplicity and robustness with N=50, let's compute all pairwise distances 
        # and boundary distances, flatten them, and take the soft min.
        
        # Pairwise distances (upper triangle)
        idx = np.triu_indices(50, k=1)
        pair_dists = np.sqrt(dists_sq[idx]) / 2.0
        
        # Boundary distances
        bound_dists = 1.0 - np.linalg.norm(centers, axis=1)
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        # Soft min
        beta = 20.0
        # Clip to avoid overflow in exp
        all_dists_clipped = np.clip(all_dists, 1e-6, None)
        soft_min_val = -1.0 / beta * np.log(np.sum(np.exp(-beta * all_dists_clipped)))
        
        # We want to maximize this, so minimize negative
        return -soft_min_val

    def objective(params):
        # Add penalty if any center is outside or if params are invalid
        # r_i must be positive
        if np.any(params[:num_rings] < 0):
            return 1e10
        # r_i must be less than 1
        if np.any(params[:num_rings] > 1.0):
            return 1e10
            
        return energy(params)

    # Initial guess
    # Radii roughly proportional to sqrt(count) or just evenly spaced?
    # Evenly spaced in area?
    # Let's start with radii: 0.2, 0.4, 0.6, 0.8
    # Angles: 0, pi/6, 0, pi/10 (alternating to interlock)
    
    x0 = np.array([0.25, 0.5, 0.75, 0.9, 0.0, np.pi/6, 0.0, np.pi/10])
    
    bounds = [(0.0, 1.0) for _ in range(num_rings)] + [(0, 2*np.pi) for _ in range(num_rings)]
    
    best_score = 0.0
    best_params = x0.copy()
    
    # Run optimization
    # Use L-BFGS-B
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
        
        if res.success or res.fun < objective(x0):
            best_params = res.x
            # Calculate actual score
            centers = get_positions(best_params)
            # Verify validity
            if np.all(np.linalg.norm(centers, axis=1) < 1.0) and np.all(np.isfinite(centers)):
                # Compute raw score
                diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=-1)
                np.fill_diagonal(dists_sq, 1e10)
                min_pair = np.sqrt(np.min(dists_sq)) / 2.0
                min_bound = 1.0 - np.max(np.linalg.norm(centers, axis=1))
                score = min(min_pair, min_bound)
                if score > best_score:
                    best_score = score
        else:
            # Fallback to initial if optimization failed badly
            centers = get_positions(x0)
            if np.all(np.linalg.norm(centers, axis=1) < 1.0):
                 diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                 dists_sq = np.sum(diff**2, axis=-1)
                 np.fill_diagonal(dists_sq, 1e10)
                 min_pair = np.sqrt(np.min(dists_sq)) / 2.0
                 min_bound = 1.0 - np.max(np.linalg.norm(centers, axis=1))
                 best_score = min(min_pair, min_bound)
                 best_params = x0

    except Exception:
        # Fallback to a simple hexagonal lattice if optimization crashes
        pass

    # If the ring structure failed or scored poorly, try a quick local refinement on a hex lattice
    # Or just return the ring result if it's valid.
    
    # Let's ensure we return a valid (50, 2) array.
    final_centers = get_positions(best_params)
    
    # Check if final_centers is valid
    if not np.all(np.isfinite(final_centers)) or np.any(np.linalg.norm(final_centers, axis=1) >= 1.0):
        # Fallback to hex lattice
        spacing = 0.25
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                final_centers = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98
            if spacing < 0.01:
                break

    return final_centers

# EVOLVE-BLOCK-END
