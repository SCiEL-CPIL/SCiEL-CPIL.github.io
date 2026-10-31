"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 150s limit, leave margin for overhead and safety

    def score_points(pts):
        """
        Calculate (max_dist / min_dist)^2 for a set of points.
        pts: np.array of shape (N, 2)
        """
        # Compute pairwise squared distances
        # Use broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Get upper triangle indices to avoid self-distances and duplicates
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    def smooth_objective(pts, beta=50.0):
        """
        Smooth surrogate for (max_dist_sq / min_dist_sq) using log-sum-exp.
        This is differentiable and helps the optimizer navigate the nonsmooth landscape.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Numerical safety
        d_sq = np.maximum(d_sq, 1e-15)
        
        # Soft max: log-sum-exp
        log_max = np.log(np.sum(np.exp(d_sq / beta))) * beta
        # Soft min: -log-sum-exp of -d_sq
        log_min = -np.log(np.sum(np.exp(-d_sq / beta))) * beta
        
        if log_min < 1e-12:
            return 1e10
        return log_max / log_min

    # 1. Generate the 19-point hexagonal cluster
    # Triangular lattice basis
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    # Generate candidates in a range that covers the 3rd ring just in case, 
    # but we specifically want the 19-point cluster (rings 0, 1, 2)
    # Ring 0: (0,0)
    # Ring 1: distance 1 (6 pts)
    # Ring 2: distance sqrt(3) (6 pts) and distance 2 (6 pts) -> Total 12 pts in ring 2?
    # Let's list them properly.
    # Center: (0,0)
    # Dist 1: (1,0), (-1,0), (0.5, sqrt3/2), (-0.5, -sqrt3/2), (0.5, -sqrt3/2), (-0.5, sqrt3/2)
    # Dist sqrt(3): (1.5, sqrt3/2), (1.5, -sqrt3/2), (-0.5, sqrt3), (-0.5, -sqrt3), (-1.5, sqrt3/2), (-1.5, -sqrt3/2) -> 6 pts
    # Dist 2: (2,0), (-2,0), (1, sqrt3), (-1, -sqrt3), (1, -sqrt3), (-1, sqrt3) -> 6 pts
    # Total 1 + 6 + 6 + 6 = 19 points.
    
    # Let's generate them systematically
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Check if it's in the 19-point cluster
            # Dist sq must be <= 4
            p = i * v1 + j * v2
            d_sq = np.sum(p**2)
            if d_sq <= 4.0 + 1e-9:
                hex_points.append(p)
    
    hex_points = np.array(hex_points)
    
    # Verify count
    # Distances:
    # 0: (0,0) -> 1
    # 1: (1,0), (-1,0), (0.5, .866), (-0.5, -.866), (0.5, -.866), (-0.5, .866) -> 6
    # 3: (1.5, .866), (1.5, -.866), (-.5, 1.732), (-.5, -1.732), (-1.5, .866), (-1.5, -.866) -> 6
    # 4: (2,0), (-2,0), (1, 1.732), (-1, -1.732), (1, -1.732), (-1, 1.732) -> 6
    # Total 19.
    if len(hex_points) != 19:
        # Fallback if generation logic fails
        print(f"Warning: Expected 19 points, got {len(hex_points)}")
        # Just take first 16
        return hex_points[:16]

    # 2. Identify the outer ring (distance squared == 4)
    d_sqs = np.sum(hex_points**2, axis=1)
    outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
    inner_indices = np.where(np.isclose(d_sqs, 3.0))[0]
    # Inner 13 points are fixed for the best compactness usually.
    # We remove 3 points from the 12 outer points.
    
    # 3. Pre-filter: Evaluate all combinations of removing 3 points from the outer ring
    # This is C(12, 3) = 220 combinations. Very fast.
    best_combos = []
    for remove_idx in itertools.combinations(outer_indices, 3):
        keep_idx = [i for i in range(19) if i not in remove_idx]
        pts = hex_points[keep_idx]
        score = score_points(pts)
        best_combos.append((score, pts))
        
    # Sort by score
    best_combos.sort(key=lambda x: x[0])
    
    # 4. Refine the top K candidates using optimization
    # We fix p0=(0,0) and p1=(1,0) to remove translation/rotation degrees of freedom.
    # We optimize the remaining 14 points (28 params).
    
    K = 5  # Number of candidates to refine
    best_score = float('inf')
    best_params = None
    best_initial_pts = None
    
    for i in range(min(K, len(best_combos))):
        if time.time() > deadline - 20:
            break
            
        score, init_pts = best_combos[i]
        
        # Find the pair with minimum distance to fix the scale and rotation
        diff = init_pts[:, np.newaxis, :] - init_pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_idx = np.argmin(d_sq)
        p, q = np.unravel_index(min_idx, (N, N))
        
        if p > q:
            p, q = q, p
        
        p0 = init_pts[p]
        p1 = init_pts[q]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            continue
            
        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        # Transform all points
        transformed = (init_pts - p0) @ R_mat.T / norm_v
        
        # Reorder points so that p is at index 0 and q is at index 1
        perm = list(range(N))
        perm[0] = p
        perm[1] = q
        remaining = [idx for idx in range(N) if idx != p and idx != q]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        # init_params are the coordinates of points 2..15
        init_params = transformed[2:].flatten()
        
        # Define smooth objective function for optimizer
        def objective_smooth(params, beta=50.0):
            if time.time() > deadline:
                return 1e10
            pts = np.zeros((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = params.reshape(-1, 2)
            
            return smooth_objective(pts, beta=beta)
        
        # Define exact objective for final evaluation
        def objective_exact(params):
            if time.time() > deadline:
                return 1e10
            pts = np.zeros((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = params.reshape(-1, 2)
            
            return score_points(pts)
        
        # Two-phase optimization:
        # Phase 1: Smooth surrogate with moderate beta
        # Phase 2: Exact objective with high-precision polish
        
        try:
            # Phase 1: Smooth optimization
            res1 = minimize(objective_smooth, init_params, method='Powell', 
                           options={'maxiter': 300, 'maxfev': 3000, 'disp': False})
            
            current_params = res1.x if res1.fun < 1e9 else init_params
            
            # Phase 2: High-precision polish with exact objective
            if time.time() < deadline - 10:
                res2 = minimize(objective_exact, current_params, method='Powell', 
                               options={'maxiter': 500, 'maxfev': 5000, 'disp': False,
                                       'ftol': 1e-10, 'xtol': 1e-10})
                current_params = res2.x
                current_score = res2.fun
            else:
                current_score = objective_exact(current_params)
            
            if current_score < best_score:
                best_score = current_score
                best_params = current_params
                best_initial_pts = init_pts
        except Exception:
            pass

    # Construct final points from best_params
    if best_params is not None:
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        
        # Ensure no points are exactly identical (numerical error)
        # If min dist is too small, fall back to the best lattice subset
        final_score = score_points(final_pts)
        if final_score > best_score * 1.1 or final_score > 10:
             # Fallback to the best unoptimized lattice subset
             _, best_unopt_pts = best_combos[0]
             final_pts = best_unopt_pts
             final_score = score_points(final_pts)
        
        return final_pts
    else:
        # Fallback: return the best lattice subset found
        _, best_unopt_pts = best_combos[0]
        return best_unopt_pts

# EVOLVE-BLOCK-END
