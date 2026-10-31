"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 16
D = 2

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    def score_points(pts):
        """Compute (max_d / min_d)^2 for a set of points."""
        # Check for distinct points
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    def prepare_params(pts):
        """
        Reduces dimensionality by fixing the first two points (closest pair)
        to (0,0) and (1,0).
        Returns:
        params: 28-dim array of the remaining 14 points' coordinates.
        transform_info: tuple (i_min, j_min, p0, p1, norm) to reconstruct if needed, 
                        though we reconstruct from params directly in objective.
        """
        d = pdist(pts, 'euclidean')
        min_idx = np.argmin(d)
        # Convert linear index to pair indices
        # pdist returns upper triangle
        # k-th element corresponds to pair (i, j) where i < j
        # Formula: i = int((sqrt(8*k + 1) - 1) / 2), j = i + 1 + (k - i*(i+1)/2) ? 
        # Easier: use np.triu_indices
        iu = np.triu_indices(N, 1)
        i_min, j_min = iu[0][min_idx], iu[1][min_idx]
        
        p0 = pts[i_min]
        p1 = pts[j_min]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
             # Fallback if points are identical (should not happen with distinct init)
             return None, None

        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        # Transform all points
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        # Reorder so i_min is 0 and j_min is 1
        perm = list(range(N))
        perm[0] = i_min
        perm[1] = j_min
        remaining = [i for i in range(N) if i != i_min and i != j_min]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        # transformed[0] should be [0,0]
        # transformed[1] should be [1,0]
        params = transformed[2:].flatten()
        return params, (i_min, j_min)

    def objective(params):
        """
        Objective function for optimization.
        Reconstructs points from params (p0=0, p1=1, rest=params) and computes score.
        """
        if time.time() > deadline:
            return 1e10
            
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        return score_points(pts)

    def generate_seeds():
        """Generate various initial configurations."""
        seeds = []
        
        # 1. Square Grid 4x4
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        seeds.append(grid)
        
        # 2. Hexagonal Lattice subset
        # Center at 0,0
        hex_pts = []
        R = 3
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append((x, y))
        hex_pts = np.array(hex_pts)
        # Take closest 16 to origin
        dists = np.sum(hex_pts**2, axis=1)
        idx = np.argsort(dists)[:16]
        seeds.append(hex_pts[idx])
        
        # 3. Circle Packing (Fibonacci/Lattice on circle)
        # Place points on a circle of radius 1, then perturb
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        seeds.append(circle_pts)
        
        # 4. Random Uniform
        rng = np.random.default_rng(42)
        rand_pts = rng.uniform(-1, 1, size=(16, 2))
        seeds.append(rand_pts)
        
        # 5. Two concentric rings
        # 8 points on inner ring, 8 on outer
        angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        r_inner = 0.5
        r_outer = 1.5
        inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        ring_pts = np.vstack([inner_pts, outer_pts])
        seeds.append(ring_pts)
        
        return seeds

    best_score = 1e10
    best_params = None
    best_pts = None

    seeds = generate_seeds()
    
    # Add a few random perturbations of the best seed found so far? 
    # Or just iterate through seeds.
    
    for seed_idx, seed_pts in enumerate(seeds):
        if time.time() > deadline - 10:
            break
            
        params, info = prepare_params(seed_pts)
        if params is None:
            continue
            
        # Initial score
        curr_score = objective(params)
        if curr_score < best_score:
            best_score = curr_score
            best_params = params.copy()
            
        # Optimize with Powell
        # Powell is derivative-free, good for non-smooth
        # maxfev: max function evaluations
        # We have ~140s total. 5 seeds. ~20s per seed.
        # Powell can be expensive. Limit maxfev to 5000?
        
        try:
            res = minimize(objective, params, method='Powell', 
                           options={'maxiter': 1000, 'maxfev': 5000, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception as e:
            pass

    # Refine with Nelder-Mead or Powell again on the best result with tighter settings?
    # Or use a smooth surrogate with L-BFGS-B for a final polish.
    # Powell is usually good enough. Let's try a second pass with L-BFGS-B using soft-min/max if time permits.
    
    if best_params is not None and time.time() < deadline - 5:
        
        def soft_objective(params, beta=50.0):
            pts = np.zeros((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = params.reshape(-1, 2)
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            iu = np.triu_indices(N, 1)
            d_sq = dists_sq[iu]
            
            d_sq = np.maximum(d_sq, 1e-12)
            
            max_d_sq = np.max(d_sq)
            soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
            
            min_d_sq = np.min(d_sq)
            soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
            
            if soft_min_d_sq < 1e-8:
                return 1e10
                
            return soft_max_d_sq / soft_min_d_sq

        try:
            # Polish with L-BFGS-B on soft objective
            res2 = minimize(soft_objective, best_params, method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            
            # Check exact score of the result
            exact_score = objective(res2.x)
            if exact_score < best_score:
                best_score = exact_score
                best_params = res2.x.copy()
                
        except Exception:
            pass

    # Reconstruct final points
    # We need to map params back to a full set of points.
    # The params correspond to points 2..15 in the canonical frame (p0=0, p1=1).
    # The returned points can be in any position, as long as they are distinct.
    # The canonical frame points are:
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    # Verify
    final_score = score_points(final_pts)
    if final_score < best_score:
        best_score = final_score
        # best_pts = final_pts # final_pts is already the best

    return final_pts

# EVOLVE-BLOCK-END
