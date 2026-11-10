import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Strategy:
    1. Define parametric families of point configurations (e.g., concentric rings).
    2. For each family, use a gradient-based optimizer (L-BFGS-B) to maximize the 
       smooth surrogate objective: log(min_dist) - log(max_dist).
       This avoids the non-smoothness of the min/max function directly by working 
       with logs and relying on the optimizer to handle the implicit max/min via 
       the gradient of the active constraints (or by using a soft approximation 
       if gradients are unstable, but L-BFGS-B on the log-sum-exp smoothed version 
       is often more robust).
       
       Actually, a better approach for L-BFGS-B is to minimize -log(min_d) + log(max_d).
       Since min_d and max_d are piecewise smooth, we can use a soft-min/soft-max 
       approximation to ensure differentiability everywhere, or just rely on the 
       optimizer's robustness to kinks if we use a finite difference gradient.
       
       Let's use a soft approximation:
       soft_min(d, t) = -t * log(sum(exp(-d^2/t)))
       soft_max(d, t) = t * log(sum(exp(d^2/t)))
       Objective to minimize: -log(soft_min) + log(soft_max)
       
    3. Compare results from different initial structures (rings, grid, random) 
       and return the best one.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 10  # 590 seconds to be safe

    def get_dists(points):
        """Calculate all pairwise squared distances."""
        # points: (n, 2)
        # Using pdist for efficiency and accuracy
        return pdist(points, 'sqeuclidean')

    def score(points):
        """Calculate the raw metric R = (d_min / d_max)^2."""
        dists_sq = get_dists(points)
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-12:
            return 0.0
        return (np.sqrt(d_min_sq) / np.sqrt(d_max_sq)) ** 2

    def smooth_score(points, temperature=0.01):
        """
        Calculate a smoothed score suitable for gradient-based optimization.
        We want to maximize min/max.
        Equivalent to minimizing: -log(min) + log(max).
        
        Using soft min/max on squared distances to keep it positive and smooth.
        soft_min_sq = -t * log(sum(exp(-d_sq / t)))
        soft_max_sq = t * log(sum(exp(d_sq / t)))
        
        We want to maximize soft_min / soft_max.
        Minimize: -log(soft_min) + log(soft_max)
        """
        dists_sq = get_dists(points)
        
        # Avoid numerical issues with large exponents
        max_d = np.max(dists_sq)
        min_d = np.min(dists_sq)
        
        if max_d == 0:
            return 0.0 # Degenerate, penalty 0? No, we want to minimize loss.
            # If all points are 0, min=max=0. Ratio is undefined. 
            # In the soft version, this might be unstable.
            # Let's add a small epsilon to distances or handle it.
            return 1e6 # Large penalty
        
        # Soft min
        # To avoid overflow in exp(-d/t), we can shift.
        # exp(-d/t) is max when d is min.
        # Let's just use the standard formula carefully.
        t = temperature
        # For soft min: -t * log(sum(exp(-d/t)))
        # For soft max: t * log(sum(exp(d/t)))
        
        # Numerical stability:
        # max of -d/t is -min_d/t
        # min of d/t is min_d/t
        
        # Soft Min
        neg_d_over_t = -dists_sq / t
        max_neg = np.max(neg_d_over_t)
        sum_exp_neg = np.sum(np.exp(neg_d_over_t - max_neg))
        soft_min_sq = -t * (max_neg + np.log(sum_exp_neg))
        
        # Soft Max
        d_over_t = dists_sq / t
        max_pos = np.max(d_over_t)
        sum_exp_pos = np.sum(np.exp(d_over_t - max_pos))
        soft_max_sq = t * (max_pos + np.log(sum_exp_pos))
        
        if soft_min_sq < 1e-12:
            return 1e6 # Penalty for points collapsing
            
        # Objective to minimize: -log(soft_min) + log(soft_max)
        # Since we care about ratio of distances, not squared distances:
        # R = (d_min/d_max)^2 = (d_min_sq/d_max_sq)
        # log R = 2 * (log d_min - log d_max) = log(d_min_sq) - log(d_max_sq)
        # So minimizing -log(d_min_sq) + log(d_max_sq) is correct.
        
        return -np.log(soft_min_sq) + np.log(soft_max_sq)

    # Define parametric structures
    
    def ring_structure(params):
        """
        params: [r1, r2, offset1, offset2, cx, cy]
        r1, r2: radii
        offset1, offset2: angular rotation of the rings
        cx, cy: center
        Structure: 8 points on ring 1, 8 points on ring 2.
        """
        r1, r2, off1, off2, cx, cy = params
        n1 = 8
        n2 = 8
        
        angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False) + off1
        angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + off2
        
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        points = np.vstack([pts1, pts2])
        points[:, 0] += cx
        points[:, 1] += cy
        
        return points

    def two_ring_with_center(params):
        """
        params: [r1, r2, off1, off2, cx, cy]
        1 point at center, 7 on ring 1, 8 on ring 2? 
        Or 1 center, 8 ring 1, 7 ring 2?
        Let's try 1 center, 8 ring 1, 7 ring 2.
        """
        r1, r2, off1, off2, cx, cy = params
        n1 = 8
        n2 = 7
        
        center = np.array([[0, 0]])
        
        angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False) + off1
        angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + off2
        
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        points = np.vstack([center, pts1, pts2])
        points[:, 0] += cx
        points[:, 1] += cy
        
        return points

    def grid_deformation(params):
        """
        4x4 grid with deformable positions.
        params: 16*2 = 32 coordinates.
        This is essentially general optimization, but starting from a grid.
        """
        return params.reshape(16, 2)

    def random_structure(params):
        """
        General 16x2 coordinates.
        """
        return params.reshape(16, 2)

    best_points = None
    best_score = -np.inf

    # Initial guesses for parametric structures
    # Structure 1: Two rings (8+8)
    # Typical packing might have inner ring smaller.
    init_rings = np.array([0.5, 1.0, 0.0, np.pi/8, 0.0, 0.0])
    
    # Structure 2: Center + 2 rings (1+8+7)
    init_center = np.array([0.5, 1.0, 0.0, 0.0, 0.0, 0.0])
    
    # Structure 3: Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    init_grid = np.column_stack((grid_x.ravel(), grid_y.ravel())).flatten()
    
    # Structure 4: Random
    np.random.seed(42)
    init_random = np.random.randn(32) * 0.5

    # Optimization Settings
    # Use L-BFGS-B for bounded optimization if needed, but radii can be negative? 
    # Radii should be positive. Angles can be anything.
    # We'll just use BFGS or L-BFGS-B with bounds if we define them.
    # For simplicity and speed, let's use L-BFGS-B with no bounds first, 
    # but ensure radii are positive by taking abs in the structure function?
    # No, better to optimize signed radii and take abs in structure, or just 
    # let the optimizer find positive values.
    
    # Let's run optimization for each structure
    
    structures = [
        ("rings_8_8", init_rings, ring_structure),
        ("center_1_8_7", init_center, two_ring_with_center),
        ("grid", init_grid, grid_deformation),
        ("random", init_random, random_structure)
    ]
    
    for name, init_params, struct_func in structures:
        if time.time() > deadline - 5:
            break
            
        try:
            # Objective function wrapper
            def objective(params):
                points = struct_func(params)
                return smooth_score(points, temperature=0.05)
            
            # Optimize
            res = minimize(
                objective, 
                init_params, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e5: # Check if it didn't fail catastrophically
                opt_params = res.x
                points = struct_func(opt_params)
                
                # Evaluate with the EXACT score (not smoothed)
                exact_score = score(points)
                
                if exact_score > best_score:
                    best_score = exact_score
                    best_points = points.copy()
                    
        except Exception as e:
            print(f"Optimization failed for {name}: {e}")
            continue

    # If no parametric structure worked well, fallback to a quick SA or just return best found
    if best_points is None:
        # Fallback to a simple random search or grid
        best_points = np.column_stack((
            np.linspace(-1, 1, 16), 
            np.linspace(-1, 1, 16)
        ))
        best_score = score(best_points)

    # Final Polish: Use Nelder-Mead on the best point found to refine the exact (non-smooth) objective
    # This helps recover from any smoothing artifacts.
    if time.time() < deadline - 10:
        try:
            # We need a function that takes a flat array and returns -score
            def exact_neg_score(flat_pts):
                pts = flat_pts.reshape(16, 2)
                return -score(pts)
            
            res_nm = minimize(
                exact_neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if res_nm.fun < -best_score: # -score improved
                final_points = res_nm.x.reshape(16, 2)
                final_score = -res_nm.fun
                if final_score > best_score:
                    best_points = final_points
                    best_score = final_score
        except Exception:
            pass

    return best_points

# Self-test / Verification
if __name__ == "__main__":
    pts = min_max_dist_dim2_16()
    print(pts)
    dists = pdist(pts, 'sqeuclidean')
    d_min = np.sqrt(np.min(dists))
    d_max = np.sqrt(np.max(dists))
    print(f"Score: {(d_min/d_max)**2}")
