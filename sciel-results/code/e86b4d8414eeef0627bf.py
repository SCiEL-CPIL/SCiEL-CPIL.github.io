"""
Optimized packing of 50 equal circles in a unit circle.
Strategy:
1. Generate multiple diverse initial configurations.
2. Coarse global search using Differential Evolution on a soft-min objective.
3. Local refinement using L-BFGS-B on soft-min objective.
4. Final hard optimization using L-BFGS-B on the exact minimum distance objective.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 50
# Known best score is approx 0.1258254895304
# We aim to beat 1.0 (which corresponds to the seed program's score).
# The seed program returns a hexagonal packing.

def score(points):
    """
    Calculate the score: max radius r such that circles are disjoint and inside container.
    r = min(min_pair_dist / 2, min(1 - |p_i|))
    """
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    
    # Check container constraint
    dists_to_origin = np.linalg.norm(points, axis=1)
    if np.any(dists_to_origin > 1.0):
        # If any point is outside, the radius is limited by how far inside it can be?
        # The task says "A centre on or outside the container ... makes the candidate invalid".
        # However, the score function computes: min_i (1 - |c_i|).
        # If |c_i| > 1, this term is negative. The min would be negative.
        # We should return a very small number or 0 if invalid?
        # The prompt says: "Score: the largest common radius r = min(...) ... computed in double precision".
        # If the configuration is invalid (center outside), the radius is effectively negative or undefined.
        # Let's assume the evaluator handles invalids by scoring 0 or negative.
        # To be safe, if any point is outside, return a small negative value.
        pass 

    # Calculate min distance to boundary
    min_boundary_dist = 1.0 - np.max(dists_to_origin)
    
    # Calculate min pair distance
    # Vectorized calculation of pairwise distances
    # diff[i, j] = points[i] - points[j]
    # We only need upper triangle to avoid duplicates and self-distances
    # For N=50, N^2 is small, so we can do it directly.
    
    # dist_matrix[i, j] = ||p_i - p_j||
    # Using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def soft_min_objective(points_flat, beta):
    """
    Objective function for optimization.
    We want to MAXIMIZE r.
    So we MINIMIZE -r.
    
    To make it differentiable, we use a soft-min approximation for the minimum distance.
    min(x_i) ≈ -1/beta * log(sum(exp(-beta * x_i)))
    
    Here x_i are the constraints:
    1. Pairwise distances / 2
    2. 1 - |p_i|
    
    We want to maximize the minimum of these constraints.
    Let C be the vector of all constraint values (pair dists/2 and boundary dists).
    We want max(min(C)).
    Approximation: max(-1/beta * log(sum(exp(-beta * C))))
    Minimize: 1/beta * log(sum(exp(-beta * C)))
    """
    points = points_flat.reshape(-1, 2)
    
    # Calculate pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Extract upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Calculate boundary distances
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    
    # Combine constraints
    # We want the minimum of (pair_dists / 2) and (boundary_dists)
    constraints = np.concatenate([pair_dists / 2.0, boundary_dists])
    
    # Soft min
    # To avoid overflow in exp, we subtract the max (which is the most negative term in -beta*C)
    # Actually, we are computing log(sum(exp(-beta * C))).
    # Let M = max(-beta * C) = -beta * min(C).
    # log(sum(exp(-beta * C))) = M + log(sum(exp(-beta * C - M)))
    
    # If beta is large, -beta * C can be very large positive for small C.
    # We want to minimize the "worst" constraint.
    # Let's just use the standard stable log-sum-exp.
    
    # If any constraint is very negative, the score is very negative.
    # The objective is to minimize the "soft minimum" penalty.
    
    # Stable log-sum-exp for sum(exp(-beta * C))
    # Let Z = -beta * C
    # max_Z = np.max(Z)
    # log_sum_exp = max_Z + np.log(np.sum(np.exp(Z - max_Z)))
    
    # We want to MINIMIZE this value?
    # If C is large (good), -beta*C is very negative. exp is small. log is negative.
    # If C is small (bad), -beta*C is less negative (or positive). exp is larger. log is larger.
    # So minimizing log_sum_exp corresponds to maximizing the minimum C.
    
    # However, if points overlap, C is negative.
    # Let's add a large penalty if points are outside or overlapping significantly?
    # The soft-min handles it naturally if beta is high enough.
    
    # One issue: if points are far apart, C is large. -beta*C is huge negative.
    # exp(huge negative) is 0. log(0) is -inf.
    # We want the objective to be finite.
    
    # Let's use a simpler approach:
    # Objective = -soft_min(C)
    # soft_min(C) = -1/beta * log(sum(exp(-beta * C)))
    # So Objective = 1/beta * log(sum(exp(-beta * C)))
    
    # To be numerically stable:
    Z = -beta * constraints
    # If Z has very large positive values (meaning C is very negative, i.e., bad),
    # exp(Z) overflows.
    # If Z has very large negative values (meaning C is very positive, i.e., good),
    # exp(Z) underflows to 0.
    
    # We want to minimize the "badness".
    # If all C are positive and large, Z is large negative. sum(exp(Z)) ~ 0. log ~ -inf.
    # This is a problem.
    
    # Alternative:
    # We want to maximize min(C).
    # Let's shift C so that min(C) is around 0? No, we don't know min(C).
    
    # Let's use the property that we are optimizing.
    # If we start with a valid packing, C is positive.
    # If we start with overlapping, C is negative.
    
    # Let's just clip Z to prevent overflow.
    # But if we clip, we lose gradient info for very bad configurations.
    
    # Better approach:
    # Use the fact that we only care about the minimum.
    # If we use a high beta, the soft-min approximates the hard-min well.
    
    # Let's try:
    # max_Z = np.max(Z)
    # if max_Z > 700: return 1e10 # Overflow protection, very bad config
    # log_sum = max_Z + np.log(np.sum(np.exp(Z - max_Z)))
    # obj = log_sum / beta
    
    # If max_Z is very negative (good config), exp(Z-max_Z) is exp(negative - negative) = exp(small).
    # This works.
    
    max_Z = np.max(Z)
    if max_Z > 700:
        return 1e10
        
    log_sum = max_Z + np.log(np.sum(np.exp(Z - max_Z)))
    obj = log_sum / beta
    
    return obj

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Initializations
    inits = []
    
    # 1. Perturbed Hexagonal Lattice
    # Create a grid of points
    # Approx radius 0.125 -> diameter 0.25
    # Hexagonal spacing approx 0.25
    spacing = 0.26
    points_hex = []
    # Generate points in a square area large enough
    # Unit circle radius 1.
    # x, y from -1.2 to 1.2
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * spacing + (0.5 * spacing if j % 2 != 0 else 0)
            y = j * spacing * np.sqrt(3) / 2
            if x*x + y*y < 1.0:
                points_hex.append([x, y])
    
    # We have more than 50 points.
    # Select 50.
    # To make it diverse, we can shuffle and take first 50, or sort by radius?
    # Sorting by radius puts center first.
    # Let's just take the first 50 from a shuffled list to break symmetry?
    # Or better: generate multiple hex grids with different offsets.
    
    # Init 1: Centered Hex
    rng = np.random.default_rng(42)
    pts = np.array(points_hex)
    # Shuffle
    idx = rng.permutation(len(pts))
    pts = pts[idx]
    # Keep only those inside unit circle with margin
    mask = np.linalg.norm(pts, axis=1) < 0.95
    pts = pts[mask]
    if len(pts) >= N:
        inits.append(pts[:N].copy())
        
    # 2. Random Initialization
    # Uniform in disk
    r = np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    pts_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    inits.append(pts_rand)
    
    # 3. Ring Initialization
    # 1 in center, 6 in ring 1, 12 in ring 2, 12 in ring 3, 9 in ring 4?
    # 1+6+12+12+9 = 40. Need 10 more.
    # 1+6+12+12+12+7 = 50.
    # Radii: 0, 0.2, 0.4, 0.6, 0.8, 0.95
    # This is a heuristic.
    rings = [1, 6, 12, 12, 12, 7]
    radii = [0.0, 0.18, 0.36, 0.54, 0.72, 0.9]
    pts_ring = []
    for count, rad in zip(rings, radii):
        if count == 0:
            continue
        if count == 1:
            pts_ring.append([0, 0])
        else:
            thetas = np.linspace(0, 2*np.pi, count, endpoint=False)
            # Add small random jitter
            thetas += rng.random(count) * 0.1
            pts = np.column_stack([rad * np.cos(thetas), rad * np.sin(thetas)])
            pts_ring.append(pts)
    pts_ring = np.vstack(pts_ring)
    inits.append(pts_ring)
    
    best_score = -1.0
    best_points = None
    
    # Process each initialization
    for i_init, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Phase 1: Differential Evolution
        # Bounds: x, y in [-1, 1]
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        # DE parameters
        # maxiter and popsize are tricky.
        # Let's limit the time spent here.
        
        try:
            # Use a wrapper to check time
            # DE doesn't have a callback for time, so we must be careful.
            # Small population and iterations for DE to keep it fast.
            # N=50 -> 100 variables.
            # popsize=10, maxiter=50 is small.
            
            # Objective for DE: soft_min with beta=10
            def obj_de(x):
                return soft_min_objective(x, beta=10.0)
                
            # Run DE
            # We might not finish in time if we run full DE.
            # Let's try a small DE run.
            # If time is tight, skip DE and go straight to L-BFGS?
            # L-BFGS is local. DE is global.
            # Given the 120s limit and 3 inits, we have ~35s per init.
            # DE with 100 vars, pop 20, iter 50 -> 1000 function evals.
            # Each eval is O(N^2) = 2500 ops. 2.5M ops. Very fast.
            
            res_de = differential_evolution(
                obj_de, 
                bounds, 
                init=init_pts.flatten(), 
                maxiter=100, 
                popsize=15, 
                tol=0.01, 
                seed=42,
                polish=False # We will polish with L-BFGS
            )
            
            current_pts = res_de.x.reshape(-1, 2)
            current_score = score(current_pts)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
                
        except Exception as e:
            pass
            
        if time.time() > deadline:
            break
            
        # Phase 2: L-BFGS-B Refinement (Soft Min)
        # Start from DE result
        start_pts = current_pts.flatten() if 'current_pts' in locals() else init_pts.flatten()
        
        def obj_lfgs_soft(x):
            return soft_min_objective(x, beta=50.0)
            
        try:
            res_lfgs = minimize(
                obj_lfgs_soft, 
                start_pts, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'gtol': 1e-5}
            )
            
            current_pts = res_lfgs.x.reshape(-1, 2)
            current_score = score(current_pts)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
                
        except Exception as e:
            pass
            
        if time.time() > deadline:
            break
            
        # Phase 3: Hard Optimization (L-BFGS-B)
        # Maximize min_dist.
        # Minimize -min_dist.
        # This is non-smooth, but L-BFGS-B can often handle it if started close.
        
        def obj_hard(x):
            pts = x.reshape(-1, 2)
            s = score(pts)
            # Add penalty if invalid (outside)
            if s < 0:
                return 1000.0 + s # Minimize large positive
            return -s
            
        try:
            res_hard = minimize(
                obj_hard, 
                current_pts.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            
            current_pts = res_hard.x.reshape(-1, 2)
            current_score = score(current_pts)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
                
        except Exception as e:
            pass
            
    # Fallback: If no valid points found, return the hex init
    if best_points is None:
        if inits:
            best_points = inits[0]
        else:
            # Generate a simple valid packing
            # 1 center, 6 around, etc.
            best_points = np.zeros((N, 2))
            # This might be invalid if they overlap.
            # Let's use the hex init from earlier if available.
            if 'pts' in locals() and len(pts) >= N:
                best_points = pts[:N]
            else:
                # Grid
                best_points = np.array([[i*0.2, j*0.2] for i in range(-2,3) for j in range(-2,3)])[:N]
                
    return best_points

# Ensure N is defined
N = 50

if __name__ == "__main__":
    # Simple test
    pts = construct()
    print(f"Score: {score(pts)}")
    print(f"Shape: {pts.shape}")
    print(f"Valid: {np.all(np.isfinite(pts)) and np.all(np.linalg.norm(pts, axis=1) <= 1.0)}")
