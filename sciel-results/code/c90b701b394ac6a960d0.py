"""
Strategy: Active-Set SLSQP Optimization for Circle Packing.

Unlike the seed program which minimizes -min_distance (a non-smooth objective)
using L-BFGS-B, this program explicitly maximizes the radius 'r' subject to
hard inequality constraints.

1. Initialization: Generate a hexagonal packing close to the expected optimal.
2. Optimization Loop:
   a. Compute current radius r and identify the "active set" of constraints:
      - Pairs of circles at distance ~2r.
      - Circles at distance ~r from walls.
   b. Use SLSQP to maximize r subject to:
      - ||c_i - c_j||^2 >= (2r)^2 for all i<j (or at least active ones, but SLSQP 
        handles all constraints if provided efficiently. For N=50, 1225 pairs is 
        manageable for SLSQP if we are careful, but usually we only need the 
        "tight" ones. However, SLSQP requires a constraint function. 
        Defining 1225 constraints might be slow. 
        Alternative: Define the objective as -r and constraints as 
        ||c_i - c_j||^2 - 4r^2 >= 0 and wall distances - r >= 0.
        Actually, SLSQP can handle this. Let's try defining the constraints 
        for ALL pairs to be safe, or just the active ones to speed up.
        Given N=50, 1225 constraints is a lot for SLSQP per iteration.
        
        Let's refine: 
        We maximize r.
        Variables: x = [c_1...c_50, r] (101 variables).
        Objective: -r (minimize -r to maximize r).
        Constraints:
           1. Wall constraints: 0.5 - |c_i,k| - r >= 0 for all i, k. (100 constraints)
           2. Pair constraints: ||c_i - c_j||^2 - (2r)^2 >= 0 for all i<j. (1225 constraints)
           
        Total ~1325 constraints. SLSQP might be slow with this many.
        
        Better Strategy: 
        Use a "soft" constraint approach or just optimize the positions for a FIXED r?
        No, we want to find the max r.
        
        Let's stick to the "Active Set" idea but implemented carefully.
        In each step:
        1. Find current r.
        2. Identify pairs with dist < 2r + epsilon and walls with dist < r + epsilon.
        3. Define a smaller set of constraints (only the active ones + a safety margin).
        4. Run SLSQP to increase r slightly while satisfying these constraints.
        
        Actually, a simpler and very robust method for this specific problem (N=50) 
        is to use **Nelder-Mead** on the positions for a fixed target radius, 
        and binary search on the radius. But Nelder-Mead is slow for 100 dimensions.
        
        Let's try the **SLSQP with all constraints** but with a tight budget. 
        Or, use **L-BFGS-B** but with a better objective: 
        Minimize sum of squared violations for a target radius R, and increase R.
        
        Let's go with a hybrid:
        1. Start with hex init.
        2. Compute r.
        3. Try to increase r by solving a quadratic program or using SLSQP 
           with constraints ONLY on the currently "tight" constraints (dist < 2r + tol).
           This keeps the number of constraints small (typically < 50).
"""

import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 50
HALF = 0.5
DEADLINE = 115.0  # Safety margin for 120s limit (Evaluator kills at 150s, but task says 120s limit usually, let's be safe)
# Actually, the prompt says "Each evaluation must finish within 120 s." and "KILLS ... after 150s".
# I will use 115s as deadline to be safe.

def _compute_radius(centers):
    """Compute the valid radius for a given set of centers."""
    if centers.shape != (N, 2):
        return 0.0
    if np.any(np.abs(centers) > HALF + 1e-12):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists) / 2.0
    
    # Wall distances
    wall_dists = np.min(np.abs(centers), axis=1)
    min_wall_dist = np.min(HALF - wall_dists)
    
    r = min(min_dist, min_wall_dist)
    return max(0.0, r)

def _hex_init():
    """
    Generate a high-quality hexagonal initialization.
    We generate a hex grid slightly larger than needed, then select the best N points
    or scale to fit.
    """
    # Target radius approx 0.0714
    # Hex grid spacing: dx = 2r, dy = r*sqrt(3)
    # Let's generate a grid of points and then optimize.
    
    # Create a dense hex grid
    r_guess = 0.075
    dx = 2 * r_guess
    dy = r_guess * np.sqrt(3)
    
    points = []
    # Cover a range larger than the square
    range_val = 1.0
    nx = int(range_val / dx) + 2
    ny = int(range_val / dy) + 2
    
    for i in range(-ny, ny):
        for j in range(-nx, nx):
            x = j * dx
            y = i * dy
            if i % 2 != 0:
                x += dx / 2
            points.append([x, y])
            
    points = np.array(points)
    
    # Center the grid
    points -= np.mean(points, axis=0)
    
    # Select the N points closest to the center? 
    # Or select a subset that maximizes min distance?
    # Simple heuristic: Sort by distance to origin, take first N? 
    # This creates a "blob" which is good for packing.
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:N]
    points = points[indices]
    
    # Scale to fit inside the square with a small margin
    # We want max(|x|, |y|) + r <= 0.5
    # Current max extent
    max_ext = np.max(np.abs(points))
    # We don't know the final r yet, but we can scale so it fits comfortably.
    # If we scale too much, we lose potential.
    # Let's just return the points. The optimizer will handle the rest.
    # But they must be inside [-0.5, 0.5].
    # If max_ext > 0.5 - r_guess, we should scale down.
    target_max = HALF - r_guess
    if max_ext > target_max:
        scale = target_max / max_ext
        points *= scale
        
    return points

def _objective_and_constraints(x, active_pairs, active_walls):
    """
    Objective and constraints for SLSQP.
    x: [c_1..c_N, r] (length 2N+1)
    """
    centers = x[:2*N].reshape(N, 2)
    r = x[-1]
    
    # Objective: maximize r => minimize -r
    obj = -r
    
    # Constraints:
    # 1. Wall constraints for active walls
    cons = []
    for i in active_walls:
        # 0.5 - |c_i,k| - r >= 0
        # We need to handle |x| and |y| separately or use min?
        # SLSQP works with scalar constraints.
        # dist_to_wall = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
        # This is non-smooth.
        # Better to define 4 constraints per wall if needed, or just the tightest one.
        # For simplicity in the active set, we only include the tightest wall constraint.
        cx, cy = centers[i]
        # Determine which wall is closest
        if abs(cx) > abs(cy):
            if cx > 0:
                dist = HALF - cx
            else:
                dist = HALF + cx
        else:
            if cy > 0:
                dist = HALF - cy
            else:
                dist = HALF + cy
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i, d=dist: d - x[-1], # Wait, dist depends on x.
        })
        # This lambda approach is tricky because 'd' is fixed at definition time.
        # We need to recompute the active set inside the loop or define the function dynamically.
        
    # This approach of fixing active set outside is problematic if the active set changes during optimization.
    # Better: Define the constraint function to check ALL constraints but only return the "violated" or "near-violated" ones?
    # No, SLSQP needs a consistent number of constraints or a list of dicts.
    
    # Let's use a different formulation for SLSQP:
    # Minimize -r
    # Subject to:
    # ||c_i - c_j||^2 - 4r^2 >= 0 for all i<j
    # 0.5 - |c_ik| - r >= 0 for all i,k
    
    # Since N=50, 1225 pair constraints + 100 wall constraints = 1325 constraints.
    # This is heavy for SLSQP.
    
    # Alternative: Use L-BFGS-B on the positions for a FIXED r, and binary search r.
    # This is often more robust.
    # Step 1: Guess r.
    # Step 2: Check if feasible. If yes, increase r. If no, decrease r.
    # Feasibility check: Minimize sum of squared violations (dist - 2r)^2 for pairs where dist < 2r, and (wall_dist - r)^2 for walls where wall_dist < r.
    # If min violation is ~0, it's feasible.
    
    # This "Penalty Method + Binary Search" is a strong strategy.
    
    return obj, cons # Not used in this version, see construct()

def _penalty_objective(centers, r_target):
    """
    Objective for a fixed target radius r_target.
    Minimize the sum of squared violations of the non-overlapping and boundary constraints.
    """
    # Pairwise violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # We only care about pairs where dist < 2*r_target
    # Violation = (2*r_target - dist)^2
    # To make it smooth and only penalize violations:
    # v = np.maximum(0, 2*r_target - dists)
    # penalty = np.sum(v**2)
    
    # However, computing this for all 1225 pairs every time is fast in numpy.
    v_pairs = np.maximum(0, 2 * r_target - dists)
    penalty_pairs = np.sum(v_pairs ** 2)
    
    # Wall violations
    # wall_dist = min(0.5 - |x|, 0.5 - |y|)
    # v_wall = np.maximum(0, r_target - wall_dist)
    
    # Vectorized wall distance
    # x: [N], y: [N]
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Distance to 4 walls
    d_left = HALF + x
    d_right = HALF - x
    d_bottom = HALF + y
    d_top = HALF - y
    
    # Min distance to any wall
    d_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    v_walls = np.maximum(0, r_target - d_wall)
    penalty_walls = np.sum(v_walls ** 2)
    
    return penalty_pairs + penalty_walls

def _is_feasible(centers, r_target, tol=1e-6):
    """Check if centers are feasible for r_target."""
    return _penalty_objective(centers, r_target) < tol

def _optimize_for_radius(r_target, x0, max_iter=500):
    """
    Optimize positions to satisfy radius r_target.
    Returns (success, centers).
    """
    bounds = [(-HALF, HALF)] * (2 * N)
    
    # Use L-BFGS-B as it handles bounds well and is fast for this smooth penalty.
    try:
        res = minimize(
            lambda c: _penalty_objective(c.reshape(N, 2), r_target),
            x0.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        new_centers = res.x.reshape(N, 2)
        # Clip to be safe
        new_centers = np.clip(new_centers, -HALF, HALF)
        
        # Check feasibility
        penalty = _penalty_objective(new_centers, r_target)
        if penalty < 1e-6:
            return True, new_centers
        else:
            return False, new_centers
    except Exception:
        return False, x0

def construct():
    start_time = time.time()
    
    # 1. Initialization
    centers = _hex_init()
    current_r = _compute_radius(centers)
    
    best_centers = centers.copy()
    best_r = current_r
    
    # 2. Binary Search on Radius with Optimization
    # We know r is between 0.07 and 0.08 approx.
    # Let's do a coarse search first, then refine.
    
    # Coarse Search
    r_low = 0.06
    r_high = 0.08
    
    # Ensure initial guess is feasible for r_low
    # If not, reduce r_low
    while not _is_feasible(centers, r_low):
        r_low *= 0.9
        if r_low < 0.01: break
        
    # Try to find a feasible r in [r_low, r_high]
    # We will perform a binary search.
    # For each candidate r, we try to optimize the positions.
    # If successful, r_low = r. If failed, r_high = r.
    
    # Number of binary search steps: ~20 steps gives precision 0.02 / 2^20 ~ 2e-8
    # But each step involves an optimization run.
    # Optimization run takes ~0.1-1s.
    # 20 steps * 1s = 20s. This is well within budget.
    # We can afford more steps or restarts.
    
    # Let's do 25 steps of binary search.
    num_steps = 25
    
    for i in range(num_steps):
        if time.time() > DEADLINE:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to optimize for r_mid
        # Use best_centers as starting point
        success, new_centers = _optimize_for_radius(r_mid, best_centers)
        
        if success:
            r_low = r_mid
            best_centers = new_centers
            best_r = r_mid
        else:
            r_high = r_mid
            
        # Update best_r if we found a better valid radius during optimization
        # (Sometimes the optimizer finds a valid config for r_mid even if penalty > tol,
        # but strictly speaking, if penalty > tol, it's not valid for r_mid.
        # However, _compute_radius gives the actual radius.
        # Let's compute the actual radius of new_centers and update best if better.
        actual_r = _compute_radius(new_centers)
        if actual_r > best_r:
            best_r = actual_r
            best_centers = new_centers
            
    # 3. Final Polish
    # After binary search, we have a good best_centers.
    # We can try to squeeze a bit more by optimizing for best_r + epsilon.
    # Or just run a few local optimizations on best_centers to maximize radius directly.
    
    # Direct Maximization Polish
    # Use L-BFGS-B to maximize -min_dist directly? 
    # Or use the penalty method with r = best_r + small_delta.
    
    if time.time() < DEADLINE - 10:
        # Try to improve by small steps
        current_r = best_r
        for _ in range(5):
            if time.time() > DEADLINE - 5:
                break
            next_r = current_r * 1.001
            success, new_centers = _optimize_for_radius(next_r, best_centers, max_iter=200)
            if success:
                current_r = next_r
                best_centers = new_centers
                best_r = current_r
            else:
                break
                
    # Final validation
    best_centers = np.clip(best_centers, -HALF, HALF)
    
    return best_centers

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We need to wrap it in the EVOLVE-BLOCK as per the instructions?
# "Keep the function names and signatures of the seed program."
# The seed program has `construct()`.
# The instructions say: "Output the COMPLETE program in a single
