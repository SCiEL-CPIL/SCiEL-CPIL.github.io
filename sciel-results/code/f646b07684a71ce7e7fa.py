import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

N = 13

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 distinct points.
    points: (N, 2) array
    """
    min_area = np.inf
    # C(13, 3) = 286 triangles. 
    # We can vectorize this for speed.
    # Create indices
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    # We only want i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return -1e10
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin
    
    # Strategy: Optimize angles of points on the unit circle.
    # Since the hull area for points on the unit circle is fixed (if they form a convex polygon),
    # maximizing min triangle area is equivalent to maximizing the score.
    # Actually, hull area varies slightly if we allow radial movement, but for a 13-gon,
    # the regular polygon is a strong candidate. Let's stick to angles first.
    
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # We will use coordinate descent on the angles.
    # Moving one angle affects triangles involving that point.
    
    best_score = -np.inf
    best_angles = angles.copy()
    
    # To speed up evaluation, we can precompute which triangles involve each point.
    # But with N=13, full vectorized evaluation is very fast (~microseconds).
    # Let's benchmark: 286 triangles * 13 points * iterations.
    
    # Helper to evaluate score given angles
    def eval_angles(ang):
        # Ensure angles are sorted? No, the order doesn't matter for the set of points,
        # but for the hull, the points should be in convex position.
        # If we just change angles, the points remain on the circle.
        # As long as no two angles are identical, they are in convex position.
        pts = np.column_stack((np.cos(ang), np.sin(ang)))
        return _score(pts)
    
    current_score = eval_angles(angles)
    best_score = current_score
    best_angles = angles.copy()
    
    # Coordinate Descent
    # Step size strategy: Start large, decay.
    # Or use a line search?
    # Let's try a simple adaptive step size.
    
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline - 5:
        improved = False
        
        for i in range(N):
            if time.time() > deadline - 5:
                break
                
            # Try moving angle[i] by +step and -step
            # Also try smaller steps if large ones fail?
            # For now, just +/- step.
            
            for sign in [1, -1]:
                new_angles = best_angles.copy()
                new_angles[i] += sign * step_size
                
                # Optional: sort angles to maintain "regularity" feel? 
                # Not strictly necessary for the math, but might help convergence if we think of them as ordered.
                # Actually, just permuting points doesn't change the set.
                
                s = eval_angles(new_angles)
                
                if s > best_score + 1e-10:
                    best_angles = new_angles
                    best_score = s
                    improved = True
                    no_improve_count = 0
                    # We can break the sign loop and move to next i, 
                    # or continue to see if the other sign is even better?
                    # Usually, if one direction improves, the other is worse.
                    # But let's check the other direction too just in case we are at a local max?
                    # No, if we moved in one direction and improved, we are now at a new point.
                    # The "other" direction from the OLD point is now the "same" direction from the NEW point?
                    # No.
                    # Let's just restart the loop for this index or continue.
                    # Restarting is safer for convergence.
                    break
            
            if improved:
                # If we improved, we might want to re-evaluate from this new state.
                # But we are updating best_angles in place.
                # The next iteration of the outer loop (next i) will use the updated best_angles.
                # This is standard coordinate descent.
                pass
        
        if not improved:
            no_improve_count += 1
            # Decay step size
            step_size *= 0.95
            if step_size < 1e-6 or no_improve_count > 1000:
                # Try a random perturbation to escape local minima
                if time.time() < deadline - 10:
                    rng = np.random.default_rng(int(time.time()))
                    perturb_idx = rng.integers(0, N)
                    best_angles[perturb_idx] += rng.uniform(-0.5, 0.5)
                    s = eval_angles(best_angles)
                    if s > best_score:
                        best_score = s
                    no_improve_count = 0
                    step_size = 0.1 # Reset step size
                else:
                    break
        else:
            # If improved, maybe keep step size or increase slightly?
            # Keep it constant for stability.
            pass
            
    # Final check: Maybe try a few random restarts with different initial angles?
    # The regular polygon is a very strong global optimum for this kind of problem.
    # But let's try one perturbed restart if we have time.
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            # Perturb regular polygon
            perturbed = angles + rng.uniform(-0.2, 0.2, N)
            # Quick local optimization (fewer iterations)
            curr_ang = perturbed.copy()
            curr_score = eval_angles(curr_ang)
            
            step = 0.05
            for _ in range(50): # Limit iterations for restart
                if time.time() > deadline - 20:
                    break
                improved = False
                for i in range(N):
                    for sign in [1, -1]:
                        new_ang = curr_ang.copy()
                        new_ang[i] += sign * step
                        s = eval_angles(new_ang)
                        if s > curr_score + 1e-10:
                            curr_ang = new_ang
                            curr_score = s
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.9
                    if step < 1e-5:
                        break
            
            if curr_score > best_score:
                best_score = curr_score
                best_angles = curr_ang

    # Convert best angles to points
    final_points = np.column_stack((np.cos(best_angles), np.sin(best_angles)))
    
    # Ensure the points are in a consistent order? 
    # The problem doesn't specify order, just the set.
    # But returning them sorted by angle is good practice.
    sort_idx = np.argsort(best_angles)
    final_points = final_points[sort_idx]
    
    return final_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
