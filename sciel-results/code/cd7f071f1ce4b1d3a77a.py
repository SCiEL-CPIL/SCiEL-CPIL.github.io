import numpy as np
import time
import random
from scipy.optimize import minimize
import math

N = 12
SQRT3 = math.sqrt(3.0)
TIME_LIMIT = 145.0

# Precompute base vertices for a unit hexagon (side 1)
BASE_ANGLES = np.radians(np.arange(0, 360, 60))
BASE_COS = np.cos(BASE_ANGLES)
BASE_SIN = np.sin(BASE_ANGLES)
BASE_VERTS = np.column_stack([BASE_COS, BASE_SIN])

def get_vertices(cx, cy, angle_rad):
    """Returns (6, 2) vertices for a unit hexagon."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (R @ BASE_VERTS.T).T + np.array([cx, cy])

def compute_outer_side_length(centers, angles):
    """
    Compute the minimum side length of a regular hexagon (centered at origin, 
    flat-top orientation) that contains all 12 unit hexagons.
    centers: (12, 2) array
    angles: (12,) array of rotation angles in radians
    """
    all_verts = np.zeros((N * 6, 2))
    for i in range(N):
        all_verts[i * 6:(i + 1) * 6] = get_vertices(centers[i, 0], centers[i, 1], angles[i])
        
    x_coords = all_verts[:, 0]
    y_coords = all_verts[:, 1]
    
    # For a regular hexagon with flat-top orientation (vertices at 0, 60, ... degrees)
    # The support function in direction theta is:
    # If theta is in [-30, 30]: s * cos(theta) + x * sin(30) + y * cos(30) ??
    # Let's use the half-width in the 6 normal directions.
    # Normals are at 30, 90, 150, 210, 270, 330 degrees.
    # h(theta) = max(x * cos(theta) + y * sin(theta))
    # Side length s = max(h(theta)) / cos(30) = max(h(theta)) / (SQRT3/2)
    
    normal_angles = np.radians(np.arange(30, 360, 60))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # all_verts: (72, 2), normals: (6, 2)
    # projections: (72, 6)
    projections = all_verts @ normals.T
    max_proj = np.max(projections, axis=0) # (6,)
    
    return np.max(max_proj) / (SQRT3 / 2.0)

def check_validity(centers, angles, eps=1e-8):
    """
    Check if the configuration of 12 hexagons is valid (disjoint).
    Uses SAT (Separating Axis Theorem).
    Returns True if valid, False otherwise.
    """
    for i in range(N):
        v1 = get_vertices(centers[i, 0], centers[i, 1], angles[i])
        for j in range(i + 1, N):
            # Quick center distance check
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq > 25.0: # (2 * 1.5)^2 = 9, but max dist is 4. Use 25 for safety.
                continue
            
            v2 = get_vertices(centers[j, 0], centers[j, 1], angles[j])
            
            # SAT Check
            # Axes are normals to edges of v1 and v2
            axes = []
            for k in range(6):
                e1 = v1[(k + 1) % 6] - v1[k]
                n1 = np.array([-e1[1], e1[0]])
                norm = np.linalg.norm(n1)
                if norm > 1e-10:
                    axes.append(n1 / norm)
                
                e2 = v2[(k + 1) % 6] - v2[k]
                n2 = np.array([-e2[1], e2[0]])
                norm = np.linalg.norm(n2)
                if norm > 1e-10:
                    axes.append(n2 / norm)
            
            # Check each axis
            found_gap = False
            for n in axes:
                p1 = v1.dot(n)
                p2 = v2.dot(n)
                
                min1, max1 = np.min(p1), np.max(p1)
                min2, max2 = np.min(p2), np.max(p2)
                
                # If there is a gap on any axis, they are disjoint
                if max1 + eps < min2 or max2 + eps < min1:
                    found_gap = True
                    break
            
            if not found_gap:
                # Check for strict overlap? 
                # If no gap found, they might be touching or overlapping.
                # For "disjoint", we usually allow touching.
                # However, if they overlap, there is no gap on ANY axis.
                # So if found_gap is False, it means for ALL axes, projections overlap.
                # This implies intersection (overlap or touch).
                # To be safe for "disjoint" requirement, we should probably treat "touch" as valid 
                # but "overlap" as invalid.
                # SAT says: Disjoint iff there exists a separating axis.
                # So if no separating axis, they are NOT disjoint (they intersect or touch).
                # Wait, "disjoint" usually means interiors are disjoint. Touching is allowed.
                # SAT: If they touch, there is an axis where max1 == min2. 
                # My check `max1 + eps < min2` allows touching (gap >= -eps).
                # If they overlap, max1 > min2 and max2 > min1 for all axes.
                # So `max1 + eps < min2` will be False.
                # Thus, if found_gap is False, they overlap (or touch exactly, which is borderline).
                # Given the problem says "disjoint", and typically packing allows touching,
                # we should be lenient with touching but strict with overlap.
                # However, distinguishing touch from overlap with floating point is tricky.
                # Let's assume if no gap > eps, it's invalid (overlapping).
                # This is a conservative check.
                return False
                
    return True

def objective(x, softness=1.0):
    """
    x: (12, 3) array of [cx, cy, angle]
    Returns: side_length + penalty
    """
    centers = x[:, :2]
    angles = x[:, 2]
    
    # Compute side length
    side = compute_outer_side_length(centers, angles)
    
    # Compute penalty using smooth approximation of SAT violation
    penalty = 0.0
    for i in range(N):
        v1 = get_vertices(centers[i, 0], centers[i, 1], angles[i])
        for j in range(i + 1, N):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq > 25.0:
                continue
            
            v2 = get_vertices(centers[j, 0], centers[j, 1], angles[j])
            
            # Compute minimum gap on all axes
            # Gap = min over axes of (separation)
            # Separation = max(0, min2 - max1, min1 - max2)
            # If separation < 0, they overlap.
            # We want a penalty for overlap.
            
            min_separation = np.inf
            for k in range(6):
                e1 = v1[(k + 1) % 6] - v1[k]
                n1 = np.array([-e1[1], e1[0]])
                norm = np.linalg.norm(n1)
                if norm > 1e-10:
                    n1 /= norm
                    p1 = v1.dot(n1)
                    p2 = v2.dot(n1)
                    sep = max(0, np.min(p2) - np.max(p1), np.min(p1) - np.max(p2))
                    if sep < min_separation:
                        min_separation = sep
                        
                e2 = v2[(k + 1) % 6] - v2[k]
                n2 = np.array([-e2[1], e2[0]])
                norm = np.linalg.norm(n2)
                if norm > 1e-10:
                    n2 /= norm
                    p1 = v1.dot(n2)
                    p2 = v2.dot(n2)
                    sep = max(0, np.min(p2) - np.max(p1), np.min(p1) - np.max(p2))
                    if sep < min_separation:
                        min_separation = sep
            
            # min_separation is 0 if they touch or overlap.
            # We need to detect overlap depth.
            # Actually, the SAT distance is the minimum over axes of the gap.
            # If they overlap, the gap is negative? No, gap is defined as distance between intervals.
            # If intervals overlap, gap is 0.
            # So min_separation will be 0 for both touch and overlap.
            # This makes it impossible to distinguish touch from overlap using just SAT gap.
            
            # To penalize overlap, we can check if centers are too close or use a different metric.
            # A common trick: If min_separation == 0, check if a vertex is inside the other.
            # But that's discrete.
            
            # Alternative: Use a "soft" constraint.
            # If we assume the optimal solution has hexagons touching, 
            # and we want to avoid overlap, we can use a penalty based on 
            # the "depth" of intersection.
            
            # Let's use the following: 
            # If min_separation < eps, consider it a violation.
            # Penalty += (eps - min_separation)^2 if min_separation < eps else 0?
            # No, if they overlap, min_separation is 0. 
            # This will penalize touching too.
            
            # Let's refine the SAT to get actual penetration depth? 
            # Too complex for this step.
            
            # Let's stick to a simpler heuristic:
            # If min_separation < 1e-6, add a penalty.
            # This pushes them apart slightly.
            if min_separation < 1e-6:
                penalty += 1.0
            
    return side + 10.0 * penalty

def construct():
    start_time = time.time()
    
    # Initial Configuration: Honeycomb
    # 1 center, 6 ring 1, 5 ring 2 (to make 12)
    # Or just a 4x3 grid distorted.
    
    # Let's use a known good starting point: 
    # 12 hexagons in a hexagonal cluster.
    # Positions in axial coordinates (q, r)
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2), (-2,1) -> This is 1+6+5=12
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_x = np.zeros((N, 3))
    for i, (q, r) in enumerate(axial_coords):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        init_x[i, 0] = x
        init_x[i, 1] = y
        init_x[i, 2] = 0.0
        
    # Ensure initial validity
    if not check_validity(init_x[:, :2], init_x[:, 2]):
        # Scale up
        scale = 1.1
        while not check_validity(init_x[:, :2] * scale, init_x[:, 2]) and scale < 5.0:
            scale += 0.1
        init_x[:, :2] *= scale
        
    best_x = init_x.copy()
    best_val = objective(best_x)
    
    # Multi-start Nelder-Mead
    # We have 12 * 3 = 36 parameters.
    # Nelder-Mead is slow for high dimensions.
    # Let's try to reduce dimensionality or use a different method.
    # Since the directive says "Centroid-Minimization with Radial Expansion" but also "Keep this design",
    # and the previous program used Nelder-Mead on 36 params, let's try to speed it up.
    
    # Optimization:
    # 1. Use a lower dimensionality search if possible? 
    #    The inspirations suggest symmetry constraints.
    #    If we assume 6-fold symmetry is NOT preserved (since 12 is not 6*k+1), 
    #    we might have a 2-fold or 3-fold symmetry?
    #    Actually, a 12-hexagon packing might not have high symmetry.
    
    # Let's stick to full 36D but optimize the function evaluation.
    # The current `objective` is a bit slow due to Python loops.
    # We can vectorize the SAT check?
    # Or just run fewer iterations.
    
    # Let's try a few restarts with Nelder-Mead.
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        # Perturb initial configuration
        if r == 0:
            x0 = best_x.copy()
        else:
            x0 = best_x + np.random.normal(0, 0.2, (N, 3))
            # Ensure angles are reasonable
            x0[:, 2] = x0[:, 2] % (2 * np.pi)
            
        # Run Nelder-Mead
        # Flatten x0
        x0_flat = x0.flatten()
        
        def obj_flat(x_flat):
            x_2d = x_flat.reshape((N, 3))
            # Normalize angles
            x_2d[:, 2] = x_2d[:, 2] % (2 * np.pi)
            return objective(x_2d)
            
        try:
            res = minimize(obj_flat, x0_flat, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.reshape((N, 3))
                best_x[:, 2] = best_x[:, 2] % (2 * np.pi)
                
        except Exception as e:
            pass
            
    # Final polish:
    # Run a tighter optimization on the best solution found
    if time.time() - start_time < TIME_LIMIT - 10:
        x0_flat = best_x.flatten()
        
        def obj_flat_tight(x_flat):
            x_2d = x_flat.reshape((N, 3))
            x_2d[:, 2] = x_2d[:, 2] % (2 * np.pi)
            return objective(x_2d)
            
        try:
            res = minimize(obj_flat_tight, x0_flat, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.reshape((N, 3))
                best_x[:, 2] = best_x[:, 2] % (2 * np.pi)
        except:
            pass
            
    # Extract result
    centers = best_x[:, :2]
    angles = best_x[:, 2]
    
    inner = []
    for i in range(N):
        x = float(centers[i, 0])
        y = float(centers[i, 1])
        ang_deg = float(np.degrees(angles[i])) % 360.0
        inner.append([x, y, ang_deg])
        
    # Compute final side length
    side_len = compute_outer_side_length(centers, angles)
    
    # Return
    return inner, (0.0, 0.0), float(side_len), 0.0

# EVOLVE-BLOCK-END
