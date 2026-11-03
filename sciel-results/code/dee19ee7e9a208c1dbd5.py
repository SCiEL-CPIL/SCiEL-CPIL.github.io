import numpy as np
import time
from itertools import combinations, product

N = 11

def _hex_vertices(cx, cy, angle_deg=0.0):
    """
    Get vertices of a unit regular hexagon centered at (cx, cy) rotated by angle_deg.
    """
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle_rad
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _min_outer_side(inner_centers, inner_angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length S of the outer hexagon.
    """
    theta_rad = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        alpha = theta_rad + np.deg2rad(30 + k * 60)
        normals.append((np.cos(alpha), np.sin(alpha)))
    
    max_proj = 0.0
    for (cx, cy), ang in zip(inner_centers, inner_angles):
        verts = _hex_vertices(cx, cy, ang)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
    
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Lattice basis
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5])
    
    # Base configuration: Center + 6 neighbors (Ring 1)
    # These are the 6 closest lattice points to origin
    base_centers = [
        (0.0, 0.0),
        (np.sqrt(3), 0.0),
        (np.sqrt(3)/2, 1.5),
        (-np.sqrt(3)/2, 1.5),
        (-np.sqrt(3), 0.0),
        (-np.sqrt(3)/2, -1.5),
        (np.sqrt(3)/2, -1.5)
    ]
    
    # Candidates for the remaining 4 hexagons
    # We consider Ring 2 (distance 3) and Ring 3 (distance 2*sqrt(3) ~ 3.46) and Ring 4?
    # Ring 2 points (i,j) such that max(|i|,|j|,|i+j|)=2
    # Actually distance is 3 * |iA + jB|.
    # Neighbors at dist 3: (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1) ... wait.
    # Let's list coordinates for Ring 2 (6 points):
    ring_2 = [
        (2*np.sqrt(3), 0.0),
        (np.sqrt(3), 3.0),
        (-np.sqrt(3), 3.0),
        (-2*np.sqrt(3), 0.0),
        (-np.sqrt(3), -3.0),
        (np.sqrt(3), -3.0)
    ]
    
    # Ring 3? Distance 3*sqrt(3) ~ 5.2? No.
    # Distance between (2,0) and (1,1) is 3.
    # Distance between (1,1) and (0,2) is 3.
    # So Ring 2 points form a hexagon of side 3.
    
    # We need to pick 4 out of 6 from Ring 2, or mix with Ring 1?
    # Ring 1 is already used.
    # Can we pick from Ring 3?
    # Ring 3 coords: (3,0), (2,1), (1,2), (0,3), (-1,2), (-2,1), (-3,0), (-2,-1), (-1,-2), (0,-3), (1,-2), (2,-1)
    # These are farther, likely worse for minimizing bounding box unless the shape is very non-compact.
    # The "flower" shape (1+6) is compact. Adding 4 more to make 11.
    # Best is likely to fill the gaps in the hexagonal cluster.
    # The cluster of 1+6 has a hexagonal shape.
    # The next layer (Ring 2) has 6 points.
    # If we pick 4, we break symmetry.
    # Let's try all combinations of 4 from Ring 2.
    
    candidates = []
    
    # Option 1: 4 from Ring 2
    for combo in combinations(range(6), 4):
        # Check disjointness among themselves
        valid = True
        for i in range(4):
            for j in range(i+1, 4):
                p1 = np.array(ring_2[combo[i]])
                p2 = np.array(ring_2[combo[j]])
                # Min distance is 3 (adjacent in ring 2). 3 > sqrt(3) ~ 1.732. So always disjoint.
                if np.linalg.norm(p1 - p2) < np.sqrt(3) - 1e-9:
                    valid = False
                    break
            if not valid: break
        if valid:
            candidates.append([ring_2[idx] for idx in combo])
            
    # Option 2: Mix Ring 2 and Ring 3?
    # Ring 3 points are farther. Probably not optimal.
    # But let's add a few Ring 3 points just in case, mixed with Ring 2.
    # Actually, let's just stick to Ring 2 first.
    
    # We will optimize the global rotation and translation of the entire set.
    # Translation doesn't change the size S if we center the bounding box, 
    # but the "center" of the outer hexagon is fixed at (0,0) in the return value?
    # No, the return value includes `center` which is the center of the outer hexagon.
    # We can choose the center of the outer hexagon to be the centroid of the inner set 
    # (or optimized). 
    # The evaluator measures the raw value S.
    # If we shift all inner hexagons by (dx, dy), the projection changes.
    # To minimize max projection, we should center the cluster.
    # For a fixed set of points, the minimal enclosing hexagon (with fixed orientation) 
    # is obtained by centering the points such that the max and min projections are balanced?
    # Actually, S = 2 * max_proj / sqrt(3).
    # The "max_proj" is the max of (v . n).
    # If we translate all points by t, proj becomes (v+t).n = v.n + t.n.
    # We want to minimize max_k (v_k.n_k + t.n_k).
    # This is a convex optimization problem for t for fixed orientation.
    # But we also have rotation.
    
    # Let's simplify: 
    # 1. Try all 15 combinations of 4 from Ring 2.
    # 2. For each, try rotations of the entire cluster (0, 15, 30... 345).
    # 3. For each rotation, compute the optimal translation t to center the cluster?
    #    Or just compute S assuming center is at (0,0)?
    #    If we assume center is (0,0), we are fixing the outer hexagon center at origin.
    #    The inner hexagons are placed at their coordinates.
    #    This might not be optimal if the cluster is not centered at origin.
    #    We should compute the optimal shift.
    #    Let's calculate the optimal shift t for a given set of vertices.
    #    We want to minimize max( (v_i + t).n_j ) over all vertices i and normals j.
    #    This is equivalent to finding the center of the minimal enclosing hexagon.
    #    We can just use the centroid as a good initial guess, or solve the LP.
    #    Given time, let's just try t = (0,0) and t = -centroid.
    #    Actually, for a hexagonally symmetric object, centroid is optimal.
    #    For asymmetric, maybe not.
    #    Let's compute the bounding box in the 6 directions and center it.
    #    For each normal n_k, find max_proj_k and min_proj_k of the unshifted set.
    #    The shift t should satisfy t.n_k = -(max_proj_k + min_proj_k) / 2 ?
    #    No, that's for each direction independently.
    #    We can't satisfy all 6 unless it's symmetric.
    #    Let's just try t = (0,0) and t = -centroid.
    
    best_S = float('inf')
    best_config = None
    
    for extra_centers in candidates:
        all_centers = base_centers + extra_centers
        all_centers_np = np.array(all_centers)
        
        # Centroid
        centroid = np.mean(all_centers_np, axis=0)
        
        # Try rotations
        for rot_deg in range(0, 360, 15):
            if time.time() > deadline - 5:
                break
                
            rot_rad = np.deg2rad(rot_deg)
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)
            
            R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            
            rotated_centers = all_centers_np @ R.T
            
            # Try two translations: 0 and -centroid_rotated
            for trans in [np.zeros(2), -np.mean(rotated_centers, axis=0)]:
                shifted_centers = rotated_centers + trans
                
                # Check if any inner hexagons overlap?
                # They are on lattice, so they don't overlap.
                # But after rotation, they are still rigid bodies with fixed relative positions.
                # So they don't overlap.
                
                # Calculate S
                S = _min_outer_side(list(shifted_centers), [0.0]*N, outer_angle_deg=0.0)
                
                if S < best_S:
                    best_S = S
                    best_config = (list(shifted_centers), [0.0]*N, 0.0)
                    
    # If we have time, try to optimize the translation for the best configuration found so far?
    # Or try more rotations?
    # Let's try a finer search on the best few configurations.
    
    if best_config is not None:
        # Let's do a local search on the best configuration's rotation and translation
        # Actually, let's just do a coarse-to-fine search on rotation for the best topology
        # But we already did 15 deg steps.
        
        # Let's try to optimize the 4 outer hexagons' positions slightly?
        # No, they are constrained to lattice? 
        # The problem says "each inner hexagon may translate and rotate".
        # So they don't have to be on the lattice!
        # My lattice assumption restricts the solution.
        # However, for hexagons, the lattice packing is very dense.
        # Moving them off the lattice might allow a smaller bounding box?
        # Yes, if we can "slide" them to reduce the extent.
        
        # Let's use scipy to optimize the positions of the 4 outer hexagons,
        # keeping the core 7 fixed (or all 11 free?).
        # If all 11 are free, it's a hard non-convex problem.
        # Let's fix the core 7 and optimize the 4 outer ones.
        
        try:
            from scipy.optimize import minimize
            
            # Pick the best topology (first one found? or the one with best S)
            # Let's use the best_config centers as initial guess.
            # We need to identify which 4 are the "outer" ones.
            # In our construction, the first 7 are base, next 4 are extra.
            
            init_centers = np.array(best_config[0])
            
            # Objective function
            def obj(params):
                # params: 8 coordinates for 4 outer hexagons
                centers = init_centers.copy()
                for k in range(4):
                    centers[7+k] = params[2*k:2*k+2]
                
                # Check overlap
                # Distance between any two must be >= sqrt(3)
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(centers[i] - centers[j])
                        if d < np.sqrt(3) - 1e-6:
                            return 1e10 # Penalty
                
                S = _min_outer_side(list(centers), [0.0]*N, outer_angle_deg=0.0)
                return S
            
            # Initial guess for the 4 outer hexagons
            x0 = init_centers[7:].flatten()
            
            # Optimize
            result = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if result.fun < best_S:
                # Update best config
                new_centers = init_centers.copy()
                for k in range(4):
                    new_centers[7+k] = result.x[2*k:2*k+2]
                best_S = result.fun
                best_config = (list(new_centers), [0.0]*N, 0.0)
                
        except:
            pass # Ignore optimization errors
            
    if best_config is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    centers, angles, outer_ang = best_config
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    return inner, (0.0, 0.0), best_S, outer_ang

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
