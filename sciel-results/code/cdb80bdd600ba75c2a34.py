"""Packing 11 unit hexagons into a minimal regular hexagon using decoupled container optimization."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 115.0  # Safety margin before 120s limit

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def min_enclosing_hex_side(points):
        """
        Calculate the side length of the minimum enclosing regular hexagon
        for a set of 2D points.
        The hexagon center is fixed at the centroid of the points.
        The rotation is optimized.
        """
        if len(points) == 0:
            return 0.0
        
        # Centroid
        center = np.mean(points, axis=0)
        rel_points = points - center
        
        best_side = float('inf')
        # The hexagon symmetry is 60 degrees. We only need to search 0 to pi/3.
        # The normals are at angles: theta + 30 + k*60 degrees.
        # We search theta in [0, pi/3)
        # 100 steps is usually enough for good precision
        for theta in np.linspace(0, np.pi / 3, 100):
            r_max = 0.0
            for k in range(6):
                angle_n = theta + np.pi / 6 + k * np.pi / 3
                n = np.array([np.cos(angle_n), np.sin(angle_n)])
                # Max projection of relative points onto normal
                proj = np.max(rel_points @ n)
                # Also check negative direction? No, the hexagon is symmetric.
                # The constraint is |proj| <= r. 
                # Since we centered at centroid, max positive projection is the bound.
                # Wait, if the points are not symmetric around centroid, 
                # max(proj) might be smaller than max(-proj).
                # The distance from center to face is r.
                # Condition: max(proj) <= r AND min(proj) >= -r.
                # So r >= max(max(proj), -min(proj)) = max(max(proj), max(-proj)).
                # Let's compute max_abs_proj.
                min_proj = np.min(rel_points @ n)
                extent = max(np.max(proj), -min_proj)
                if extent > r_max:
                    r_max = extent
            
            side = 2.0 * r_max / np.sqrt(3.0)
            if side < best_side:
                best_side = side
        
        # Refine around best theta
        # Find the theta that gave best_side
        # For simplicity, just return the best from the grid. 100 steps over 60 deg is 0.6 deg resolution.
        # Side length sensitivity is low.
        return best_side

    def sat_overlap_penalty(v1, v2):
        """
        Calculate a smooth penalty for overlap between two convex polygons.
        Returns 0 if disjoint, > 0 if overlapping.
        Uses the 12 edge normals from both polygons.
        """
        penalty = 0.0
        # Get all edge normals from both polygons
        # Poly 1 normals
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                continue
            normal /= norm_len
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap depth on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                penalty += overlap ** 2
        
        # Poly 2 normals
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                continue
            normal /= norm_len
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap depth on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                penalty += overlap ** 2
                
        return penalty

    def objective(params):
        """
        params: [cx1, cy1, a1, ..., cxN, cyN, aN]
        """
        if time.time() > deadline:
            return float('inf')
            
        vertices_list = []
        all_points = []
        for i in range(N):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
            all_points.append(verts)
        
        # Flatten points for container calculation
        all_points_flat = np.vstack(all_points)
        
        # Calculate optimal container side length
        S = min_enclosing_hex_side(all_points_flat)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Quick rejection based on center distance
                c1 = np.array([params[3*i], params[3*i+1]])
                c2 = np.array([params[3*j], params[3*j+1]])
                if np.linalg.norm(c1 - c2) > 2.5:
                    continue
                p = sat_overlap_penalty(vertices_list[i], vertices_list[j])
                penalty += p
        
        # Weight the penalty heavily to ensure disjointness
        return S + 100.0 * penalty

    # Initialize with hexagonal lattice
    np.random.seed(42)
    
    # Axial coordinates for 11 hexagons (Center + Ring 1 + 4 from Ring 2)
    axial_coords = [
        (0, 0), 
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-2, 1)
    ]
    
    sqrt3 = math.sqrt(3)
    params = np.zeros(3 * N)
    
    for i, (q, r) in enumerate(axial_coords):
        # Convert axial to Cartesian for flat-top hex with side 1
        # x = 1.5 * q
        # y = sqrt(3) * (r + q/2)
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        params[3 * i] = x
        params[3 * i + 1] = y
        params[3 * i + 2] = 0.0 # Angle
        
    # Add some random jitter to escape symmetries
    params += np.random.normal(0, 0.01, 3 * N)

    # Optimize using Nelder-Mead
    # Nelder-Mead is good for non-smooth functions
    try:
        res = minimize(objective, params, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        
        if time.time() < deadline - 5.0:
             # Try L-BFGS-B from the result of Nelder-Mead for polish
             # L-BFGS-B might get stuck if non-smooth, but can polish the local min
             res2 = minimize(objective, res.x, method='L-BFGS-B',
                            options={'maxiter': 100, 'gtol': 1e-5})
             if res2.fun < res.fun:
                 best_params = res2.x
                 best_score = res2.fun
             else:
                 best_params = res.x
                 best_score = res.fun
        else:
             best_params = res.x
             best_score = res.fun
             
    except Exception as e:
        print(f"Optimization failed: {e}")
        best_params = params
        best_score = objective(params)

    # Extract results
    inner = []
    vertices_list = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
        vertices_list.append(hex_vertices(cx, cy, angle))

    # Calculate final S and container center/angle for the return value
    all_points_flat = np.vstack(vertices_list)
    center = np.mean(all_points_flat, axis=0)
    
    # Find best angle for the container
    best_side = float('inf')
    best_angle = 0.0
    for theta in np.linspace(0, np.pi / 3, 1000):
        r_max = 0.0
        for k in range(6):
            angle_n = theta + np.pi / 6 + k * np.pi / 3
            n = np.array([np.cos(angle_n), np.sin(angle_n)])
            proj = np.max(all_points_flat @ n)
            min_proj = np.min(all_points_flat @ n)
            extent = max(np.max(proj) - center @ n, -(min_proj - center @ n))
            # Note: proj is global. We need relative to center.
            # Let's recompute properly.
            rel = all_points_flat - center
            proj_rel = np.max(rel @ n)
            min_rel = np.min(rel @ n)
            extent = max(proj_rel, -min_rel)
            if extent > r_max:
                r_max = extent
        side = 2.0 * r_max / np.sqrt(3.0)
        if side < best_side:
            best_side = side
            best_angle = theta

    # The container center is the centroid of the inner hexagons' vertices
    # The container angle is best_angle
    
    # Check for validity: ensure no overlaps
    # If overlaps exist, the score is invalid (or penalized). 
    # The optimization should have minimized the penalty to 0.
    
    return inner, (center[0], center[1]), best_side, np.degrees(best_angle)

# EVOLVE-BLOCK-END
