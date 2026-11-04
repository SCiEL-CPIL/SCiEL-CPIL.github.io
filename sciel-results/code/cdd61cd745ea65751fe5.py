import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

_HEX_VERTS_0 = np.array([[math.cos(i * math.pi / 3.0), math.sin(i * math.pi / 3.0)] for i in range(6)])

def _get_hex_vertices(center, angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return _HEX_VERTS_0 @ R.T + center

def _sat_overlap_penalty(v1, v2):
    """
    Returns 0 if polygons v1, v2 do not overlap.
    Returns a positive penalty if they overlap.
    v1, v2 are Nx2 arrays of vertices.
    """
    # Get edges of v1 and v2
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    # Normals to edges
    normals1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
    normals2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
    
    # Normalize normals (not strictly necessary for sign check, but good for projection scaling)
    # Actually, we just need to project onto them.
    
    min_overlap = float('inf')
    
    def project_on_normal(normal):
        # Project both polygons onto the normal
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            return 0.0 # Separated
        
        # Calculate overlap depth
        overlap = min(max1, max2) - max(min1, min2)
        return overlap

    # Check normals of v1
    for n in normals1:
        # Skip zero-length edges (shouldn't happen for regular hex)
        if np.linalg.norm(n) < 1e-9: continue
        ov = project_on_normal(n)
        if ov == 0.0:
            return 0.0
        min_overlap = min(min_overlap, ov)
        
    # Check normals of v2
    for n in normals2:
        if np.linalg.norm(n) < 1e-9: continue
        ov = project_on_normal(n)
        if ov == 0.0:
            return 0.0
        min_overlap = min(min_overlap, ov)
        
    if min_overlap == float('inf'):
        return 0.0 # Should not happen if not separated
        
    return min_overlap

def _compute_side_length(centers, angles, outer_angle_rad=0.0):
    """
    Compute minimum side length S of outer hexagon centered at origin with given rotation
    that contains all inner hexagons.
    """
    outer_normals = []
    for k in range(6):
        theta = outer_angle_rad + k * math.pi / 3.0
        outer_normals.append([math.cos(theta), math.sin(theta)])
    outer_normals = np.array(outer_normals)
    
    max_extent = 0.0
    for i in range(len(centers)):
        verts = _get_hex_vertices(centers[i], angles[i])
        projections = verts @ outer_normals.T
        abs_proj = np.abs(projections)
        max_extent = max(max_extent, np.max(abs_proj))
    
    # For a regular hexagon of side S, the distance from center to side is S * sqrt(3) / 2
    # So S = max_extent / (sqrt(3)/2)
    S = max_extent / (math.sqrt(3) / 2.0)
    return S

def _symmetric_config(r1, r2, theta_inner):
    """
    Generate centers and angles for a symmetric 11-hexagon packing.
    r1: radius of inner ring (6 hexes)
    r2: radius of outer ring (4 hexes)
    theta_inner: rotation of inner hexagons relative to container (radians)
    """
    centers = []
    angles = []
    
    # Central
    centers.append([0.0, 0.0])
    angles.append(theta_inner)
    
    # Inner ring (6)
    for i in range(6):
        ang = i * math.pi / 3.0
        x = r1 * math.cos(ang)
        y = r1 * math.sin(ang)
        centers.append([x, y])
        angles.append(theta_inner)
        
    # Outer ring (4) - interleaved at 30, 150, 210, 330 degrees
    # This places them between the inner ring hexes
    outer_angles = [math.pi / 6.0, math.pi / 6.0 + math.pi / 2.0, 
                    math.pi / 6.0 + math.pi, math.pi / 6.0 + 3 * math.pi / 2.0]
    for ang in outer_angles:
        x = r2 * math.cos(ang)
        y = r2 * math.sin(ang)
        centers.append([x, y])
        angles.append(theta_inner)
        
    return np.array(centers), np.array(angles)

def _objective(params):
    r1, r2, theta_inner = params
    
    # Basic bounds check to avoid crazy values
    if r1 < 0.5 or r2 < r1 or r1 > 5 or r2 > 10:
        return 1e10
        
    centers, angles = _symmetric_config(r1, r2, theta_inner)
    
    # Calculate Side Length
    S = _compute_side_length(centers, angles, 0.0)
    
    # Calculate Overlap Penalty
    penalty = 0.0
    for i in range(N):
        v1 = _get_hex_vertices(centers[i], angles[i])
        for j in range(i + 1, N):
            v2 = _get_hex_vertices(centers[j], angles[j])
            ov = _sat_overlap_penalty(v1, v2)
            if ov > 0:
                penalty += ov ** 2
                
    return S + 100.0 * penalty

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Initial guess
    # For aligned hexes (theta=0), min dist is sqrt(3) ~ 1.732
    # For 30 deg rotated, min dist is 2.0
    # Let's start with theta=0, r1=sqrt(3), r2=2*sqrt(3)
    r0 = math.sqrt(3)
    x0 = [r0, 2 * r0, 0.0]
    
    best_S = float('inf')
    best_params = x0
    
    # Local Optimization
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000, 'maxfev': 20000})
        
        if res.fun < best_S:
            best_params = res.x
            best_S = res.fun
            
    except Exception:
        pass
    
    # If the symmetric solution has overlaps, try to fix them or find a better local min
    # We can also try a few random starts in the 3D space if time permits
    if time.time() < deadline - 10.0:
        np.random.seed(42)
        for _ in range(50):
            r1 = np.random.uniform(1.5, 4.0)
            r2 = np.random.uniform(r1, 6.0)
            theta = np.random.uniform(0, math.pi/3)
            x_try = [r1, r2, theta]
            
            res_try = minimize(_objective, x_try, method='Nelder-Mead', 
                               options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 5000})
            if res_try.fun < best_S:
                best_S = res_try.fun
                best_params = res_try.x
                
            if time.time() > deadline - 5.0:
                break

    r1, r2, theta_inner = best_params
    centers, angles = _symmetric_config(r1, r2, theta_inner)
    
    # Final verification and side length calculation
    S = _compute_side_length(centers, angles, 0.0)
    
    # Check validity
    valid = True
    for i in range(N):
        v1 = _get_hex_vertices(centers[i], angles[i])
        for j in range(i + 1, N):
            v2 = _get_hex_vertices(centers[j], angles[j])
            if _sat_overlap_penalty(v1, v2) > 1e-6:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: Use a known valid packing, e.g. slightly expanded
        # Expand r1 and r2 until valid
        # This is a crude fallback
        scale = 1.0
        while not valid and scale < 2.0 and time.time() < deadline - 1.0:
            scale += 0.01
            r1_s = r1 * scale
            r2_s = r2 * scale
            centers, angles = _symmetric_config(r1_s, r2_s, theta_inner)
            valid = True
            for i in range(N):
                v1 = _get_hex_vertices(centers[i], angles[i])
                for j in range(i + 1, N):
                    v2 = _get_hex_vertices(centers[j], angles[j])
                    if _sat_overlap_penalty(v1, v2) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                centers, angles = _symmetric_config(r1_s, r2_s, theta_inner)
                S = _compute_side_length(centers, angles, 0.0)
                break

    # Format output
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], math.degrees(angles[i])])
        
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
