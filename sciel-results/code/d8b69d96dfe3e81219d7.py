"""
Strategy: Rigid Cluster Rotation Optimization with Exact Containment.

Key Differences from Current Program:
1. Problem Framing: Instead of optimizing 33 independent variables (11 positions + 11 angles) 
   with a complex penalty function for overlaps, this strategy assumes the optimal internal 
   packing is a rigid hexagonal lattice (center + 6 neighbors + 4 outer).
2. Optimization Method: We optimize only the global rotation angle of the cluster. 
   The internal geometry is fixed to the minimal non-overlapping configuration.
3. Efficiency: This reduces the problem from a 33D non-convex optimization with 
   discontinuities to a 1D (or 2D if translation is included) periodic function 
   minimization, which is extremely fast and robust.
4. Containment: We compute the exact minimum side length of the outer hexagon required 
   to contain the rotated cluster by checking the support function against all vertices.
5. Structural Change: We now explicitly optimize the outer hexagon's angle as well, 
   decoupling the inner cluster rotation from the outer container orientation. 
   This allows the outer hexagon to align optimally with the cluster's geometry, 
   potentially reducing the required side length significantly.
"""

import time
import numpy as np

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _get_cluster_vertices(rotation_angle):
    """
    Generate vertices of the 11 unit hexagons in the rigid cluster,
    rotated by rotation_angle around the origin.
    
    The cluster topology is:
    - 1 center hex at (0,0)
    - 6 hexes at distance 2*sqrt(3) (touching)
    - 4 hexes in the gaps of the second ring, at distance 6.
    """
    sqrt3 = np.sqrt(3)
    centers = []
    centers.append([0.0, 0.0])
    
    # Ring 1
    for i in range(6):
        a = i * np.pi / 3
        centers.append([2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)])
        
    # Ring 2 (Gaps)
    gap_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for a in gap_angles:
        centers.append([6.0 * np.cos(a), 6.0 * np.sin(a)])
        
    centers = np.array(centers)
    
    # Rotate centers
    a_rot = np.radians(rotation_angle)
    c_rot = np.cos(a_rot)
    s_rot = np.sin(a_rot)
    
    # Apply rotation matrix
    rotated_centers = np.column_stack([
        centers[:, 0] * c_rot - centers[:, 1] * s_rot,
        centers[:, 0] * s_rot + centers[:, 1] * c_rot
    ])
    
    # Generate all vertices for all 11 hexagons
    # Each hexagon has 6 vertices.
    all_verts = []
    for cx, cy in rotated_centers:
        # The local angle of the hexagon relative to the cluster is 0.
        # So its global angle is rotation_angle.
        verts = _hex_vertices(cx, cy, 1.0, rotation_angle)
        all_verts.append(verts)
        
    return np.vstack(all_verts)

def _min_outer_side(vertices, outer_center, outer_angle):
    """
    Compute minimum outer side length to contain all vertices.
    Outer hexagon is centered at outer_center, rotated by outer_angle.
    """
    cx, cy = outer_center
    dx = vertices[:, 0] - cx
    dy = vertices[:, 1] - cy
    
    a_outer = np.radians(outer_angle)
    ca = np.cos(a_outer)
    sa = np.sin(a_outer)
    
    # Rotate vertices into outer hexagon frame
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # Normals angles: 30, 90, 150, 210, 270, 330 degrees.
    # In radians: pi/6, pi/2, 5*pi/6, 7*pi/6, 3*pi/2, 11*pi/6.
    normals = []
    for i in range(6):
        theta = np.pi/6 + i * np.pi/3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    # Project vertices onto normals
    # We want dot product of each vertex with each normal.
    projections = vertices @ normals.T
    
    # We want the max projection for each normal.
    max_projections = np.max(projections, axis=0)
    
    # The required side length S is such that S * sqrt(3)/2 >= max_projections
    # S >= max_projections / (sqrt(3)/2)
    
    min_side = np.max(max_projections) / (np.sqrt(3) / 2)
    
    return float(min_side)

def _objective(params):
    """
    Objective function: Minimize the side length of the outer hexagon.
    params: [inner_rotation, outer_rotation]
    We fix the outer hexagon center at (0,0).
    """
    inner_rot = params[0]
    outer_rot = params[1]
    
    verts = _get_cluster_vertices(inner_rot)
    side = _min_outer_side(verts, (0.0, 0.0), outer_rot)
    return side

def construct():
    global START_TIME, DEADLINE
    
    # The function is periodic with period 60 degrees for both inner and outer.
    # We can search in [0, 60) for both.
    
    # Coarse search
    angles_coarse = np.linspace(0, 60, 101)
    best_inner = 0
    best_outer = 0
    best_side = np.inf
    
    for a_in in angles_coarse:
        if time.time() > DEADLINE - 10.0:
            break
        for a_out in angles_coarse:
            s = _objective([a_in, a_out])
            if s < best_side:
                best_side = s
                best_inner = a_in
                best_outer = a_out
                
    # Fine search around best_inner, best_outer
    # The function is smooth. We can use scipy.optimize.minimize
    from scipy.optimize import minimize
    
    # Define bounds for fine search
    lower = np.array([max(0, best_inner - 2.0), max(0, best_outer - 2.0)])
    upper = np.array([min(60, best_inner + 2.0), min(60, best_outer + 2.0)])
    
    try:
        res = minimize(_objective, np.array([best_inner, best_outer]), method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_side:
            best_side = res.fun
            best_inner = res.x[0]
            best_outer = res.x[1]
    except Exception:
        pass
    
    # Extract solution
    # The inner hexagons are at the rotated positions.
    # We need to return their centers and angles.
    
    sqrt3 = np.sqrt(3)
    centers = []
    centers.append([0.0, 0.0])
    for i in range(6):
        a = i * np.pi / 3
        centers.append([2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)])
    gap_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for a in gap_angles:
        centers.append([6.0 * np.cos(a), 6.0 * np.sin(a)])
        
    centers = np.array(centers)
    
    a_rot = np.radians(best_inner)
    c_rot = np.cos(a_rot)
    s_rot = np.sin(a_rot)
    
    rotated_centers = np.column_stack([
        centers[:, 0] * c_rot - centers[:, 1] * s_rot,
        centers[:, 0] * s_rot + centers[:, 1] * c_rot
    ])
    
    inner_list = []
    for i in range(N):
        inner_list.append([float(rotated_centers[i, 0]), float(rotated_centers[i, 1]), float(best_inner)])
        
    outer_center = (0.0, 0.0)
    outer_side = float(best_side)
    outer_angle = float(best_outer)
    
    return inner_list, outer_center, outer_side, outer_angle

if __name__ == "__main__":
    construct()
