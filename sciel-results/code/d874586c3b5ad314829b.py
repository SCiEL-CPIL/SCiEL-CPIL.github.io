"""
Strategy: Rigid Body Grid Search with Global Orientation Optimization.

This approach fundamentally changes the representation of the problem.
Instead of treating the 11 hexagons as 11 independent rigid bodies with
free translation and rotation (33 degrees of freedom), we treat them as a
single, rigid "cluster" or "molecule". 

The topology of the cluster is fixed based on a high-quality symmetric
arrangement (a central hexagon surrounded by 6, and 4 more in the outer
ring). This reduces the internal degrees of freedom to zero. The only
degrees of freedom left are:
1. The global rotation of the entire cluster.
2. The global rotation of the outer container hexagon.
3. The side length of the outer container hexagon (which is minimized to
   fit the rotated cluster).

The translation is fixed by centering the cluster's centroid at the origin
and the outer hexagon's center at the origin. This is optimal for a
symmetric container and a symmetric cluster.

The optimization problem is now a 2D search: (cluster_rotation, 
outer_rotation). We perform a coarse grid search over these two angles, 
and then a fine local refinement (e.g., Nelder-Mead or a finer grid) 
around the best candidate. This is much more robust and faster than 
high-dimensional nonlinear optimization, and it exploits the symmetry 
of the problem.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill

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

def _generate_rigid_cluster():
    """
    Generate the fixed relative positions of the 11 hexagons in the cluster.
    Topology:
    - Hex 0: Center (0,0)
    - Hex 1-6: Ring 1 at distance 2*sqrt(3) (touching Hex 0)
    - Hex 7-10: Ring 2 at distance 6*sqrt(3) / 2? No, let's use the standard hex grid.
    
    Standard hex grid coordinates (axial):
    Center: (0,0)
    Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1)
    
    We only need 4 more for a total of 11. A symmetric choice for the 4 in Ring 2 
    would be at the "corners" or "edges". 
    Let's place them at (2,0), (0,2), (-2,0), (0,-2) in axial coordinates? 
    Or maybe (2,0), (-2,0), (1,1), (-1,-1)? 
    Actually, the most compact symmetric arrangement for 11 is the full ring 1 (7) 
    plus 4 from ring 2. 
    Let's try placing the 4 at the "points" of the hexagon formed by ring 1.
    Axial to Cartesian:
    x = size * (q + r/2)
    y = size * (sqrt(3)/2 * r)
    Here size is the distance between centers of touching hexagons = 2*sqrt(3).
    
    Let's define the positions directly in Cartesian coordinates for simplicity.
    Distance between centers of touching hexagons: d = 2 * sqrt(3) * side = 2 * sqrt(3).
    
    Center: (0,0)
    Ring 1 (6 hexagons):
    Angles: 0, 60, 120, 180, 240, 300 degrees.
    Positions: d * (cos(theta), sin(theta))
    
    Ring 2 (we need 4 more):
    Let's pick 4 symmetrically. 
    Option A: (2,0), (0,2), (-2,0), (0,-2) in axial.
    Axial (2,0) -> Cartesian (2*d, 0)
    Axial (0,2) -> Cartesian (d, d*sqrt(3))
    Axial (-2,0) -> Cartesian (-2*d, 0)
    Axial (0,-2) -> Cartesian (-d, -d*sqrt(3))
    This is not rotationally symmetric (only 2-fold).
    
    Option B: (2,0), (-2,0), (1,1), (-1,-1) in axial.
    (2,0) -> (2d, 0)
    (-2,0) -> (-2d, 0)
    (1,1) -> (1.5d, d*sqrt(3)/2)
    (-1,-1) -> (-1.5d, -d*sqrt(3)/2)
    This is also only 2-fold symmetric.
    
    Option C: Use the 4 positions in Ring 2 that are "aligned" with the gaps of Ring 1.
    Ring 1 positions are at angles 0, 60, ...
    The "gaps" are at 30, 90, ...
    The closest points in Ring 2 to these gaps?
    Actually, let's just use a known good packing for 11 hexagons.
    A common dense packing is the "flower" (7) + 4 more.
    Let's try placing the 4 at the vertices of a square? No, hexagons don't like squares.
    
    Let's use the 4 positions from Ring 2 that are at angles 0, 120, 240, and... wait, 
    if we pick 3, it's 3-fold symmetric. We need 4.
    Maybe the optimal topology isn't perfectly symmetric? 
    However, for a first try, let's assume a topology that is at least 2-fold symmetric 
    or close to it, and let the global rotation optimize the fit.
    
    Let's use the topology:
    0: (0,0)
    1-6: Ring 1
    7: (2d, 0)  [Axial 2,0]
    8: (-2d, 0) [Axial -2,0]
    9: (1.5d, d*sqrt(3)/2) [Axial 1,1]
    10: (-1.5d, -d*sqrt(3)/2) [Axial -1,-1]
    
    This cluster is centrally symmetric (180 deg rotation).
    """
    d = 2 * np.sqrt(3)
    pos = np.zeros((N, 2))
    pos[0] = [0, 0]
    
    # Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [d * np.cos(a), d * np.sin(a)]
        
    # Ring 2 (4 hexagons)
    # Axial (2,0) -> (2d, 0)
    pos[7] = [2*d, 0]
    # Axial (-2,0) -> (-2d, 0)
    pos[8] = [-2*d, 0]
    # Axial (1,1) -> (1.5d, d*sqrt(3)/2)
    pos[9] = [1.5*d, d*np.sqrt(3)/2]
    # Axial (-1,-1) -> (-1.5d, -d*sqrt(3)/2)
    pos[10] = [-1.5*d, -d*np.sqrt(3)/2]
    
    return pos

def _min_outer_side_rigid(cluster_pos, cluster_rot, outer_rot):
    """
    Compute the minimum outer side length to contain the rigid cluster.
    cluster_pos: (N, 2) array of relative positions.
    cluster_rot: rotation angle of the cluster (radians).
    outer_rot: rotation angle of the outer hexagon (radians).
    """
    # Rotate cluster positions
    c, s = np.cos(cluster_rot), np.sin(cluster_rot)
    rot_matrix = np.array([[c, -s], [s, c]])
    rotated_pos = cluster_pos @ rot_matrix.T
    
    # Generate vertices for all hexagons in the rotated cluster
    # All inner hexagons have the same orientation relative to the cluster?
    # Yes, in a rigid body model, they are fixed relative to each other.
    # So if the cluster rotates by theta, all hexagons rotate by theta.
    
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3 + cluster_rot
    total_angles = np.tile(base_angles, (N, 1)) # (N, 6)
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = rotated_pos[:, 0:1] + offsets_x
    verts_y = rotated_pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    # Outer hexagon is centered at (0,0) and rotated by outer_rot
    a_outer = outer_rot
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * all_verts_x - sa * all_verts_y
    ly = sa * all_verts_x + ca * all_verts_y
    
    sqrt3 = np.sqrt(3)
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _check_validity_rigid(cluster_pos, cluster_rot, outer_side, outer_rot, tol=1e-5):
    """
    Check if the rigid packing is valid.
    Since the topology is fixed and known to be non-overlapping (touching),
    we only need to check containment.
    """
    c, s = np.cos(cluster_rot), np.sin(cluster_rot)
    rot_matrix = np.array([[c, -s], [s, c]])
    rotated_pos = cluster_pos @ rot_matrix.T
    
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3 + cluster_rot
    total_angles = np.tile(base_angles, (N, 1))
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = rotated_pos[:, 0:1] + offsets_x
    verts_y = rotated_pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    a_outer = outer_rot
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * all_verts_x - sa * all_verts_y
    ly = sa * all_verts_x + ca * all_verts_y
    
    sqrt3 = np.sqrt(3)
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    max_req = np.max(np.maximum(c1, np.maximum(c2, c3)))
    
    # If max_req <= outer_side + tol, it's contained
    return max_req <= outer_side + tol

def construct():
    global START_TIME, DEADLINE
    
    cluster_pos = _generate_rigid_cluster()
    
    best_side = np.inf
    best_cluster_rot = 0.0
    best_outer_rot = 0.0
    
    # Grid search over cluster_rot and outer_rot
    # Ranges: 0 to pi/3 (60 deg) is sufficient due to hexagonal symmetry?
    # The cluster has 180 deg symmetry. The outer hexagon has 60 deg symmetry.
    # So we can search cluster_rot in [0, pi/2) and outer_rot in [0, pi/3).
    # Let's be safe and search [0, pi) for cluster and [0, pi/3) for outer.
    
    # Coarse grid
    n_steps = 60
    cluster_rots = np.linspace(0, np.pi/2, n_steps, endpoint=False)
    outer_rots = np.linspace(0, np.pi/3, n_steps, endpoint=False)
    
    for cr in cluster_rots:
        if time.time() > DEADLINE - 20.0:
            break
        for orr in outer_rots:
            side = _min_outer_side_rigid(cluster_pos, cr, orr)
            if side < best_side:
                best_side = side
                best_cluster_rot = cr
                best_outer_rot = orr
    
    # Fine grid around the best
    # Expand search range slightly to ensure we didn't miss a boundary
    # But symmetry suggests minima are inside.
    
    # Refinement with Nelder-Mead
    def objective(params):
        cr, orr = params
        return _min_outer_side_rigid(cluster_pos, cr, orr)
    
    x0 = np.array([best_cluster_rot, best_outer_rot])
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < best_side:
            best_side = res.fun
            best_cluster_rot = res.x[0]
            best_outer_rot = res.x[1]
            
    except Exception:
        pass
    
    # Multiple restarts to avoid local minima in the 2D landscape
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > DEADLINE - 10.0:
            break
            
        # Random restart
        cr_start = np.random.uniform(0, np.pi/2)
        orr_start = np.random.uniform(0, np.pi/3)
        
        x0 = np.array([cr_start, orr_start])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_side:
                best_side = res.fun
                best_cluster_rot = res.x[0]
                best_outer_rot = res.x[1]
                
        except Exception:
            pass
            
    # Final validity check
    if not _check_validity_rigid(cluster_pos, best_cluster_rot, best_side, best_outer_rot):
        # Fallback: expand side slightly
        # Find the exact required side
        best_side = _min_outer_side_rigid(cluster_pos, best_cluster_rot, best_outer_rot)
        # Add a tiny epsilon for floating point safety in evaluator
        best_side += 1e-5
        
    # Construct the output
    # Inner hexagons: positions are cluster_pos rotated by best_cluster_rot
    # Angles are best_cluster_rot (in degrees)
    
    c, s = np.cos(best_cluster_rot), np.sin(best_cluster_rot)
    rot_matrix = np.array([[c, -s], [s, c]])
    final_pos = cluster_pos @ rot_matrix.T
    
    inner_list = []
    for i in range(N):
        x, y = final_pos[i]
        angle_deg = np.degrees(best_cluster_rot)
        inner_list.append([float(x), float(y), float(angle_deg)])
        
    outer_center = [0.0, 0.0]
    outer_side = float(best_side)
    outer_angle = float(np.degrees(best_outer_rot))
    
    return inner_list, outer_center, outer_side, outer_angle

if __name__ == "__main__":
    construct()
