"""Hexagon packing: 11 unit hexagons inside a regular hexagon (Rigid Body Approach)."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(angle_deg=0.0):
    """Return 6 vertices of a unit regular hexagon rotated by angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append([math.cos(theta), math.sin(theta)])
    return np.array(verts)

def _generate_rigid_cluster():
    """
    Generate the fixed relative positions and orientations for a compact 11-hexagon cluster.
    Configuration: 1 central, 6 surrounding (ring 1), 4 from ring 2.
    All hexagons are aligned (0 degrees relative rotation) for maximum packing density in hexagonal lattice.
    """
    d = math.sqrt(3)  # Distance between centers of adjacent touching hexagons
    
    centers = []
    angles = []
    
    # 1. Central hexagon
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    # 2. Ring 1: 6 hexagons
    for i in range(6):
        theta = i * math.pi / 3.0
        centers.append([d * math.cos(theta), d * math.sin(theta)])
        angles.append(0.0)
        
    # 3. Ring 2: Select 4 hexagons to form a compact shape.
    # The full ring 2 has 12 positions. We want a shape that is as "round" as possible.
    # Positions in axial coordinates (q, r) where distance is max(|q|,|r|,|q+r|)
    # Ring 2 positions:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,0)... wait.
    # Let's just use Cartesian coordinates for Ring 2.
    # Ring 2 centers are at distance 2*d from center? No, in hex grid, distance to ring 2 is 2*d.
    # The 12 positions in Ring 2 are at angles 0, 30, 60... 330 degrees?
    # No, the hex grid ring 2 vertices are at:
    # (2,0) -> (2d, 0)
    # (1,1) -> (d + d/2, d*sqrt(3)/2) = (1.5d, d*sqrt(3)/2)
    # Let's generate all 12 and pick 4 that are symmetric.
    
    ring2_pos = []
    for i in range(12):
        theta = i * math.pi / 6.0
        # The radius for ring 2 in a hex grid is not constant in Euclidean distance?
        # Actually, for a regular hexagon packing, the centers form a triangular lattice.
        # Distance from center to ring 1 is d.
        # Distance from center to ring 2 is 2*d?
        # Let's check: Center (0,0). Neighbor (d, 0). Next neighbor of (d,0) in outward direction is (2d, 0).
        # Another neighbor of (0,0) is (d/2, d*sqrt(3)/2).
        # The positions at distance 2d from origin in the triangular lattice are:
        # Angle 0: (2d, 0)
        # Angle 30: (2d * cos(30), 2d * sin(30))? No, that's not a lattice point.
        
    # Correct generation of Ring 2 in Triangular Lattice:
    # Basis vectors: v1 = (d, 0), v2 = (d/2, d*sqrt(3)/2)
    # Points are i*v1 + j*v2.
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1) ... wait, that's 10?
    # Let's list them properly.
    # (2,0) -> 2*v1
    # (1,1) -> v1+v2
    # (0,2) -> 2*v2
    # (-1,2) -> -v1+2v2
    # (-2,1) -> -2v1+v2
    # (-2,0) -> -2v1
    # (-1,-1) -> -v1-v2
    # (0,-2) -> -2v2
    # (1,-2) -> v1-2v2
    # (2,-1) -> 2v1-v2
    # (2,0) is already there.
    # (1,1) is already there.
    # Wait, the 12 points of ring 2 are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # That is 10 points?
    # Ah, the ring 2 in a hex grid has 12 points.
    # Let's just calculate distances.
    
    # Actually, a simpler "compact" cluster for 11 is:
    # 1 center
    # 6 neighbors
    # 4 "corners" of the hexagon formed by the 6 neighbors.
    # The 6 neighbors form a hexagon. The "corners" are the positions that touch 2 neighbors each.
    # These are at distance 2*d from the center? No.
    # Let's use the coordinates:
    # v1 = (d, 0)
    # v2 = (d/2, d*sqrt(3)/2)
    
    # Ring 2 candidates (i,j) with max(|i|,|j|,|i+j|) == 2?
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # This is 10 points. Where are the other 2?
    # (2,0) and (-2,0) are opposite.
    # (1,1) and (-1,-1) are opposite.
    # (0,2) and (0,-2) are opposite.
    # (-1,2) and (1,-2) are opposite.
    # (-2,1) and (2,-1) are opposite.
    # That's 5 pairs = 10 points.
    # Is the ring size 12?
    # For a hex grid, the number of points at distance k is 6k.
    # So for k=2, there are 12 points.
    # My list above:
    # (2,0) -> dist 2
    # (1,1) -> dist 2
    # (0,2) -> dist 2
    # (-1,2) -> dist 2
    # (-2,1) -> dist 2
    # (-2,0) -> dist 2
    # (-1,-1) -> dist 2
    # (0,-2) -> dist 2
    # (1,-2) -> dist 2
    # (2,-1) -> dist 2
    # I am missing 2 points.
    # (2,0) is (2,0).
    # (1,1) is (1,1).
    # (0,2) is (0,2).
    # (-1,2) is (-1,2).
    # (-2,1) is (-2,1).
    # (-2,0) is (-2,0).
    # (-1,-1) is (-1,-1).
    # (0,-2) is (0,-2).
    # (1,-2) is (1,-2).
    # (2,-1) is (2,-1).
    # Wait, (2,0) and (-2,0) are included.
    # (1,1) and (-1,-1) are included.
    # (0,2) and (0,-2) are included.
    # (-1,2) and (1,-2) are included.
    # (-2,1) and (2,-1) are included.
    # That is 10 points.
    # What about (2, -1)? It is there.
    # What about (1, -2)? It is there.
    # 
    # Let's check (2,0) again.
    # The 12 points are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # I must be missing (2, -1) and (1, -2)? No, they are in the list.
    # Let's count:
    # 1: (2,0)
    # 2: (1,1)
    # 3: (0,2)
    # 4: (-1,2)
    # 5: (-2,1)
    # 6: (-2,0)
    # 7: (-1,-1)
    # 8: (0,-2)
    # 9: (1,-2)
    # 10: (2,-1)
    # 
    # Ah, (2,0) is one. (-2,0) is six.
    # (1,1) is two. (-1,-1) is seven.
    # (0,2) is three. (0,-2) is eight.
    # (-1,2) is four. (1,-2) is nine.
    # (-2,1) is five. (2,-1) is ten.
    # 
    # Where are the other 2?
    # (2, -1) is 10.
    # (1, -2) is 9.
    # 
    # Maybe (2,0) and (0,2) are not the only ones?
    # Let's generate all (i,j) such that max(|i|,|j|,|i+j|) == 2.
    # i ranges -2 to 2.
    # i=2: j=0 (2,0), j=-1 (2,-1), j=-2 (2,-2 -> max(2,2,0)=2). So (2,-2) is also ring 2!
    # i=-2: j=0 (-2,0), j=1 (-2,1), j=2 (-2,2 -> max(2,2,0)=2). So (-2,2) is also ring 2!
    # 
    # So the 12 points are:
    # (2,0), (2,-1), (2,-2)
    # (1,1), (1,-2)
    # (0,2), (0,-2)
    # (-1,2), (-1,-1)
    # (-2,1), (-2,2), (-2,0)
    # 
    # Let's pick 4 that are symmetric.
    # (2,0) is at angle 0.
    # (0,2) is at angle 90? No.
    # (0,2) = 2*v2 = (d, d*sqrt(3)). Angle is 60 deg.
    # (-2,0) is at 180.
    # (0,-2) is at 240.
    # 
    # Let's pick (2,0), (0,2), (-2,0), (0,-2). These are the "cardinal" points of the hex grid (axes of symmetry).
    
    selected_ring2 = [(2,0), (0,2), (-2,0), (0,-2)]
    
    for i, j in selected_ring2:
        x = i * d + j * (d / 2.0)
        y = j * (d * math.sqrt(3) / 2.0)
        centers.append([x, y])
        angles.append(0.0)
        
    return np.array(centers), np.array(angles)

def _cluster_vertices(centers, angles, tx, ty, rot_deg):
    """
    Compute absolute vertices of all hexagons in the cluster given global translation (tx, ty) and rotation rot_deg.
    """
    rot_rad = math.radians(rot_deg)
    c_rot = math.cos(rot_rad)
    s_rot = math.sin(rot_rad)
    
    all_verts = []
    for i in range(N):
        # Rotate local center
        cx = centers[i][0]
        cy = centers[i][1]
        rx = cx * c_rot - cy * s_rot
        ry = cx * s_rot + cy * c_rot
        
        # Translate
        rx += tx
        ry += ty
        
        # Get local vertices (already rotated by angles[i] which is 0 here)
        # Since angles are 0, we just rotate the standard hexagon vertices by rot_rad
        # Standard hexagon vertices (radius 1)
        for k in range(6):
            theta = rot_rad + k * math.pi / 3.0
            vx = rx + math.cos(theta)
            vy = ry + math.sin(theta)
            all_verts.append([vx, vy])
            
    return np.array(all_verts)

def _outer_side_length(vertices, outer_rot_deg=0.0):
    """
    Calculate the minimum side length of an outer hexagon (fixed rotation) containing the given vertices.
    """
    # Outer hexagon normals
    a = math.radians(outer_rot_deg)
    normals = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        normals.append([math.cos(theta), math.sin(theta)])
    normals = np.array(normals)
    
    # Project vertices onto normals
    projections = vertices @ normals.T # (M, 6)
    abs_proj = np.abs(projections)
    max_extent = np.max(abs_proj)
    
    # Side length S is related to max extent by: max_extent = S * cos(pi/6) = S * sqrt(3)/2
    S = max_extent / (math.sqrt(3) / 2.0)
    return S

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    centers, angles = _generate_rigid_cluster()
    
    # Objective function
    def objective(params):
        if time.time() > deadline:
            return 1e10
        tx, ty, rot = params
        verts = _cluster_vertices(centers, angles, tx, ty, rot)
        return _outer_side_length(verts, 0.0)
        
    # We can fix the center of the cluster at (0,0) by symmetry?
    # The optimal outer hexagon will be centered at the centroid of the cluster if the cluster is symmetric?
    # The cluster (1+6+4) is not fully symmetric.
    # However, we can optimize tx, ty, rot.
    
    # Initial guess: Center at 0,0, rotation 0
    x0 = [0.0, 0.0, 0.0]
    
    # Try multiple rotations to avoid local minima
    best_S = float('inf')
    best_params = x0
    
    # Grid search over rotation in 15 degree steps, then refine
    for rot_init in range(0, 180, 15):
        if time.time() > deadline - 5:
            break
            
        # Local optimization around this rotation
        x0_local = [0.0, 0.0, float(rot_init)]
        
        try:
            res = minimize(objective, x0_local, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 5000})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Further refinement with BFGS if time permits
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='BFGS', 
                           options={'gtol': 1e-10, 'maxiter': 1000})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Extract best parameters
    tx, ty, rot = best_params
    
    # Calculate final vertices and side length
    verts = _cluster_vertices(centers, angles, tx, ty, rot)
    final_S = _outer_side_length(verts, 0.0)
    
    # Add tiny epsilon to ensure validity due to floating point
    final_S *= 1.000001
    
    # Format output
    # The output requires: inner (list of [x, y, angle_deg]), center (x, y), side, angle
    # inner hexagons are rotated by `rot` and translated by (tx, ty)
    inner = []
    for i in range(N):
        cx = centers[i][0]
        cy = centers[i][1]
        # Rotate center
        rot_rad = math.radians(rot)
        c_rot = math.cos(rot_rad)
        s_rot = math.sin(rot_rad)
        rx = cx * c_rot - cy * s_rot + tx
        ry = cx * s_rot + cy * c_rot + ty
        inner.append([float(rx), float(ry), float(rot)])
        
    return inner, (0.0, 0.0), float(final_S), 0.0

# EVOLVE-BLOCK-END
