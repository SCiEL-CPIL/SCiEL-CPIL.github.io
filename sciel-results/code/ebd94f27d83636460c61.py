# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex(vertices, outer_angle_deg):
    """
    Find minimum side length and center of a regular hexagon with given angle
    that contains all given vertices.
    """
    a = np.radians(outer_angle_deg)
    # Edge normals at a + 30 + 60*k
    normals = []
    for k in range(6):
        theta = a + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    normals = np.array(normals)
    
    proj = vertices @ normals.T  # (n, 6)
    spreads = proj.max(axis=0) - proj.min(axis=0)
    min_s = spreads.max() / np.sqrt(3)
    
    r = min_s * np.sqrt(3) / 2
    midpoints = (proj.max(axis=0) + proj.min(axis=0)) / 2
    c, _, _, _ = np.linalg.lstsq(normals, midpoints, rcond=None)
    
    return min_s, c


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    sqrt3 = np.sqrt(3)
    
    # Compact hexagonal cluster of 12: rows of 2, 3, 4, 3
    # Using hex grid: x = sqrt(3)*(col + row/2), y = 1.5*row
    hex_centers = []
    
    # Row 0 (y=0): 2 hexagons at cols 0.5, 1.5 (offset to center)
    # Actually, let me center the cluster. 
    # Total width varies by row. Let me just define positions directly.
    
    # More symmetric: use a shape close to hexagonal
    # Center at origin. 
    # Row y=0: 4 hexagons
    # Row y=1.5: 3 hexagons (offset)
    # Row y=3: 3 hexagons
    # Row y=4.5: 2 hexagons (offset)
    # Total: 12
    
    # Let me use:
    # y=0: x = -3*sqrt3/2, -sqrt3/2, sqrt3/2, 3*sqrt3/2  (4)
    # y=1.5: x = -sqrt3, 0, sqrt3  (3, offset by sqrt3/2)
    # y=3: x = -3*sqrt3/2, -sqrt3/2, sqrt3/2, 3*sqrt3/2  (4) -- that's 11
    # Hmm, let me just do 3-4-3-2 or similar
    
    # Simplest compact: 3 columns, 4 rows with offsets
    # y=0: x = 0, sqrt3, 2*sqrt3
    # y=1.5: x = sqrt3/2, 3*sqrt3/2
    # y=3: x = 0, sqrt3, 2*sqrt3
    # y=4.5: x = sqrt3/2, 3*sqrt3/2
    # Total: 3+2+3+2 = 10. Not enough.
    
    # Let me do 4-3-4-1 or better yet, a proper hexagonal arrangement
    # For 12 hexagons, a good shape is a hexagon missing 7 from the 19-hex hexagon
    # Or just use a 3x4 parallelogram
    
    # 3 columns x 4 rows in hex grid:
    # Row 0: (0,0), (sqrt3,0), (2*sqrt3,0)
    # Row 1: (sqrt3/2,1.5), (3*sqrt3/2,1.5), (5*sqrt3/2,1.5)
    # Row 2: (0,3), (sqrt3,3), (2*sqrt3,3)
    # Row 3: (sqrt3/2,4.5), (3*sqrt3/2,4.5), (5*sqrt3/2,4.5)
    # Total: 12
    
    hex_centers = [
        (0, 0), (sqrt3, 0), (2*sqrt3, 0),
        (sqrt3/2, 1.5), (3*sqrt3/2, 1.5), (5*sqrt3/2, 1.5),
        (0, 3), (sqrt3, 3), (2*sqrt3, 3),
        (sqrt3/2, 4.5), (3*sqrt3/2, 4.5), (5*sqrt3/2, 4.5)
    ]
    hex_centers = np.array(hex_centers)
    
    # Compute all 72 vertices
    all_vertices = []
    for cx, cy in hex_centers:
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    centroid = hex_centers.mean(axis=0)
    verts_centered = all_vertices - centroid
    
    def objective(params):
        rot_angle = params[0]
        outer_angle = params[1]
        
        rot_rad = np.radians(rot_angle)
        R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                      [np.sin(rot_rad), np.cos(rot_rad)]])
        
        verts_rot = verts_centered @ R.T
        
        s, _ = _min_containing_hex(verts_rot, outer_angle)
        return s
    
    # Optimization
    best_s = np.inf
    best_params = np.array([0.0, 0.0])
    
    np.random.seed(123)
    n_starts = 40
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            x0 = np.array([0.0, 0.0])
        elif i == 1:
            x0 = np.array([30.0, 0.0])
        elif i == 2:
            x0 = np.array([60.0, 0.0])
        else:
            x0 = np.array([np.random.uniform(-90, 90), np.random.uniform(-90, 90)])
        
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9})
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x.copy()
    
    # Final refinement
    res = minimize(objective, best_params, method='Nelder-Mead',
                 options={'maxiter': 10000, 'xatol': 1e-11, 'fatol': 1e-11})
    if res.fun < best_s:
        best_s = res.fun
        best_params = res.x.copy()
    
    rot_angle, outer_angle = best_params
    rot_rad = np.radians(rot_angle)
    R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                  [np.sin(rot_rad), np.cos(rot_rad)]])
    
    # Final positions
    centers_rot = (hex_centers - centroid) @ R.T + centroid
    verts_rot = verts_centered @ R.T
    
    s, outer_center = _min_containing_hex(verts_rot, outer_angle)
    
    inner_hex_data = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(centers_rot):
        inner_hex_data[i] = [cx, cy, rot_angle]
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
