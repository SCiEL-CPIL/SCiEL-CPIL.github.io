import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with given center, side length, and rotation."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def compute_enclosing_hexagon(inner_hexagons):
    """
    Find the smallest regular hexagon that encloses all given inner hexagons.
    Each inner hexagon is specified as (cx, cy, angle_deg, side).
    Returns (center_x, center_y, side_length, angle_deg).
    """
    # Get all vertices of all inner hexagons
    all_points = []
    for (cx, cy, angle, side) in inner_hexagons:
        verts = hexagon_vertices(cx, cy, side, angle)
        all_points.extend(verts.tolist())
    all_points = np.array(all_points)
    
    def objective(params):
        cx, cy, side, angle_deg = params
        if side < 0.1:
            return 1e10
        angle_rad = np.radians(angle_deg)
        inradius = side * np.sqrt(3) / 2
        
        # Outward normals for each edge of the regular hexagon
        normals = []
        for k in range(6):
            theta = angle_rad + (k + 0.5) * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        diffs = all_points - np.array([cx, cy])
        projections = diffs @ normals.T
        max_proj = np.max(projections, axis=1)
        violation = np.max(max_proj - inradius)
        if violation > 0:
            return side + 100 * violation
        return side
    
    cx0 = np.mean(all_points[:, 0])
    cy0 = np.mean(all_points[:, 1])
    x_range = np.max(all_points[:, 0]) - np.min(all_points[:, 0])
    y_range = np.max(all_points[:, 1]) - np.min(all_points[:, 1])
    side0 = max(x_range, y_range) / np.sqrt(3) * 1.5
    
    best_result = None
    best_obj = 1e10
    
    for angle_init in [0, 30, 60, 90, 120, 150]:
        x0 = [cx0, cy0, side0, angle_init]
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
        except:
            pass
    
    if best_result is None:
        return (cx0, cy0, side0 * 2, 0.0)
    
    cx, cy, side, angle = best_result
    if side < 0.1:
        side = side0 * 2
    return (cx, cy, side, angle)

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    
    # Arrange 11 unit hexagons in a compact hexagonal lattice pattern
    spacing = np.sqrt(3)
    
    # Compact 11-hexagon arrangement in axial coordinates
    axial_positions = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2)
    ]
    
    # Convert to cartesian
    centers = []
    for (q, r) in axial_positions:
        x = spacing * (q + r / 2)
        y = spacing * (np.sqrt(3) / 2) * r
        centers.append((x, y))
    
    # All hexagons have the same orientation
    inner_hexagons = [(cx, cy, 0.0, 1.0) for (cx, cy) in centers]
    
    # Find enclosing hexagon
    cx, cy, side, angle = compute_enclosing_hexagon(inner_hexagons)
    
    # Build inner list: (x, y, angle_degrees)
    inner = [[cx, cy, 0.0] for (cx, cy, _, _) in inner_hexagons]
    
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
