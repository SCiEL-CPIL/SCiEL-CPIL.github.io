import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Compact arrangement of 11 hexagons
    # Use axial coordinates (q, r) for flat-top hex grid
    # Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    axial_positions = [
        (0, 0),    # center
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),  # first ring (6)
        (2, 0), (0, 2), (-2, 0), (0, -2)  # 4 more from second ring
    ]
    
    positions = []
    for q, r in axial_positions:
        x = sqrt3 * (q + r/2)
        y = 1.5 * r
        positions.append((x, y))
    
    # Hexagon vertices (unit side, flat-top, vertex at angle 0)
    hex_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    hex_verts_local = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    all_points = []
    for (cx, cy) in positions:
        verts = hex_verts_local + np.array([cx, cy])
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    hull = ConvexHull(all_points)
    hull_points = all_points[hull.vertices]
    
    center = np.mean(hull_points, axis=0)
    
    def enclosing_side(theta):
        normals = np.array([[np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)] for k in range(6)])
        rel = hull_points - center
        projections = rel @ normals.T
        max_proj = np.max(projections, axis=0)
        min_proj = np.min(projections, axis=0)
        apothem = np.max(np.maximum(max_proj, -min_proj))
        return 2 * apothem / np.sqrt(3)
    
    result = minimize_scalar(enclosing_side, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-10})
    best_theta = result.x
    best_side = result.fun
    
    inner = [[float(x), float(y), 0.0] for (x, y) in positions]
    
    return inner, (float(center[0]), float(center[1])), float(best_side), float(best_theta * 180 / np.pi)
# EVOLVE-BLOCK-END
