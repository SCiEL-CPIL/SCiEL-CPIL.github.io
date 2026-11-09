"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def min_enclosing_hexagon_side(points, hex_angle_deg):
    """Find minimum side length of a regular hexagon (with given orientation) enclosing all points."""
    phi = np.radians(hex_angle_deg)
    normals = []
    for k in range(6):
        theta = phi + k * np.pi / 3 + np.pi / 6
        normals.append((np.cos(theta), np.sin(theta)))
    
    extents = []
    for k in range(6):
        nx, ny = normals[k]
        projs = points[:, 0] * nx + points[:, 1] * ny
        extent = np.max(projs) - np.min(projs)
        extents.append(extent)
    
    max_extent = max(extents)
    side = max_extent / np.sqrt(3)
    return side

def construct():
    best_side = float('inf')
    best_result = None
    
    arrangements = []
    
    # Arrangement A: 4-4-4 honeycomb
    pts = []
    for row in range(3):
        for col in range(4):
            x = col * np.sqrt(3) + (row % 2) * np.sqrt(3) / 2
            y = row * 1.5
            pts.append((x, y))
    arrangements.append(np.array(pts))
    
    # Arrangement B: 6x2 honeycomb
    pts = []
    for row in range(2):
        for col in range(6):
            x = col * np.sqrt(3) + (row % 2) * np.sqrt(3) / 2
            y = row * 1.5
            pts.append((x, y))
    arrangements.append(np.array(pts))
    
    # Arrangement C: 3-4-3-2 honeycomb
    pts = []
    for col in range(3):
        pts.append((col * np.sqrt(3), 0))
    for col in range(4):
        pts.append((col * np.sqrt(3) + np.sqrt(3)/2, 1.5))
    for col in range(3):
        pts.append((col * np.sqrt(3), 3.0))
    for col in range(2):
        pts.append((col * np.sqrt(3) + np.sqrt(3)/2, 4.5))
    arrangements.append(np.array(pts))
    
    # Arrangement D: 2-4-4-2 honeycomb
    pts = []
    for col in range(2):
        pts.append((col * np.sqrt(3) + np.sqrt(3), 0))
    for col in range(4):
        pts.append((col * np.sqrt(3), 1.5))
    for col in range(4):
        pts.append((col * np.sqrt(3) + np.sqrt(3)/2, 3.0))
    for col in range(2):
        pts.append((col * np.sqrt(3) + np.sqrt(3), 4.5))
    arrangements.append(np.array(pts))
    
    # Arrangement E: Hexagonal cluster (1+6+5)
    pts = []
    pts.append((0, 0))
    for k in range(6):
        angle = k * np.pi / 3
        pts.append((np.sqrt(3) * np.cos(angle), 1.5 * np.sin(angle)))
    for k in range(5):
        angle = k * np.pi / 3 + np.pi / 6
        pts.append((2 * np.sqrt(3) * np.cos(angle), 3 * np.sin(angle)))
    arrangements.append(np.array(pts))
    
    for pts in arrangements:
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        
        for outer_angle in range(0, 180, 10):
            side = min_enclosing_hexagon_side(pts_centered, outer_angle)
            if side < best_side:
                best_side = side
                best_result = (pts_centered, outer_angle)
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    pts_centered, outer_angle = best_result
    inner = []
    for x, y in pts_centered:
        inner.append([float(x), float(y), 0.0])
    
    return inner, (0.0, 0.0), float(best_side), float(outer_angle)
# EVOLVE-BLOCK-END
