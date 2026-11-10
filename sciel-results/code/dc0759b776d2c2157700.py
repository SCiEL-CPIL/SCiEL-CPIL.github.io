import numpy as np
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, s) where inner is (12,3), outer is (3,), s is outer side length.
    """
    deadline = time.time() + 600.0
    
    sqrt3 = np.sqrt(3.0)
    gap = 1e-5  # tiny gap to ensure separation
    
    # Hexagonal cluster: 1 center + 6 first ring + 5 second ring
    # Spacing increased by gap to ensure no contact
    spacing1 = sqrt3 + gap  # first ring distance
    spacing2 = 2 * sqrt3 + gap  # second ring distance
    
    centers = [[0.0, 0.0]]
    
    # First ring: 6 hexagons
    for i in range(6):
        angle = np.radians(60 * i)
        cx = spacing1 * np.cos(angle)
        cy = spacing1 * np.sin(angle)
        centers.append([cx, cy])
    
    # Second ring: 5 hexagons at offset angles
    second_ring_angles = [30, 90, 150, 210, 270]
    for angle_deg in second_ring_angles:
        angle = np.radians(angle_deg)
        cx = spacing2 * np.cos(angle)
        cy = spacing2 * np.sin(angle)
        centers.append([cx, cy])
    
    centers = np.array(centers)
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = centers[:, 0]
    inner_hex_data[:, 1] = centers[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    def get_hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            a = angle_rad + np.radians(60 * i)
            x = cx + side * np.cos(a)
            y = cy + side * np.sin(a)
            vertices.append([x, y])
        return np.array(vertices)
    
    all_vertices = []
    for i in range(12):
        cx, cy, ang = inner_hex_data[i]
        verts = get_hexagon_vertices(cx, cy, 1.0, ang)
        all_vertices.extend(verts.tolist())
    all_vertices = np.array(all_vertices)
    
    def point_in_hexagon(px, py, s, angle_deg=0.0):
        angle_rad = np.radians(angle_deg)
        for i in range(6):
            a1 = angle_rad + np.radians(60 * i)
            a2 = angle_rad + np.radians(60 * (i + 1))
            p1 = np.array([s * np.cos(a1), s * np.sin(a1)])
            p2 = np.array([s * np.cos(a2), s * np.sin(a2)])
            edge = p2 - p1
            point = np.array([px, py]) - p1
            cross = edge[0] * point[1] - edge[1] * point[0]
            if cross < -1e-6:
                return False
        return True
    
    lo = np.sqrt(12.0)
    hi = 10.0
    
    for _ in range(100):
        mid = (lo + hi) / 2
        all_inside = True
        for v in all_vertices:
            if not point_in_hexagon(v[0], v[1], mid, 0.0):
                all_inside = False
                break
        if all_inside:
            hi = mid
        else:
            lo = mid
        if time.time() > deadline - 30:
            break
    
    s = hi + 1e-8
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s
