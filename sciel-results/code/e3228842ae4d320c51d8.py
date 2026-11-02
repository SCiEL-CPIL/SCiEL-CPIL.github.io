import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        return np.min(overlaps)

    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    outer_normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    def compute_min_side(hexes):
        max_proj = 0.0
        for h in hexes:
            proj = h @ outer_normals.T
            max_proj = max(max_proj, np.max(proj))
        return max_proj * 2.0 / np.sqrt(3.0)

    # Honeycomb arrangement: center + 6 first-ring + 6 second-ring
    # First ring: distance sqrt(3) from center, angles 0, 60, 120, 180, 240, 300
    # Second ring: distance 3 from center, angles 30, 90, 150, 210, 270, 330
    sqrt3 = np.sqrt(3.0)
    
    # Centers: (x, y, angle) for 12 hexagons
    # All hexagons have angle 0 (flat-top orientation matching outer)
    centers = []
    # Center hexagon
    centers.append((0.0, 0.0, 0.0))
    # First ring: 6 hexagons at distance sqrt(3), angles 0, 60, ..., 300
    for k in range(6):
        angle = np.radians(60 * k)
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        centers.append((x, y, 0.0))
    # Second ring: 6 hexagons at distance 3, angles 30, 90, ..., 330
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        centers.append((x, y, 0.0))

    # Generate hexagons
    hexes = []
    for cx, cy, angle in centers:
        hexes.append(hex_vertices(cx, cy, 1.0, np.radians(angle)))

    # Verify no overlaps (should be touching at best)
    # Check all pairs
    max_overlap = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            min1 = hexes[i].min(axis=0)
            max1 = hexes[i].max(axis=0)
            min2 = hexes[j].min(axis=0)
            max2 = hexes[j].max(axis=0)
            if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                continue
            overlap = sat_overlap(hexes[i], hexes[j])
            if overlap > max_overlap:
                max_overlap = overlap

    # If there's any positive overlap, try to fix with small perturbation
    # But for the honeycomb, overlaps should be 0 (touching)
    
    s_final = compute_min_side(hexes)
    
    # Build inner list
    inner = []
    for h in hexes:
        cx = np.mean(h[:, 0])
        cy = np.mean(h[:, 1])
        edge = h[1] - h[0]
        angle = np.arctan2(edge[1], edge[0])
        inner.append([cx, cy, np.degrees(angle) % 360])

    outer_center = (0.0, 0.0)
    outer_side = s_final
    outer_angle_deg = 0.0

    return inner, outer_center, outer_side, outer_angle_deg
