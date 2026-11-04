import numpy as np
import math

N = 11

def construct():
    sqrt3 = math.sqrt(3)
    
    # Place 11 unit hexagons in a compact hexagonal-lattice arrangement (flat-top)
    # Rows: 3, 4, 3, 1 = 11 hexagons
    h_spacing = sqrt3      # horizontal spacing in same row
    v_spacing = 1.5        # vertical spacing between rows
    h_offset = sqrt3 / 2   # horizontal offset between rows
    
    inner = []
    # Row 0 (y=0): 3 hexagons
    for i in range(3):
        x = (i - 1) * h_spacing
        inner.append([x, 0.0, 0.0])
    
    # Row 1 (y=v_spacing): 4 hexagons, offset by h_offset
    for i in range(4):
        x = (i - 1.5) * h_spacing + h_offset
        inner.append([x, v_spacing, 0.0])
    
    # Row 2 (y=2*v_spacing): 3 hexagons
    for i in range(3):
        x = (i - 1) * h_spacing
        inner.append([x, 2 * v_spacing, 0.0])
    
    # Row 3 (y=3*v_spacing): 1 hexagon
    inner.append([0.0, 3 * v_spacing, 0.0])
    
    pts = np.array([[p[0], p[1]] for p in inner])
    centroid = np.mean(pts, axis=0)
    
    def min_L_for_center_rotation(cx, cy, theta, pts):
        max_proj = 0.0
        for k in range(6):
            angle = theta + k * math.pi / 3
            nx = math.cos(angle)
            ny = math.sin(angle)
            dists = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(dists))
        return 2.0 * max_proj / sqrt3
    
    best_L = float('inf')
    best_cx, best_cy, best_theta = centroid[0], centroid[1], 0.0
    
    # Coarse search
    for theta_deg in range(0, 60, 3):
        theta = math.radians(theta_deg)
        for cx in np.linspace(centroid[0] - 1.5, centroid[0] + 1.5, 15):
            for cy in np.linspace(centroid[1] - 1.5, centroid[1] + 1.5, 15):
                L = min_L_for_center_rotation(cx, cy, theta, pts)
                if L < best_L:
                    best_L = L
                    best_cx, best_cy, best_theta = cx, cy, theta
    
    # Fine search
    for theta_deg in np.linspace(math.degrees(best_theta) - 3, math.degrees(best_theta) + 3, 15):
        theta = math.radians(theta_deg)
        for cx in np.linspace(best_cx - 0.3, best_cx + 0.3, 15):
            for cy in np.linspace(best_cy - 0.3, best_cy + 0.3, 15):
                L = min_L_for_center_rotation(cx, cy, theta, pts)
                if L < best_L:
                    best_L = L
                    best_cx, best_cy, best_theta = cx, cy, theta
    
    return inner, (best_cx, best_cy), best_L, math.degrees(best_theta)
