import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 12
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_cluster_vertices(row_counts, inner_angle_rad):
        verts = []
        dy = 1.5
        for r, count in enumerate(row_counts):
            y = r * dy
            if r % 2 == 0:
                xs = np.arange(-count + 1, count, 2) * (SQRT3 / 2)
            else:
                xs = (np.arange(-count + 1, count, 2) + 1) * (SQRT3 / 2)
                
            for x in xs:
                v = hex_vertices(x, y, 1.0, inner_angle_rad)
                verts.append(v)
        
        all_verts = np.vstack(verts)
        center = all_verts.mean(axis=0)
        all_verts = all_verts - center
        return all_verts

    def min_enclosing_hexagon_side(verts, theta_rad):
        normals_angles = np.radians(30) + np.radians(np.arange(0, 360, 60)) + theta_rad
        normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
        projections = verts @ normals.T
        max_proj = projections.max(axis=0)
        max_dist = np.max(max_proj)
        s = max_dist * 2 / SQRT3
        return s

    topologies = [
        [3, 4, 3, 2],
        [2, 3, 4, 3],
        [1, 3, 4, 3, 1],
        [2, 4, 4, 2],
        [1, 2, 3, 4, 2],
        [4, 4, 4],
        [3, 3, 3, 3],
        [2, 2, 4, 4],
        [1, 4, 4, 3],
        [3, 4, 4, 1],
        [2, 5, 5],
        [5, 5, 2],
    ]
    
    best_s = np.inf
    best_row_counts = None
    best_inner_angle_rad = 0.0
    best_theta = 0.0

    for row_counts in topologies:
        if sum(row_counts) != 12:
            continue
            
        for inner_angle_deg in [0, 30]:
            inner_angle_rad = np.radians(inner_angle_deg)
            verts = get_cluster_vertices(row_counts, inner_angle_rad)
            
            try:
                res = minimize_scalar(
                    lambda t: min_enclosing_hexagon_side(verts, t),
                    bounds=(0, np.radians(30)),
                    method='bounded',
                    options={'xatol': 1e-6}
                )
                if res.success:
                    s = res.fun
                    if s < best_s:
                        best_s = s
                        best_row_counts = row_counts
                        best_inner_angle_rad = inner_angle_rad
                        best_theta = res.x
            except Exception as e:
                pass

    if best_row_counts is None:
        best_row_counts = [3, 4, 3, 2]
        best_inner_angle_rad = 0.0
        best_theta = 0.0
        verts = get_cluster_vertices(best_row_counts, best_inner_angle_rad)
        best_s = min_enclosing_hexagon_side(verts, best_theta)

    centers = []
    dy = 1.5
    for r, count in enumerate(best_row_counts):
        y = r * dy
        if r % 2 == 0:
            xs = np.arange(-count + 1, count, 2) * (SQRT3 / 2)
        else:
            xs = (np.arange(-count + 1, count, 2) + 1) * (SQRT3 / 2)
        for x in xs:
            centers.append([x, y])
            
    centers = np.array(centers)
    center_of_cluster = centers.mean(axis=0)
    centers = centers - center_of_cluster
    
    inner = []
    for i in range(N):
        x, y = centers[i]
        angle_deg = np.degrees(best_inner_angle_rad)
        inner.append([x, y, angle_deg])
        
    outer_center = (0.0, 0.0)
    outer_side = best_s
    outer_angle_deg = np.degrees(best_theta)
    
    return inner, outer_center, outer_side, outer_angle_deg
