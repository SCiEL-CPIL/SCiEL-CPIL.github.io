"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def _min_enclosing_hexagon(points, angle_deg):
    """Find minimum side length of a regular hexagon (with given rotation and center at origin) that contains all points."""
    a = np.radians(angle_deg)
    normals = []
    for k in range(3):
        theta = a + k * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    projections = points @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    required = np.maximum(max_proj, -min_proj)
    S_min = np.max(required) * 2 / np.sqrt(3)
    return S_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_side = np.inf
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try hexagonal lattice arrangements
    configs = []
    
    # Center + 6 neighbors + 5 from second ring
    centers = [[0, 0]]
    for k in range(6):
        angle = k * np.pi / 3
        centers.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
    count = 1
    for k in range(6):
        angle = k * np.pi / 3 + np.pi / 6
        centers.append([2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)])
        count += 1
        if count == 12:
            break
    configs.append(centers)
    
    # Grid arrangements
    for rows in [2, 3, 4, 6]:
        for cols in [2, 3, 4, 6]:
            if rows * cols != 12:
                continue
            for offset in [0, 0.5]:
                centers = []
                for r in range(rows):
                    for c in range(cols):
                        x = c * np.sqrt(3) + (offset * np.sqrt(3) if r % 2 else 0)
                        y = r * 1.5
                        centers.append([x, y])
                configs.append(centers)
    
    # For each config, find optimal outer hexagon
    for centers in configs:
        if time.time() > deadline:
            break
        centers = np.array(centers)
        
        # Try different rotations for inner hexagons
        for inner_angle in [0, 30]:
            all_verts = []
            for cx, cy in centers:
                verts = _hex_vertices(cx, cy, 1.0, inner_angle)
                all_verts.append(verts)
            all_verts = np.vstack(all_verts)
            
            # Find best outer angle
            best_S = np.inf
            best_ao = 0
            for ao in np.arange(0, 60, 5):
                S = _min_enclosing_hexagon(all_verts, ao)
                if S < best_S:
                    best_S = S
                    best_ao = ao
            for ao in np.arange(best_ao - 5, best_ao + 5, 0.5):
                S = _min_enclosing_hexagon(all_verts, ao)
                if S < best_S:
                    best_S = S
                    best_ao = ao
            for ao in np.arange(best_ao - 0.5, best_ao + 0.5, 0.05):
                S = _min_enclosing_hexagon(all_verts, ao)
                if S < best_S:
                    best_S = S
                    best_ao = ao
            
            if best_S < best_side:
                best_side = best_S
                best_inner = [[float(cx), float(cy), float(inner_angle)] for cx, cy in centers]
                best_outer_angle = best_ao
                # Center outer hex at centroid
                cx = np.mean(all_verts[:, 0])
                cy = np.mean(all_verts[:, 1])
                best_outer_center = (float(cx), float(cy))
    
    if best_inner is None:
        centers = [[k * np.sqrt(3), 0.0] for k in range(N)]
        best_inner = [[float(cx), float(cy), 0.0] for cx, cy in centers]
        all_verts = np.vstack([_hex_vertices(cx, cy, 1.0, 0) for cx, cy in centers])
        best_side = _min_enclosing_hexagon(all_verts, 0)
        best_outer_center = (float(np.mean(all_verts[:, 0])), float(np.mean(all_verts[:, 1])))
        best_outer_angle = 0.0
    
    return best_inner, best_outer_center, float(best_side), float(best_outer_angle)

# EVOLVE-BLOCK-END
