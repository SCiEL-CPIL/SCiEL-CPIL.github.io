"""
Optimized packing of 11 unit regular hexagons into a minimum-size regular hexagon.
Strategy:
1. Construct a specific topology: 1 center, 6 neighbors, 4 outer.
2. Refine the outer 4 hexagons to minimize the cluster's bounding box.
3. Optimize the global rotation of the cluster relative to the container.
4. Polish the container center and rotation for final precision.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    outer_center: (x, y)
    outer_angle_deg: rotation of the container in degrees
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    # Normals in outer frame (30 + i*60 degrees)
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        # Transform to outer local frame
        local = (verts - np.array([ox, oy])) @ R.T
        # Project onto all 6 normals
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    return 2.0 * max_proj / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (lists of (x,y))."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
    for ax in axes:
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """Quadratic penalty when center distances fall below min_dist."""
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Keep 10s safety margin

    # --- Step 1: Construct a specific topology for 11 hexagons ---
    # 1 center (0,0), 6 neighbors, 4 outer.
    # Axial coordinates (q, r) for hex grid.
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 candidates: We want to minimize the bounding box.
    # A known compact configuration is to place the extra 4 in the "gaps" of the outer ring.
    # Let's define initial positions in a local frame where the cluster is centered at (0,0).
    
    # Hex grid spacing for side 1:
    # Distance between centers is SQRT3.
    # Angle between axes: 60 deg.
    # Let's use axial coordinates to generate positions.
    # pos = (1.5 * q, SQRT3 * (r + q/2))
    
    # Topology:
    # 0: (0,0)
    # 1: (1,0)
    # 2: (0,1)
    # 3: (-1,1)
    # 4: (-1,0)
    # 5: (0,-1)
    # 6: (1,-1)
    # The 4 extra hexagons can be placed at:
    # 7: (2,-1)  (adjacent to 1 and 6)
    # 8: (-2,1)  (adjacent to 3 and 4)
    # 9: (1,1)   (adjacent to 1 and 2)
    # 10: (-1,-1) (adjacent to 4 and 5)
    # This forms a roughly circular shape.
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (-2, 1), (1, 1), (-1, -1)
    ]
    
    initial_positions = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        initial_positions.append([x, y, 0.0])
        
    # Center the cluster
    cx_cluster = sum(p[0] for p in initial_positions) / N
    cy_cluster = sum(p[1] for p in initial_positions) / N
    for p in initial_positions:
        p[0] -= cx_cluster
        p[1] -= cy_cluster

    # --- Step 2: Refine the cluster shape ---
    # We treat the 11 hexagons as a single object. We will optimize the positions of the 4 outer hexagons
    # to minimize the "width" of the cluster in all 6 directions, while maintaining non-overlap.
    # Actually, a simpler and more robust approach: 
    # Optimize the positions of ALL 11 hexagons relative to each other, with a strong penalty for overlap,
    # and minimize the bounding box size of the cluster itself.
    
    def cluster_bounding_box_size(params):
        # params: 3*N floats (x, y, angle for each hex)
        inner = []
        for i in range(N):
            inner.append([params[i*3], params[i*3+1], params[i*3+2]])
        
        # Compute bounding box size relative to cluster center (0,0)
        # We want to minimize the max projection in 6 directions.
        max_proj = 0.0
        for (ix, iy, iang) in inner:
            verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
            # Project onto 6 axes (aligned with hex normals)
            for k in range(6):
                angle = math.radians(30 + k * 60)
                nx = math.cos(angle)
                ny = math.sin(angle)
                proj = np.max(verts[:, 0] * nx + verts[:, 1] * ny)
                if proj > max_proj:
                    max_proj = proj
        # The side length of the container for this cluster (if centered at 0,0 and aligned) is:
        # side = 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    def obj_cluster(params):
        inner = []
        for i in range(N):
            inner.append([params[i*3], params[i*3+1], params[i*3+2]])
        
        # Penalty for overlap
        pen = overlap_penalty(inner, min_dist=SQRT3 - 0.01)
        
        # Objective: minimize bounding box size + penalty
        size = cluster_bounding_box_size(params)
        return size + 100.0 * pen

    # Initial parameters
    p0 = np.zeros(N * 3)
    for i, (x, y, a) in enumerate(initial_positions):
        p0[i*3] = x
        p0[i*3+1] = y
        p0[i*3+2] = a

    # Optimize cluster shape
    # Use Nelder-Mead for local refinement
    try:
        if time.time() < deadline - 60:
            res_cluster = minimize(obj_cluster, p0, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 5000})
            if res_cluster.fun < obj_cluster(p0):
                p_opt = res_cluster.x
            else:
                p_opt = p0
        else:
            p_opt = p0
    except Exception:
        p_opt = p0

    # Extract optimized cluster positions
    optimized_cluster = []
    for i in range(N):
        optimized_cluster.append([float(p_opt[i*3]), float(p_opt[i*3+1]), float(p_opt[i*3+2])])
        
    # Verify disjointness
    if not check_disjoint(optimized_cluster, eps=1e-5):
        # Fallback to initial lattice if optimization caused overlap
        optimized_cluster = initial_positions

    # --- Step 3: Optimize global rotation and container center ---
    # The cluster is now fixed in shape. We need to find the best rotation and translation
    # to fit it into the container.
    
    def obj_container(params):
        # params: [cx, cy, rot_deg]
        cx, cy, rot = params
        # Apply rotation to cluster
        rot_rad = math.radians(rot)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        
        rotated_cluster = []
        for (x, y, a) in optimized_cluster:
            # Rotate point (x,y) around origin
            rx = x * cos_r - y * sin_r
            ry = x * sin_r + y * cos_r
            # Rotate hexagon angle
            new_a = a + rot
            rotated_cluster.append([rx, ry, new_a])
        
        # Translate to (cx, cy)
        translated_cluster = []
        for (x, y, a) in rotated_cluster:
            translated_cluster.append([x + cx, y + cy, a])
            
        # Compute required side length
        return compute_required_side_fast(translated_cluster, (0, 0), 0.0)

    # Initial guess: center at (0,0), rotation 0
    p_init = np.array([0.0, 0.0, 0.0])
    
    best_container_params = p_init
    best_side = float('inf')
    
    # Multi-start for rotation
    rot_starts = [0, 15, 30, 45, 60, 90, 120, 150, 180]
    
    for r0 in rot_starts:
        if time.time() > deadline - 20:
            break
        p_try = np.array([0.0, 0.0, float(r0)])
        try:
            res_c = minimize(obj_container, p_try, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000})
            if res_c.fun < best_side:
                best_side = res_c.fun
                best_container_params = res_c.x
        except Exception:
            pass

    # Extract best container params
    cx_best, cy_best, rot_best = best_container_params
    
    # Apply final rotation and translation to cluster
    final_inner = []
    rot_rad = math.radians(rot_best)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    
    for (x, y, a) in optimized_cluster:
        rx = x * cos_r - y * sin_r
        ry = x * sin_r + y * cos_r
        new_a = a + rot_best
        final_inner.append([rx + cx_best, ry + cy_best, new_a])
        
    # Final side length
    final_side = compute_required_side_fast(final_inner, (0, 0), 0.0)
    
    # --- Step 4: Final Polish ---
    # Polish the container center and rotation with the fixed inner cluster
    def obj_polish(params):
        cx, cy, rot = params
        rot_rad = math.radians(rot)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        
        rotated_cluster = []
        for (x, y, a) in optimized_cluster:
            rx = x * cos_r - y * sin_r
            ry = x * sin_r + y * cos_r
            new_a = a + rot
            rotated_cluster.append([rx, ry, new_a])
            
        translated_cluster = []
        for (x, y, a) in rotated_cluster:
            translated_cluster.append([x + cx, y + cy, a])
            
        return compute_required_side_fast(translated_cluster, (0, 0), 0.0)

    p_polish_init = np.array([cx_best, cy_best, rot_best])
    try:
        if time.time() < deadline - 5:
            res_polish = minimize(obj_polish, p_polish_init, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 10000})
            if res_polish.fun < final_side:
                final_side = res_polish.fun
                cx_best, cy_best, rot_best = res_polish.x
                
                # Recalculate final_inner
                final_inner = []
                rot_rad = math.radians(rot_best)
                cos_r = math.cos(rot_rad)
                sin_r = math.sin(rot_rad)
                for (x, y, a) in optimized_cluster:
                    rx = x * cos_r - y * sin_r
                    ry = x * sin_r + y * cos_r
                    new_a = a + rot_best
                    final_inner.append([rx + cx_best, ry + cy_best, new_a])
    except Exception:
        pass

    # Return format: (inner, center, side, angle)
    # inner: list of (x, y, angle_degrees)
    # center: (x, y)
    # side: float
    # angle: float (rotation of outer hexagon)
    
    # Note: The container is centered at (0,0) and rotated by 0 in the calculation, 
    # but the inner hexagons are rotated by rot_best and translated.
    # The "angle" of the outer hexagon in the return value should be 0.0 because 
    # we aligned the container to the axes in our calculation.
    # However, the problem allows the outer hexagon to rotate. 
    # Our calculation effectively rotated the inner hexagons to align with a fixed outer hexagon.
    # So the outer hexagon angle is 0.
    
    return final_inner, (0.0, 0.0), final_side, 0.0

# EVOLVE-BLOCK-END
