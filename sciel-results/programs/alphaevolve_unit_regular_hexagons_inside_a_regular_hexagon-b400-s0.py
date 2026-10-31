import math
import time
import numpy as np

N = 11
SQRT3 = math.sqrt(3)

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Define 11 hexagons on a hexagonal lattice
    # Axial coordinates (q, r)
    # Center (0,0)
    # Ring 1 (6 hexagons)
    # Ring 2 partial (4 hexagons)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    # Ensure we have exactly N
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    elif len(axial_coords) < N:
        # Fallback: add more from ring 2 (should not happen for N=11)
        pass

    # 2. Convert axial to Cartesian
    # Flat-Top lattice:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # Add small epsilon to scale to ensure disjointness in floating point checks
    epsilon = 1e-6
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 3. Optimize Outer Hexagon
    # The cluster is not perfectly symmetric, so we need to find the best rotation and center.
    
    # Coarse search over rotation
    for outer_angle_deg in range(0, 360, 5):
        # For each rotation, find the best center via local search
        # Initial center guess: centroid of the cluster
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        current_cx = cx_avg
        current_cy = cy_avg
        
        # Local search on center
        step = 0.5
        for _ in range(3): # 3 iterations of refinement
            improved = False
            for dx in np.linspace(-step, step, 5):
                for dy in np.linspace(-step, step, 5):
                    oc = (current_cx + dx, current_cy + dy)
                    side = compute_required_side(initial_inner, oc, outer_angle_deg)
                    if side < best_side - 1e-5:
                        best_side = side
                        best_outer_center = oc
                        best_outer_angle = outer_angle_deg
                        current_cx = oc[0]
                        current_cy = oc[1]
                        improved = True
            if not improved:
                break
            step *= 0.5
            
        if time.time() > deadline:
            break

    # Final refinement with finer steps if time permits
    if time.time() < deadline - 10:
        current_cx = best_outer_center[0]
        current_cy = best_outer_center[1]
        current_angle = best_outer_angle
        
        step = 0.1
        for _ in range(3):
            improved = False
            for dx in np.linspace(-step, step, 5):
                for dy in np.linspace(-step, step, 5):
                    for da in np.linspace(-step, step, 5):
                        oc = (current_cx + dx, current_cy + dy)
                        oa = current_angle + da
                        side = compute_required_side(initial_inner, oc, oa)
                        if side < best_side - 1e-5:
                            best_side = side
                            best_outer_center = oc
                            best_outer_angle = oa
                            current_cx = oc[0]
                            current_cy = oc[1]
                            current_angle = oa
                            improved = True
            if not improved:
                break
            step *= 0.5

    # Add tiny epsilon to side to ensure validity
    best_side += 1e-9

    return initial_inner, best_outer_center, best_side, best_outer_angle
