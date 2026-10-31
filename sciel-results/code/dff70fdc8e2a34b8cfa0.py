"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 140.0  # safety margin before 150s kill

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def hexagons_overlap(v1, v2, eps=1e-9):
    """
    Check if two hexagons (defined by their 6 vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap.
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = float('inf')
        max1 = float('-inf')
        min2 = float('inf')
        max2 = float('-inf')
        
        for p in v1:
            proj = p[0] * axis[0] + p[1] * axis[1]
            if proj < min1: min1 = proj
            if proj > max1: max1 = proj
            
        for p in v2:
            proj = p[0] * axis[0] + p[1] * axis[1]
            if proj < min2: min2 = proj
            if proj > max2: max2 = proj
            
        # Check for gap
        if max1 < min2 - eps or max2 < min1 - eps:
            return False
    return True

def is_layout_valid(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """
    Check if all inner hexagons fit inside the outer hexagon and don't overlap.
    """
    # Check containment
    outer_verts = get_hex_vertices(outer_center[0], outer_center[1], outer_side, math.radians(outer_angle_deg))
    
    # Precompute outer axes
    outer_axes = []
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            outer_axes.append((normal[0] / norm, normal[1] / norm))
    
    inner_verts_list = []
    for i, (x, y, ang) in enumerate(inner):
        v = get_hex_vertices(x, y, 1.0, math.radians(ang))
        inner_verts_list.append(v)
        
        # Check containment using projection
        # For each outer normal, the projection of any inner vertex must be within
        # the projection range of the outer hexagon
        for axis in outer_axes:
            min_outer = float('inf')
            max_outer = float('-inf')
            for p in outer_verts:
                proj = p[0] * axis[0] + p[1] * axis[1]
                if proj < min_outer: min_outer = proj
                if proj > max_outer: max_outer = proj
            
            for p in v:
                proj = p[0] * axis[0] + p[1] * axis[1]
                if proj < min_outer - eps or proj > max_outer + eps:
                    return False
    
    # Check pairwise overlaps
    for i in range(len(inner_verts_list)):
        for j in range(i + 1, len(inner_verts_list)):
            if hexagons_overlap(inner_verts_list[i], inner_verts_list[j], eps=1e-4):
                return False
    
    return True

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
    """
    Place N unit hexagons inside a regular hexagon, minimizing the outer side length.
    Returns: (inner, center, side, angle)
    inner: list of [x, y, angle_degrees]
    center: (x, y) of outer hexagon
    side: side length of outer hexagon
    angle: rotation of outer hexagon in degrees
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
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
    best_inner = [list(x) for x in initial_inner]
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
            
        if time.time() > deadline - 20:
            break

    # 4. Local search on inner positions to tighten packing
    # This allows the hexagons to move slightly to reduce the bounding hexagon size
    if time.time() < deadline - 30:
        current_inner = [list(x) for x in best_inner]
        current_center = best_outer_center
        current_angle = best_outer_angle
        current_side = best_side
        
        step_size = 0.05
        np.random.seed(42)
        
        # We will do a simple hill climbing on the inner positions
        # and re-optimize the outer container for each move
        max_iter = 5000
        iter_count = 0
        
        while time.time() < deadline - 10 and iter_count < max_iter:
            iter_count += 1
            
            # Pick a random inner hexagon and perturb it
            idx = np.random.randint(N)
            orig_x, orig_y, orig_ang = current_inner[idx]
            
            # Random perturbation
            new_x = orig_x + np.random.uniform(-step_size, step_size)
            new_y = orig_y + np.random.uniform(-step_size, step_size)
            new_ang = orig_ang + np.random.uniform(-5, 5)
            
            current_inner[idx] = [new_x, new_y, new_ang]
            
            # Check validity (overlaps)
            valid = True
            verts_list = [get_hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in current_inner]
            for i in range(N):
                for j in range(i + 1, N):
                    if hexagons_overlap(verts_list[i], verts_list[j], eps=1e-4):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Re-optimize outer container
                # Quick local search on center and angle
                best_local_side = current_side
                best_local_center = current_center
                best_local_angle = current_angle
                
                # Try small perturbations of center and angle
                for dcx in np.linspace(-0.1, 0.1, 3):
                    for dcy in np.linspace(-0.1, 0.1, 3):
                        for da in np.linspace(-1, 1, 3):
                            nc = (current_center[0] + dcx, current_center[1] + dcy)
                            na = current_angle + da
                            s = compute_required_side(current_inner, nc, na)
                            if s < best_local_side:
                                best_local_side = s
                                best_local_center = nc
                                best_local_angle = na
                
                if best_local_side < current_side - 1e-6:
                    current_side = best_local_side
                    current_center = best_local_center
                    current_angle = best_local_angle
                    
                    if current_side < best_side - 1e-6:
                        best_side = current_side
                        best_inner = [list(x) for x in current_inner]
                        best_outer_center = current_center
                        best_outer_angle = current_angle
                        step_size = max(step_size * 0.95, 0.005)
                else:
                    # Revert
                    current_inner[idx] = [orig_x, orig_y, orig_ang]
                    step_size = max(step_size * 0.9, 0.005)
            else:
                # Revert
                current_inner[idx] = [orig_x, orig_y, orig_ang]
                step_size = max(step_size * 0.9, 0.005)

    # 5. Final refinement of the best found solution
    if time.time() < deadline - 5:
        # Fine-tune the outer container for the best inner configuration
        current_inner = best_inner
        current_center = best_outer_center
        current_angle = best_outer_angle
        
        step = 0.05
        for _ in range(3):
            improved = False
            for dx in np.linspace(-step, step, 5):
                for dy in np.linspace(-step, step, 5):
                    for da in np.linspace(-step, step, 5):
                        oc = (current_center[0] + dx, current_center[1] + dy)
                        oa = current_angle + da
                        side = compute_required_side(current_inner, oc, oa)
                        if side < best_side - 1e-6:
                            best_side = side
                            best_outer_center = oc
                            best_outer_angle = oa
                            current_center = oc
                            current_angle = oa
                            improved = True
            if not improved:
                break
            step *= 0.5

    # Add tiny epsilon to side to ensure validity
    best_side += 1e-9

    return best_inner, best_outer_center, best_side, best_outer_angle
