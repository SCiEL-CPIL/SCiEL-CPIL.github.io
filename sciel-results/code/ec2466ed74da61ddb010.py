"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 12
DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def _polygon_overlap(v1, v2):
    """Check if two convex polygons overlap using separating axis theorem.
    Returns True if they overlap (including touching with small epsilon)."""
    eps = 1e-9
    axes = []
    for i in range(6):
        j = (i + 1) % 6
        # Edge normal
        ex = v2[j][0] - v2[i][0]
        ey = v2[j][1] - v2[i][1]
        # Normal (perpendicular)
        nx, ny = -ey, ex
        length = math.sqrt(nx * nx + ny * ny)
        if length > 0:
            axes.append((nx / length, ny / length))
    
    for i in range(6):
        j = (i + 1) % 6
        ex = v1[j][0] - v1[i][0]
        ey = v1[j][1] - v1[i][1]
        nx, ny = -ey, ex
        length = math.sqrt(nx * nx + ny * ny)
        if length > 0:
            axes.append((nx / length, ny / length))
    
    for nx, ny in axes:
        min1, max1 = float('inf'), float('-inf')
        min2, max2 = float('inf'), float('-inf')
        for x, y in v1:
            proj = x * nx + y * ny
            min1 = min(min1, proj)
            max1 = max(max1, proj)
        for x, y in v2:
            proj = x * nx + y * ny
            min2 = min(min2, proj)
            max2 = max(max2, proj)
        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def _check_inner_overlap(inner, i, j):
    """Check if inner hexagons i and j overlap."""
    x1, y1, a1 = inner[i]
    x2, y2, a2 = inner[j]
    v1 = _hex_vertices(x1, y1, 1.0, a1)
    v2 = _hex_vertices(x2, y2, 1.0, a2)
    return _polygon_overlap(v1, v2)

def _min_outer_side(inner):
    """Compute the minimum outer hexagon side length that contains all inner hexagons.
    We search over outer hexagon center and angle."""
    # This is a 3D optimization: outer_center_x, outer_center_y, outer_angle
    # For each candidate outer hexagon, check if all inner hexagons are inside.
    # The outer hexagon contains a point if the point is inside the polygon.
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon with center (cx,cy), side, angle."""
        # Transform to local coords
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        # Rotate back
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        # Check if inside unit hexagon scaled by side
        # A regular hexagon with side s has vertices at s * (cos(k*pi/3), sin(k*pi/3))
        # The hexagon is the intersection of 6 half-planes.
        # For a hexagon centered at origin with vertices at angles 0, 60, 120, 180, 240, 300 degrees:
        # The supporting lines are at distance s*cos(30) = s*sqrt(3)/2 from center
        # Actually, let's use the property that a regular hexagon with side s has
        # inradius r = s * sqrt(3)/2
        # The hexagon is the set of points where max over 6 directions of dot product <= r
        # Directions are at angles 30, 90, 150, 210, 270, 330 degrees (perpendicular to edges)
        r = side * math.sqrt(3) / 2
        for k in range(6):
            theta = math.radians(30 + k * 60)
            dot = lx * math.cos(theta) + ly * math.sin(theta)
            if dot > r + 1e-9:
                return False
        return True

    def hexagon_inside_hexagon(inner_hex, outer_hex):
        """Check if inner hexagon is inside outer hexagon."""
        ix, iy, ia = inner_hex
        ocx, ocy, oside, oangle = outer_hex
        # Check all 6 vertices of inner hexagon
        iv = _hex_vertices(ix, iy, 1.0, ia)
        for vx, vy in iv:
            if not point_in_hexagon(vx, vy, ocx, ocy, oside, oangle):
                return False
        return True

    # We need to find min side such that there exists (cx, cy, angle) where all inner hexes fit.
    # Strategy: The optimal outer hexagon center is likely near the centroid of inner hexagons.
    # We'll do a grid search over angle, and for each angle, optimize center and side.
    
    centroid_x = sum(h[0] for h in inner) / N
    centroid_y = sum(h[1] for h in inner) / N
    
    best_side = float('inf')
    
    # Grid search over outer angle
    for oangle in range(0, 60, 5):
        # For this angle, find min side by optimizing center
        # The constraint is that all inner hex vertices must be inside.
        # For a fixed outer angle and center, the required side is determined by the
        # maximum "distance" of any inner vertex from the center in the hexagonal metric.
        
        # For a hexagon with given angle, the "radius" in direction theta is r / cos(theta - edge_normal_angle)
        # Actually, let's compute for each inner vertex, what side length is needed.
        
        # For outer hexagon with center (cx,cy), angle oangle, side s:
        # A point (px,py) is inside if for all 6 edge normals, dot <= s*sqrt(3)/2
        # So s >= max over all points and all 6 directions of (dot / (sqrt(3)/2))
        
        sqrt3_2 = math.sqrt(3) / 2
        
        # For a candidate center, the required side is:
        # s >= max_{p in all inner vertices, k in 0..5} (dot(p, n_k) - dot(c, n_k)) / (sqrt(3)/2)
        # where n_k are the 6 outward normals of the outer hexagon.
        
        # To minimize s, we want to choose c to minimize this max.
        # This is a convex optimization problem (minimize max of linear functions).
        # We can solve it with a simple approach: the optimal center is where the 
        # "supporting" constraints are balanced.
        
        # Let's compute the 6 normals for the outer hexagon.
        # The outer hexagon has vertices at angles oangle, oangle+60, ..., oangle+300
        # The edge normals are at angles oangle+30, oangle+90, ..., oangle+330
        normals = []
        for k in range(6):
            theta = math.radians(oangle + 30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))
        
        # Collect all inner vertices
        all_verts = []
        for ix, iy, ia in inner:
            iv = _hex_vertices(ix, iy, 1.0, ia)
            all_verts.extend(iv)
        
        # For each vertex p and each normal n_k, the constraint is:
        # p·n_k - c·n_k <= s * sqrt(3)/2
        # => s >= (p·n_k - c·n_k) / (sqrt(3)/2)
        
        # We want to find c that minimizes max_{p, k} (p·n_k - c·n_k) / (sqrt(3)/2)
        # = (1/sqrt(3)/2) * min_c max_{p,k} (p·n_k - c·n_k)
        
        # This is equivalent to: min_c max_{p,k} (p·n_k - c·n_k)
        # Let f_k(c) = max_p (p·n_k) - c·n_k
        # We want min_c max_k f_k(c)
        
        # This is a 2D convex optimization. We can use a simple grid search or 
        # observe that the optimal c is where at least 3 constraints are active.
        
        # Precompute max_p (p·n_k) for each k
        max_dots = []
        for k in range(6):
            nx, ny = normals[k]
            max_d = max(px * nx + py * ny for px, py in all_verts)
            max_dots.append(max_d)
        
        # Now we need min_c max_k (max_dots[k] - c·n_k)
        # Let's do a fine grid search around the centroid
        cx0, cy0 = centroid_x, centroid_y
        
        # The optimal center should be within a bounded region.
        # Let's search in a box of size ~10 around centroid.
        best_s_for_angle = float('inf')
        
        # Grid search
        for cx in np.arange(cx0 - 15, cx0 + 15, 0.5):
            for cy in np.arange(cy0 - 15, cy0 + 15, 0.5):
                max_val = 0
                for k in range(6):
                    nx, ny = normals[k]
                    val = max_dots[k] - (cx * nx + cy * ny)
                    max_val = max(max_val, val)
                s_needed = max_val / sqrt3_2
                if s_needed < best_s_for_angle:
                    best_s_for_angle = s_needed
        
        if best_s_for_angle < best_side:
            best_side = best_s_for_angle
    
    return best_side

def construct():
    start_time = time.time()
    
    # Initial configuration: place hexagons in a compact pattern
    # A hexagonal lattice arrangement
    # For 12 hexagons, we can do a 3-4-3 or 2-4-4-2 pattern
    
    # Let's start with a hexagonal packing
    # Hexagon side = 1, so distance between centers in hex lattice = 2*sqrt(3)/2 = sqrt(3)
    # Actually, for regular hexagons of side 1, the distance between adjacent centers is 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)
    
    hex_spacing = math.sqrt(3)
    
    # Pattern: rows of hexagons
    # Row 0: 2 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 4 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    inner = []
    row_counts = [2, 4, 4, 2]
    row_y_offset = 1.5  # vertical distance between rows in hex packing
    
    for row_idx, count in enumerate(row_counts):
        y = (row_idx - 1.5) * row_y_offset
        x_start = -(count - 1) * hex_spacing / 2
        for i in range(count):
            x = x_start + i * hex_spacing
            inner.append([x, y, 0.0])
    
    # Evaluate initial solution
    current_side = _min_outer_side(inner)
    best_side = current_side
    best_inner = [list(h) for h in inner]
    
    # Simulated Annealing
    np.random.seed(42)
    T = 2.0
    T_min = 0.01
    cooling = 0.9995
    step_size = 1.0
    
    while time.time() - start_time < DEADLINE:
        # Pick a random hexagon to move
        i = np.random.randint(N)
        old_x, old_y, old_a = inner[i]
        
        # Random perturbation
        new_x = old_x + np.random.uniform(-step_size, step_size)
        new_y = old_y + np.random.uniform(-step_size, step_size)
        new_a = old_a + np.random.uniform(-15, 15)
        
        # Check if new position causes overlap with other inner hexagons
        inner[i] = [new_x, new_y, new_a]
        overlap = False
        for j in range(N):
            if j != i:
                if _check_inner_overlap(inner, i, j):
                    overlap = True
                    break
        
        if overlap:
            # Reject if overlapping
            inner[i] = [old_x, old_y, old_a]
        else:
            new_side = _min_outer_side(inner)
            if new_side < current_side or np.random.rand() < math.exp(-(new_side - current_side) / T):
                current_side = new_side
                if new_side < best_side:
                    best_side = new_side
                    best_inner = [list(h) for h in inner]
            else:
                inner[i] = [old_x, old_y, old_a]
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with higher T occasionally
            if np.random.rand() < 0.1:
                T = 1.0
        
        # Adapt step size
        if np.random.rand() < 0.01:
            step_size = max(0.1, min(5.0, step_size * np.random.uniform(0.8, 1.2)))
    
    # Use best solution found
    inner = best_inner
    side = best_side
    centroid_x = sum(h[0] for h in inner) / N
    centroid_y = sum(h[1] for h in inner) / N
    
    return inner, (centroid_x, centroid_y), side, 0.0
# EVOLVE-BLOCK-END
