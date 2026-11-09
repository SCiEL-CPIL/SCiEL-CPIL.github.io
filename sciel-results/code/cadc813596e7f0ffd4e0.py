"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def check_overlap(centers, angles, eps=1e-6):
        """Check if any two inner hexagons overlap."""
        for i in range(N):
            for j in range(i + 1, N):
                # Quick distance check: if centers are far apart, no overlap
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx*dx + dy*dy
                # Max distance between any two points in two unit hexagons is 4 (vertex to vertex if aligned)
                # Min distance to avoid overlap is 2.0 (side to side if aligned)
                # If dist > 4.0, definitely no overlap.
                if dist_sq > 16.0:
                    continue
                
                vi = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
                vj = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                
                # Check if any vertex of i is inside j, or vice versa
                # Simplified: Check if any edge intersects. 
                # For robustness and speed in local search, we can check if distance between centers is < 2.0 - eps
                # This is a conservative check that might reject some valid rotated configurations
                # but guarantees no overlap for the common case.
                # Let's use a slightly more precise check: 
                # If dist < 2.0, they *might* overlap. We need to verify.
                # If dist >= 2.0 + eps, they definitely do not overlap? No, if rotated, 
                # the "width" is smaller. The minimum width of a unit hexagon is sqrt(3) ~ 1.732.
                # So if dist >= sqrt(3) + eps, they don't overlap? 
                # Actually, the minimum distance between two disjoint unit hexagons is sqrt(3) (side-to-side).
                # So if dist > 2.0, they are definitely disjoint.
                # If dist < 2.0, they might overlap.
                # To be safe and fast, we will assume they overlap if dist < 1.9.
                # If 1.9 <= dist < 2.0, we do a detailed check.
                
                if dist_sq < (1.9)**2:
                    return False
                
                # Detailed check for close distances
                # Check if any vertex of i is inside polygon j
                def point_in_hex(px, py, poly):
                    # Ray casting or winding number. 
                    # For a convex polygon, we can check if point is on same side of all edges.
                    # Or simply: check if point is inside the bounding box and then check triangles.
                    # Simpler: Use shapely if available? No, stdlib only.
                    # Let's use the fact that hexagon is convex.
                    # A point is inside if it is to the left of all edges (or right, depending on orientation).
                    # But vertices are generated in CCW order?
                    # cos/sin with increasing angle is CCW.
                    # So cross product of edge vector and point vector should be >= 0 (or <= 0).
                    inside = True
                    for k in range(6):
                        x1, y1 = poly[k]
                        x2, y2 = poly[(k+1)%6]
                        # Vector edge
                        ex, ey = x2-x1, y2-y1
                        # Vector to point
                        px1, py1 = px-x1, py-y1
                        cross = ex*py1 - ey*px1
                        if cross < -eps: # If strictly to the right (outside)
                            inside = False
                            break
                    return inside

                # Check vertices of i against j
                for vx, vy in vi:
                    if point_in_hex(vx, vy, vj):
                        return False
                # Check vertices of j against i
                for vx, vy in vj:
                    if point_in_hex(vx, vy, vi):
                        return False
        return True

    def enclosing_side_length(centers, angles, outer_angle=0.0):
        """
        Compute the minimal side length of a regular hexagon (centered at origin, 
        with given rotation) that contains all inner unit hexagons.
        """
        # 6 outward normal directions for outer hexagon with rotation outer_angle
        normals = []
        for k in range(6):
            angle = math.radians(outer_angle + 30 + k * 60)
            normals.append((math.cos(angle), math.sin(angle)))
        
        max_extent = 0.0
        for i in range(N):
            cx, cy = centers[i]
            inner_angle_rad = math.radians(angles[i])
            
            for nk in range(6):
                nx, ny = normals[nk]
                n_angle = math.atan2(ny, nx)
                max_dot = 0.0
                for j in range(6):
                    v_angle = inner_angle_rad + j * math.pi / 3.0
                    dot_val = math.cos(v_angle - n_angle)
                    if dot_val > max_dot:
                        max_dot = dot_val
                
                extent = cx * nx + cy * ny + max_dot
                if extent > max_extent:
                    max_extent = extent
        
        return 2.0 * max_extent / sqrt3

    # Initialize with a valid hexagonal lattice configuration
    # Hex grid spacing is 2.0 for side 1 hexagons to touch.
    # We pick 12 positions from a hex grid centered at 0,0.
    # Coordinates in axial system (q, r) -> Cartesian (x, y)
    # x = 2 * q + sqrt3 * r
    # y = sqrt3 * (r + 0.5 * q) ? No, standard hex grid mapping:
    # Let's use the neighbor vectors directly.
    
    # Neighbors of (0,0) in hex grid (side 1, touching):
    # dist = 2.0
    # Angles: 0, 60, 120, 180, 240, 300
    # But wait, if side is 1, the distance between centers of touching hexagons is 2 * (apothem) * 2? No.
    # Width of hexagon (vertex to vertex) is 2.
    # Distance between centers of touching hexagons is 2 * (side) * sin(60) * 2? No.
    # If two hexagons share a side, the distance between their centers is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ~ 1.732.
    # Wait, let's verify.
    # Apothem = side * sqrt(3)/2.
    # Distance between centers = 2 * apothem = side * sqrt(3).
    # So for side 1, distance is sqrt(3) ~ 1.732.
    
    # Let's re-evaluate the grid.
    # If I place hexagons at distance sqrt(3), they touch side-to-side.
    # This is a valid packing.
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate hex grid positions
    # Center (0,0)
    # Ring 1: 6 positions at distance sqrt(3)
    # Ring 2: 12 positions at distance 2*sqrt(3)
    
    positions = []
    # Center
    positions.append((0.0, 0.0))
    
    # Ring 1
    for k in range(6):
        angle = math.radians(k * 60)
        x = sqrt3 * math.cos(angle)
        y = sqrt3 * math.sin(angle)
        positions.append((x, y))
        
    # Ring 2 (we need 5 more to make 12)
    # Positions in ring 2 are at distance 2*sqrt(3) from center, or sqrt(3) from ring 1 positions.
    # Let's just generate a larger grid and pick.
    # Grid points (q, r) with q+r in range...
    # Let's use the Cartesian coordinates of a hex grid.
    # Basis vectors: e1 = (sqrt3, 0.5), e2 = (0, 1)? No.
    # Standard hex grid basis for flat-top hexagons:
    # e1 = (1.5, sqrt3/2) -> distance 2? No.
    # If side=1, vertex to vertex is 2.
    # Center to center of touching hex is sqrt(3).
    # Let's use vectors of length sqrt(3) at 0, 60, 120 deg.
    
    v1 = (sqrt3, 0)
    v2 = (sqrt3 * math.cos(math.radians(60)), sqrt3 * math.sin(math.radians(60)))
    v3 = (sqrt3 * math.cos(math.radians(120)), sqrt3 * math.sin(math.radians(120)))
    
    grid_points = set()
    for q in range(-2, 3):
        for r in range(-2, 3):
            for s in range(-2, 3):
                if abs(q+r+s) > 2: continue # Hexagonal radius 2
                # This is getting complicated. Let's just use the known neighbors.
                pass
    
    # Simpler: Just list the 12 positions manually or generate via BFS.
    # Let's use the 1 center + 6 neighbors + 5 specific neighbors of neighbors.
    
    # Neighbors of (0,0):
    n0 = (sqrt3, 0)
    n1 = (sqrt3/2, 1.5)
    n2 = (-sqrt3/2, 1.5)
    n3 = (-sqrt3, 0)
    n4 = (-sqrt3/2, -1.5)
    n5 = (sqrt3/2, -1.5)
    
    # These are at distance sqrt(3).
    # Let's add 5 more from the next ring.
    # Next ring positions are at distance 2*sqrt(3) from center? No.
    # The "corners" of the hexagonal cluster of 7 are at distance 2*sqrt(3)?
    # Let's calculate distance from (0,0) to (sqrt3, 0) + (sqrt3, 0) = (2*sqrt3, 0). Dist = 2*sqrt3.
    # Is (2*sqrt3, 0) a valid hex grid position?
    # Yes, it's two steps in the same direction.
    # Let's add these 6 "double" neighbors, but we only need 5.
    
    d0 = (2*sqrt3, 0)
    d1 = (sqrt3 + sqrt3/2, 1.5) # (1.5*sqrt3, 1.5)
    d2 = (-sqrt3/2 + sqrt3/2, 1.5 + 1.5) # (0, 3) ? No.
    # Let's just compute them properly.
    # v1 = (sqrt3, 0)
    # v2 = (sqrt3/2, 1.5)
    # v3 = (-sqrt3/2, 1.5)
    
    # Neighbors of n0 (sqrt3, 0):
    # n0 + v1 = (2*sqrt3, 0)
    # n0 + v2 = (1.5*sqrt3, 1.5)
    # n0 + v3 = (0.5*sqrt3, 1.5)
    
    # Let's pick 5 distinct positions from the next ring.
    # To maintain symmetry, we can't just pick 5 arbitrarily without breaking symmetry,
    # but we can try to optimize.
    # For now, let's pick 5 that are spread out.
    
    extra_positions = [
        (2*sqrt3, 0),
        (1.5*sqrt3, 1.5),
        (0, 3), # This is 2*v2? v2=(sqrt3/2, 1.5). 2*v2=(sqrt3, 3). No.
        # Let's calculate (0, 3). Is it on the grid?
        # (0,3) = q*v1 + r*v2?
        # x = sqrt3*q + sqrt3/2*r = 0
        # y = 1.5*r = 3 => r=2.
        # sqrt3*q + sqrt3 = 0 => q = -1.
        # So (-1, 2) is a valid grid coord.
        # Let's use grid coords to be safe.
    ]
    
    # Let's define grid coords (q, r) and convert.
    # x = sqrt3 * q + sqrt3/2 * r
    # y = 1.5 * r
    
    def grid_to_cart(q, r):
        x = sqrt3 * q + sqrt3/2 * r
        y = 1.5 * r
        return (x, y)
    
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1), (1,2)
    # We need 12 points.
    # Use (0,0) + 6 from Ring 1 + 5 from Ring 2.
    
    init_grid_coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    
    centers = [grid_to_cart(q, r) for q, r in init_grid_coords]
    angles = [0.0] * N
    
    # Verify this initial config is valid
    if not check_overlap(centers, angles):
        # Fallback to a very sparse config if the grid is somehow invalid (shouldn't happen)
        centers = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        angles = [0.0] * N

    best_L = enclosing_side_length(centers, angles, 0.0)
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    best_outer_angle = 0.0
    
    # Local Search
    step_size = 0.1
    num_iterations = 500
    current_L = best_L
    current_centers = best_centers
    current_angles = best_angles
    current_outer_angle = best_outer_angle
    
    for it in range(num_iterations):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        
        # Try moving one center
        for i in range(N):
            for dx in [-step_size, 0.0, step_size]:
                for dy in [-step_size, 0.0, step_size]:
                    if dx == 0.0 and dy == 0.0: continue
                    new_centers = [c[:] for c in current_centers]
                    new_centers[i] = (new_centers[i][0] + dx, new_centers[i][1] + dy)
                    
                    if check_overlap(new_centers, current_angles):
                        new_L = enclosing_side_length(new_centers, current_angles, current_outer_angle)
                        if new_L < current_L - 1e-6:
                            current_L = new_L
                            current_centers = new_centers
                            improved = True
                            if current_L < best_L:
                                best_L = current_L
                                best_centers = [c[:] for c in current_centers]
                                best_angles = current_angles[:]
                                best_outer_angle = current_outer_angle
            if time.time() > deadline - 10.0:
                break
        
        # Try rotating outer hexagon
        for da in [-2.0, 0.0, 2.0]:
            if da == 0.0: continue
            new_outer_angle = (current_outer_angle + da) % 60.0
            new_L = enclosing_side_length(current_centers, current_angles, new_outer_angle)
            if new_L < current_L - 1e-6:
                current_L = new_L
                current_outer_angle = new_outer_angle
                improved = True
                if current_L < best_L:
                    best_L = current_L
                    best_centers = [c[:] for c in current_centers]
                    best_angles = current_angles[:]
                    best_outer_angle = current_outer_angle
        
        if not improved:
            # Try smaller step or break
            step_size *= 0.8
            if step_size < 0.01:
                break

    # Final check and return
    if not check_overlap(best_centers, best_angles):
        # If best is invalid, use the initial valid one
        best_centers = [grid_to_cart(q, r) for q, r in init_grid_coords]
        best_angles = [0.0] * N
        best_L = enclosing_side_length(best_centers, best_angles, 0.0)
        best_outer_angle = 0.0

    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])
    
    center = (0.0, 0.0)
    side = best_L
    angle = best_outer_angle
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
