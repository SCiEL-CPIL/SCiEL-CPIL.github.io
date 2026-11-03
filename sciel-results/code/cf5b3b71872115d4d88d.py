"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # 150s limit, keep 10s margin
    rng = random.Random(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon with given center, side length, and rotation."""
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        # Transform point to hexagon's local coordinate system
        dx = px - cx
        dy = py - cy
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # Rotate point into hexagon frame
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        # Regular hexagon with flat-top orientation: vertices at angles 0, 60, 120, 180, 240, 300
        # The hexagon has "radius" (center to vertex) = side, and "apothem" (center to edge) = side * sqrt(3)/2
        # For a hexagon with vertices at angles 0,60,...: 
        #   |x| <= side and |y| <= side*sqrt(3)/2 and |x| + |y|/sqrt(3) <= side
        # Actually for pointy-top vs flat-top... let me use the standard:
        # Vertices at angle 0, 60, 120, 180, 240, 300 degrees from center
        # This is a "pointy-top" hexagon if we consider y-up
        # The constraint: for each edge, the point must be on the correct side
        # Edge i goes from vertex i to vertex i+1, normal points inward
        r = side  # circumradius
        # The 6 edges have outward normals at angles 30, 90, 150, 210, 270, 330 degrees
        # For edge with outward normal at angle phi, the constraint is:
        #   lx * cos(phi) + ly * sin(phi) <= side * cos(30) = side * sqrt(3)/2
        for i in range(6):
            phi = math.radians(30 + i * 60)
            if lx * math.cos(phi) + ly * math.sin(phi) > side * math.sqrt(3) / 2 + 1e-10:
                return False
        return True

    def hexagons_disjoint(v1, v2, ang1, ang2, cx1, cy1, cx2, cy2, side=1.0, tol=1e-9):
        """Check if two hexagons are disjoint (no overlapping interiors)."""
        # Quick reject: distance between centers
        dx = cx2 - cx1
        dy = cy2 - cy1
        dist = math.sqrt(dx * dx + dy * dy)
        # Minimum distance for disjoint hexagons of side 1 is 2 (if aligned), 
        # but with rotation it can be less. Use a conservative bound: 2*side - tol
        if dist > 2.0 * side - tol:
            return True
        # Check if any vertex of hex1 is inside hex2, or vice versa
        verts1 = get_hex_vertices(cx1, cy1, side, ang1)
        verts2 = get_hex_vertices(cx2, cy2, side, ang2)
        for (vx, vy) in verts1:
            if point_in_hexagon(vx, vy, cx2, cy2, side, ang2):
                return False
        for (vx, vy) in verts2:
            if point_in_hexagon(vx, vy, cx1, cy1, side, ang1):
                return False
        # Check edge intersections
        for i in range(6):
            p1 = verts1[i]
            p2 = verts1[(i + 1) % 6]
            for j in range(6):
                p3 = verts2[j]
                p4 = verts2[(j + 1) % 6]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        """Check if line segments p1-p2 and p3-p4 intersect."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def hex_in_container(cx, cy, angle_deg, outer_cx, outer_cy, outer_side, outer_angle):
        """Check if unit hexagon at (cx,cy) is inside the outer hexagon."""
        verts = get_hex_vertices(cx, cy, 1.0, angle_deg)
        for (vx, vy) in verts:
            if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                return False
        return True

    def all_valid(positions, outer_cx, outer_cy, outer_side, outer_angle):
        """Check if all hexagons are inside container and pairwise disjoint."""
        # Check containment
        for i in range(N):
            if not hex_in_container(positions[i][0], positions[i][1], positions[i][2],
                                    outer_cx, outer_cy, outer_side, outer_angle):
                return False
        # Check pairwise disjointness
        for i in range(N):
            for j in range(i + 1, N):
                if not hexagons_disjoint(positions[i][2], positions[j][2],
                                         positions[i][0], positions[i][1],
                                         positions[j][0], positions[j][1], 1.0):
                    return False
        return True

    def compute_min_outer(positions):
        """Compute the minimum outer hexagon side length and center that contains all inner hexagons.
        We try a few outer angles and find the best."""
        best_side = float('inf')
        best_center = (0.0, 0.0)
        best_angle = 0.0
        
        # For each candidate outer angle, find the minimum enclosing hexagon
        for outer_angle in [0, 30, 15, 45, 60]:
            # Get all vertices of all inner hexagons
            all_verts = []
            for i in range(N):
                verts = get_hex_vertices(positions[i][0], positions[i][1], 1.0, positions[i][2])
                all_verts.append(verts)
            all_verts = np.vstack(all_verts)
            
            # For a regular hexagon with given rotation, find min side and center
            # The hexagon is defined by 6 half-planes. We need to find the smallest one containing all points.
            # This is equivalent to: for each of the 6 directions, find the max extent, then the side length
            # is determined by the maximum of these extents.
            
            # Actually, let's think differently. A regular hexagon with center (cx,cy), side s, angle a
            # contains point (px,py) if for all 6 edge normals n_k:
            #   (px-cx)*n_k.x + (py-cy)*n_k.y <= s * sqrt(3)/2
            # where n_k are unit vectors at angles (a + 30 + 60k) degrees.
            
            # To minimize s, we can use: for a given center and angle, s must be >= max over all points and all 6 directions of:
            #   ((px-cx)*n_k.x + (py-cy)*n_k.y) / (sqrt(3)/2)
            # And s must be >= 0.
            
            # But we also need to optimize over center. This is a 2D optimization.
            # Let's use a simpler approach: the center should be near the centroid of all vertices.
            centroid_x = np.mean(all_verts[:, 0])
            centroid_y = np.mean(all_verts[:, 1])
            
            # Try the centroid as center, and also try a few offsets
            centers_to_try = [
                (centroid_x, centroid_y),
                (centroid_x + 0.5, centroid_y),
                (centroid_x - 0.5, centroid_y),
                (centroid_x, centroid_y + 0.5),
                (centroid_x, centroid_y - 0.5),
                (centroid_x + 0.3, centroid_y + 0.3),
                (centroid_x - 0.3, centroid_y - 0.3),
            ]
            
            for (cx, cy) in centers_to_try:
                max_extent = 0.0
                for k in range(6):
                    phi = math.radians(outer_angle + 30 + k * 60)
                    nx = math.cos(phi)
                    ny = math.sin(phi)
                    # For all points, compute the projection
                    projections = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
                    max_proj = np.max(projections)
                    # Also need to check the other side? No, because the hexagon is symmetric.
                    # Actually, we need BOTH sides: the hexagon extends in both directions along each normal.
                    # Wait, no. A regular hexagon with center (cx,cy) and the 6 edge normals pointing outward:
                    # The constraint is that for each normal n_k, ALL points satisfy:
                    #   (p - c) . n_k <= s * sqrt(3)/2
                    # AND
                    #   (p - c) . (-n_k) <= s * sqrt(3)/2  i.e., (p-c).n_k >= -s*sqrt(3)/2
                    # So actually we need: max_k max_p |(p-c).n_k| <= s * sqrt(3)/2
                    min_proj = np.min(projections)
                    extent = max(abs(max_proj), abs(min_proj))
                    if extent > max_extent:
                        max_extent = extent
                s = max_extent / (math.sqrt(3) / 2)
                if s < best_side:
                    best_side = s
                    best_center = (cx, cy)
                    best_angle = outer_angle
        
        return best_side, best_center, best_angle

    # Initial layout: hexagonal grid arrangement
    # Hexagonal grid: rows offset
    # For 12 hexagons, a good arrangement is 3-4-3-2 or similar
    # Let's use a hexagonal packing: 
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Or better: 2-3-4-3 or a more symmetric arrangement
    
    # Standard hex grid spacing: horizontal = 2*side*sqrt(3)/2 = sqrt(3)*side for pointy-top
    # For flat-top hexagons (vertices at 0, 60, ...): 
    #   horizontal spacing = 2 * side * cos(30) = side * sqrt(3)
    #   vertical spacing = side * 1.5 (for adjacent rows)
    #   row offset = side * sqrt(3)/2
    
    # Let's use a symmetric arrangement
    # Arrangement: center + 6 around + 5 more
    # Or use a 4x3 grid with hex offsets
    
    # Let me try: place them in a hexagonal pattern
    # Centers in hex grid (using axial coordinates converted to cartesian)
    # For flat-top hexagons:
    #   x = size * sqrt(3) * (col + row/2)
    #   y = size * 3/2 * row
    
    # Let's create a compact arrangement
    # 12 hexagons in a hexagonal cluster:
    # Row -1: 3 hexagons (cols -1, 0, 1)
    # Row 0:  4 hexagons (cols -1, 0, 1, 2) -- offset
    # Row 1:  3 hexagons (cols -1, 0, 1)
    # Row 2:  2 hexagons (cols 0, 1) -- offset
    # Total: 3+4+3+2 = 12
    
    # Actually, let me use a simpler approach: just place them in a grid and optimize
    # Start with a 4x3 grid
    positions = []
    # 4 columns, 3 rows
    for row in range(3):
        for col in range(4):
            x = col * math.sqrt(3) + (row % 2) * math.sqrt(3) / 2
            y = row * 1.5
            positions.append([x, y, 0.0])
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_mean = (min(xs) + max(xs)) / 2
    cy_mean = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx_mean
        p[1] -= cy_mean
    
    # Initial outer
    outer_side, outer_center, outer_angle = compute_min_outer(positions)
    
    best_positions = [list(p) for p in positions]
    best_outer_side = outer_side
    best_outer_center = outer_center
    best_outer_angle = outer_angle
    best_score = outer_side
    
    # Local optimization: perturb positions and recompute
    def optimize_local(positions, outer_side, outer_center, outer_angle, max_iter=5000):
        """Local optimization using random perturbations."""
        pos = [list(p) for p in positions]
        o_side = outer_side
        o_cx, o_cy = outer_center
        o_angle = outer_angle
        score = o_side
        step_size = 0.1
        no_improve = 0
        max_no_improve = 500
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            if no_improve > max_no_improve:
                break
            
            # Randomly perturb one hexagon
            idx = rng.randint(0, N - 1)
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            dangle = rng.uniform(-5, 5)
            
            old_x, old_y, old_ang = pos[idx][0], pos[idx][1], pos[idx][2]
            pos[idx][0] += dx
            pos[idx][1] += dy
            pos[idx][2] += dangle
            
            # Check if still valid (disjoint and contained in a slightly larger container)
            # For speed, just check disjointness with neighbors
            valid = True
            for j in range(N):
                if j == idx:
                    continue
                if not hexagons_disjoint(pos[idx][2], pos[j][2],
                                         pos[idx][0], pos[idx][1],
                                         pos[j][0], pos[j][1], 1.0):
                    valid = False
                    break
            
            if valid:
                new_side, new_center, new_angle = compute_min_outer(pos)
                if new_side < score - 1e-10:
                    score = new_side
                    o_side = new_side
                    o_cx, o_cy = new_center
                    o_angle = new_angle
                    no_improve = 0
                    # Decay step size
                    if step_size > 0.01:
                        step_size *= 0.99
                else:
                    # Revert
                    pos[idx][0] = old_x
                    pos[idx][1] = old_y
                    pos[idx][2] = old_ang
                    no_improve += 1
            else:
                pos[idx][0] = old_x
                pos[idx][1] = old_y
                pos[idx][2] = old_ang
                no_improve += 1
        
        return pos, o_side, (o_cx, o_cy), o_angle, score
    
    # Run local optimization
    best_positions, best_outer_side, best_outer_center, best_outer_angle, best_score = \
        optimize_local(positions, outer_side, outer_center, outer_angle)
    
    # Try multiple random restarts if time permits
    num_restarts = 0
    while time.time() < deadline - 20:
        num_restarts += 1
        # Generate a new random initial configuration
        new_positions = []
        for i in range(N):
            x = rng.uniform(-3, 3)
            y = rng.uniform(-3, 3)
            ang = rng.uniform(0, 60)
            new_positions.append([x, y, ang])
        
        # Check if this random config is valid
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if not hexagons_disjoint(new_positions[i][2], new_positions[j][2],
                                         new_positions[i][0], new_positions[i][1],
                                         new_positions[j][0], new_positions[j][1], 1.0):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            ns, nc, na = compute_min_outer(new_positions)
            if ns < best_score:
                # Optimize this configuration
                opt_pos, opt_side, opt_center, opt_angle, opt_score = \
                    optimize_local(new_positions, ns, nc, na, max_iter=3000)
                if opt_score < best_score:
                    best_positions = opt_pos
                    best_outer_side = opt_side
                    best_outer_center = opt_center
                    best_outer_angle = opt_angle
                    best_score = opt_score
    
    # Final verification and return
    inner = [[p[0], p[1], p[2]] for p in best_positions]
    return inner, best_outer_center, best_outer_side, best_outer_angle
# EVOLVE-BLOCK-END
