import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 148.0

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE - elapsed()

    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon (center cx,cy, side length, rotation in degrees)."""
        a = np.deg2rad(angle_deg)
        verts = []
        for k in range(6):
            theta = a + np.deg2rad(60 * k)
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            verts.append([x, y])
        return np.array(verts)

    def point_in_poly(pt, poly):
        """Ray casting point-in-polygon test."""
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > pt[1]) != (yj > pt[1])) and \
               (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def seg_seg_dist(p1, p2, p3, p4):
        """Minimum distance between two line segments."""
        def pt_seg_dist(p, a, b):
            ab = b - a
            ab2 = np.dot(ab, ab)
            if ab2 < 1e-15:
                return np.linalg.norm(p - a)
            t = np.dot(p - a, ab) / ab2
            t = np.clip(t, 0.0, 1.0)
            proj = a + t * ab
            return np.linalg.norm(p - proj)

        d1 = pt_seg_dist(p1, p3, p4)
        d2 = pt_seg_dist(p2, p3, p4)
        d3 = pt_seg_dist(p3, p1, p2)
        d4 = pt_seg_dist(p4, p1, p2)
        return min(d1, d2, d3, d4)

    def min_dist_convex(p1, p2):
        """Minimum distance between two convex polygons. Returns 0 if they intersect."""
        # Check vertex-in-polygon
        for v in p1:
            if point_in_poly(v, p2):
                return 0.0
        for v in p2:
            if point_in_poly(v, p1):
                return 0.0

        # Compute min distance between all edge pairs
        min_d = float('inf')
        for i in range(len(p1)):
            a1 = p1[i]
            a2 = p1[(i + 1) % len(p1)]
            for j in range(len(p2)):
                b1 = p2[j]
                b2 = p2[(j + 1) % len(p2)]
                d = seg_seg_dist(a1, a2, b1, b2)
                if d < min_d:
                    min_d = d
        return min_d

    def is_valid_config(inner_centers, inner_angles, outer_center, outer_side, outer_angle):
        """Check disjointness and containment. Returns True if valid."""
        n = len(inner_centers)
        inner_hexes = []
        for i in range(n):
            x, y = inner_centers[i]
            inner_hexes.append(hex_vertices(x, y, 1.0, inner_angles[i]))

        # Check pairwise disjointness
        for i in range(n):
            for j in range(i + 1, n):
                d = min_dist_convex(inner_hexes[i], inner_hexes[j])
                if d < 1e-9:
                    return False

        # Check containment
        outer_hex = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
        for hx in inner_hexes:
            for v in hx:
                if not point_in_poly(v, outer_hex):
                    return False
        return True

    def min_enclosing_hex_side(inner_hexes, outer_center, outer_angle):
        """Find minimum side length of outer hexagon (at given center and angle) to contain all inner hexes."""
        outer_hex = hex_vertices(outer_center[0], outer_center[1], 1.0, outer_angle)
        # The outer hexagon at side length S is a scaled version of outer_hex about outer_center
        # For each inner vertex, find the minimum scale factor needed
        min_scale = 0.0
        for hx in inner_hexes:
            for v in hx:
                # Distance from outer_center to v in the direction of each normal
                # A regular hexagon with side S and angle phi: the distance from center to each edge is S*sqrt(3)/2
                # The hexagon is the intersection of 6 half-planes
                # Normal directions are at angles phi + 30 + 60k
                for k in range(6):
                    theta = np.deg2rad(outer_angle) + np.deg2rad(30 + 60 * k)
                    nx = np.cos(theta)
                    ny = np.sin(theta)
                    # Project vertex onto normal direction
                    proj = (v[0] - outer_center[0]) * nx + (v[1] - outer_center[1]) * ny
                    # The edge is at distance S*sqrt(3)/2 from center
                    # We need proj <= S*sqrt(3)/2
                    if proj > 0:
                        scale = proj / (np.sqrt(3) / 2)
                        min_scale = max(min_scale, scale)
        return min_scale

    # ---- Build the 11-hexagon flower packing ----
    # Center hexagon at origin, 6 around it, 4 more in the outer ring
    # Using triangular lattice positions
    # For unit hexagons (side 1), the distance between centers of adjacent hexagons is 2*sqrt(3)/2 * 1 = sqrt(3)
    # Actually for regular hexagons of side 1, the distance between centers when they share an edge is 2 * (sqrt(3)/2) * 1 = sqrt(3)
    # Wait, the apothem of a unit hexagon is sqrt(3)/2, so center-to-center distance for edge-sharing is 2*sqrt(3)/2 = sqrt(3)
    
    # Let me use the standard flower: center + 6 surrounding + 4 more
    # Actually, the classic 11-hexagon packing: 1 center, 6 in first ring, 4 in second ring
    # First ring: 6 hexagons at distance sqrt(3) from center, at angles 0, 60, 120, 180, 240, 300
    # Second ring: 4 hexagons at distance 2*sqrt(3) from center, at angles 30, 90, 150, 210, 270, 330
    # Wait, that would be 1 + 6 + 6 = 13. Let me reconsider.
    
    # For 11 hexagons: 1 center, 6 in first ring, 4 in second ring
    # The 4 in the second ring are at the "corner" positions
    
    eps = 1e-9
    centers = [(0.0, 0.0)]  # center
    # First ring: 6 hexagons
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers.append((r * np.cos(angle), r * np.sin(angle)))
    # Second ring: 4 hexagons at the "gaps" between first ring hexagons
    # These are at distance 2*sqrt(3) from center, at angles 30, 150, 270, 30+180=210... 
    # Actually let me think more carefully. The 6 first-ring hexagons are at angles 0,60,120,180,240,300
    # The gaps are at 30, 90, 150, 210, 270, 330
    # But we only have 4 more to place. Let's place them at 30, 150, 270, and... 
    # Hmm, maybe a different arrangement. Let me use 1+6+4 where the 4 are at 90, 270, and two others.
    # Actually, a common 11-hex packing: center, 6 around, and 4 at the "diagonal" positions
    # Let me just use the positions that work:
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)  # 30, 90, 150, 210
        r = 2 * np.sqrt(3) + eps
        centers.append((r * np.cos(angle), r * np.sin(angle)))
    
    n = len(centers)  # should be 11
    init_inner_angles = [30.0] * n
    init_outer_center = (0.0, 0.0)
    init_outer_angle = 30.0

    # Compute initial outer side
    inner_hexes = []
    for x, y in centers:
        inner_hexes.append(hex_vertices(x, y, 1.0, 30.0))
    init_side = min_enclosing_hex_side(inner_hexes, (0.0, 0.0), 30.0) + 1e-6

    # ---- Optimize over outer hexagon angle and inner cluster rotation ----
    # The key insight: we can rotate the entire cluster and the outer hexagon independently
    # to find the minimum enclosing side length
    
    best_side = init_side
    best_cluster_angle = 0.0
    best_outer_angle = 30.0
    best_centers = centers[:]
    best_inner_angles = init_inner_angles[:]

    # Try different cluster rotations and outer hexagon angles
    # Due to symmetry, we only need to check angles in [0, 30) degrees
    # But let's be more thorough with a fine grid first, then refine
    
    def evaluate_config(cluster_rotation, outer_rotation):
        """Given cluster rotation and outer hexagon rotation, compute min enclosing side."""
        # Rotate all centers around origin by cluster_rotation
        cr = np.deg2rad(cluster_rotation)
        rotated_centers = []
        for x, y in centers:
            rx = x * np.cos(cr) - y * np.sin(cr)
            ry = x * np.sin(cr) + y * np.cos(cr)
            rotated_centers.append((rx, ry))
        
        # Rotate all inner hexagon angles
        rotated_angles = [a + cluster_rotation for a in init_inner_angles]
        
        # Compute inner hexagons
        rot_hexes = []
        for i in range(n):
            rot_hexes.append(hex_vertices(rotated_centers[i][0], rotated_centers[i][1], 1.0, rotated_angles[i]))
        
        # Find min enclosing side
        side = min_enclosing_hex_side(rot_hexes, (0.0, 0.0), outer_rotation)
        return side, rotated_centers, rotated_angles

    # Coarse grid search
    coarse_steps = 30
    for i in range(coarse_steps + 1):
        if remaining() < 10.0:
            break
        cluster_rot = i * 30.0 / coarse_steps
        for j in range(coarse_steps + 1):
            if remaining() < 10.0:
                break
            outer_rot = j * 30.0 / coarse_steps
            side, rot_c, rot_a = evaluate_config(cluster_rot, outer_rot)
            if side < best_side:
                best_side = side
                best_cluster_angle = cluster_rot
                best_outer_angle = outer_rot
                best_centers = rot_c
                best_inner_angles = rot_a

    # Fine grid search around the best found
    if remaining() > 20.0:
        fine_steps = 60
        c_lo = max(0.0, best_cluster_angle - 30.0 / coarse_steps)
        c_hi = min(30.0, best_cluster_angle + 30.0 / coarse_steps)
        o_lo = max(0.0, best_outer_angle - 30.0 / coarse_steps)
        o_hi = min(30.0, best_outer_angle + 30.0 / coarse_steps)
        
        for i in range(fine_steps + 1):
            if remaining() < 15.0:
                break
            cluster_rot = c_lo + (c_hi - c_lo) * i / fine_steps
            for j in range(fine_steps + 1):
                if remaining() < 15.0:
                    break
                outer_rot = o_lo + (o_hi - o_lo) * j / fine_steps
                side, rot_c, rot_a = evaluate_config(cluster_rot, outer_rot)
                if side < best_side:
                    best_side = side
                    best_cluster_angle = cluster_rot
                    best_outer_angle = outer_rot
                    best_centers = rot_c
                    best_inner_angles = rot_a

    # Even finer search
    if remaining() > 15.0:
        finer_steps = 120
        delta = 0.01
        c_lo = max(0.0, best_cluster_angle - delta)
        c_hi = min(30.0, best_cluster_angle + delta)
        o_lo = max(0.0, best_outer_angle - delta)
        o_hi = min(30.0, best_outer_angle + delta)
        
        for i in range(finer_steps + 1):
            if remaining() < 10.0:
                break
            cluster_rot = c_lo + (c_hi - c_lo) * i / finer_steps
            for j in range(finer_steps + 1):
                if remaining() < 10.0:
                    break
                outer_rot = o_lo + (o_hi - o_lo) * j / finer_steps
                side, rot_c, rot_a = evaluate_config(cluster_rot, outer_rot)
                if side < best_side:
                    best_side = side
                    best_cluster_angle = cluster_rot
                    best_outer_angle = outer_rot
                    best_centers = rot_c
                    best_inner_angles = rot_a

    # Use the best configuration found
    inner_centers = best_centers
    inner_angles = best_inner_angles
    outer_center = (0.0, 0.0)
    outer_side = best_side + 1e-8  # tiny margin
    outer_angle = best_outer_angle

    # Final validity check
    if is_valid_config(inner_centers, inner_angles, outer_center, outer_side, outer_angle):
        inner = [[inner_centers[i][0], inner_centers[i][1], inner_angles[i]] for i in range(n)]
        return inner, outer_center, outer_side, outer_angle
    else:
        # Fallback: use the initial configuration with a safe margin
        inner = []
        for x, y in centers:
            inner.append([x, y, 30.0])
        return inner, (0.0, 0.0), init_side, 30.0
