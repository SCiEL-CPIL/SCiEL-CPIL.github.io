<<<<<<< SEARCH
    def check_disjoint(c1, c2, a1, a2, min_dist_sq=3.0 - 1e-9):
        """Two unit hexagons are disjoint if center distance > sqrt(3) (for aligned).
        For rotated, we need to check vertex-vertex or use conservative bound.
        Conservative: if center distance > 2, definitely disjoint.
        If center distance < sqrt(3) - epsilon, might overlap for aligned.
        For safety, check if any vertex of one is inside the other."""
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist_sq = dx*dx + dy*dy
        # If far enough, definitely disjoint
        if dist_sq > 4.0:  # distance > 2
            return True
        # Check if any vertex of hex1 is inside hex2 or vice versa
        v1 = hex_vertices(c1[0], c1[1], 1.0)
        v2 = hex_vertices(c2[0], c2[1], 1.0)
        
        a1_rad = math.radians(a1)
        a2_rad = math.radians(a2)
        
        def rotate_pts(verts, cx, cy, angle_rad):
            ca, sa = math.cos(angle_rad), math.sin(angle_rad)
            result = []
            for vx, vy in verts:
                dx = vx - cx
                dy = vy - cy
                rx = cx + dx * ca - dy * sa
                ry = cy + dx * sa + dy * ca
                result.append((rx, ry))
            return result
        
        rv1 = rotate_pts(v1, c1[0], c1[1], a1_rad)
        rv2 = rotate_pts(v2, c2[0], c2[1], a2_rad)
        
        # Check if any vertex of rv1 is inside hex2 (centered at c2, rotated a2)
        for vx, vy in rv1:
            if point_in_hexagon(vx, vy, c2[0], c2[1], 1.0, a2):
                return False
        # Check if any vertex of rv2 is inside hex1
        for vx, vy in rv2:
            if point_in_hexagon(vx, vy, c1[0], c1[1], 1.0, a1):
                return False
        return True
=======
    def check_disjoint(c1, c2, a1, a2, min_dist_sq=3.0 - 1e-9):
        """Check if two unit hexagons are disjoint (non-overlapping)."""
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist_sq = dx*dx + dy*dy
        # If far enough, definitely disjoint
        if dist_sq > 4.0:  # distance > 2
            return True
        
        # Get rotated vertices
        def get_rotated_verts(cx, cy, angle_deg):
            verts = hex_vertices(cx, cy, 1.0)
            a_rad = math.radians(angle_deg)
            ca, sa = math.cos(a_rad), math.sin(a_rad)
            result = []
            for vx, vy in verts:
                dx = vx - cx
                dy = vy - cy
                rx = cx + dx * ca - dy * sa
                ry = cy + dx * sa + dy * ca
                result.append((rx, ry))
            return result
        
        rv1 = get_rotated_verts(c1[0], c1[1], a1)
        rv2 = get_rotated_verts(c2[0], c2[1], a2)
        
        # Check if any vertex of rv1 is strictly inside hex2
        # Using a slightly relaxed point_in_hexagon or checking strictly inside
        # If a vertex is on the boundary, it's touching, which is allowed.
        # But if edges cross, it's overlap.
        
        # Robust check: Segment intersection
        def segments_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
            
        # Check all pairs of edges
        for i in range(6):
            p1 = rv1[i]
            p2 = rv1[(i+1)%6]
            for j in range(6):
                p3 = rv2[j]
                p4 = rv2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return False
                    
        # Check if any vertex is strictly inside the other
        # We use point_in_hexagon but need to be careful with boundaries.
        # If point_in_hexagon returns True, it might be on boundary.
        # If it's on boundary, we should check if it's a shared vertex or just touching.
        # For simplicity, if vertex is inside (including boundary), we assume potential overlap unless it's a known tiling case.
        # However, in a tiling, vertices touch.
        # Let's rely on segment intersection for overlaps.
        # If no segments intersect, they are disjoint or touching.
        return True
>>>>>>> REPLACE

The `point_in_hexagon` function uses `proj > apothem` which means points on the boundary are considered "inside" (or rather, not outside). This is fine for containment checks. For disjointness, the segment intersection check is the key fix.

Additionally, the `minimal_enclosing_hexagon_side` function might return an invalid center. I will add a check to ensure the returned center is valid, and if not, try a few other candidate centers (like the origin, or the average of the min/max bounds projected back).

