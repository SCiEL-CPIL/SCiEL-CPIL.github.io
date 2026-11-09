"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 12

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 130.0

    # Unit regular hexagon with side 1, flat-top orientation (angle 0).
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    base_vertices = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(0, 360, 60)]

    # Axial coordinate to Cartesian (hex grid, side=1, flat-top hexagons touching edge-to-edge)
    # For a regular hex grid where adjacent hexagons share an edge:
    #   x = 1.5 * q + 0.5 * r  (for pointy-top) -- but let's use flat-top
    # Actually, for flat-top hexagons with side 1:
    #   x = 1.5 * q
    #   y = sqrt(3) * (r + 0.5 * q)
    # This gives distance between adjacent centers = sqrt(3) which is the edge length... no.
    # Distance between adjacent flat-top hex centers: 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)? 
    # No. Two flat-top hexagons sharing a vertical edge: center distance = 1.5 (horizontal) ... 
    # Let me reconsider.
    #
    # Flat-top hexagon, side length 1:
    #   Width (horizontal) = 2
    #   Height (vertical) = sqrt(3)
    #   Apothem (center to edge midpoint) = sqrt(3)/2
    #   Two hexagons sharing a left/right edge: horizontal separation = 1.5, vertical = 0
    #   Two hexagons sharing a top/bottom edge: separation = (0.75, sqrt(3)/2 * ... )
    #
    # Standard axial to Cartesian for flat-top, side=1:
    #   x = 1.5 * q
    #   y = sqrt(3) * (r + q/2)
    # Check: (0,0) and (1,0) -> x: 0 vs 1.5, y: 0 vs sqrt(3)*0.5 = sqrt(3)/2
    #   dist = sqrt(1.5^2 + (sqrt(3)/2)^2) = sqrt(2.25 + 0.75) = sqrt(3) ≈ 1.732
    # But the minimum distance between two unit hexagons touching edge-to-edge should be... 
    # The apothem is sqrt(3)/2 ≈ 0.866. Two hexagons sharing an edge: center distance = 2 * apothem = sqrt(3) ≈ 1.732.
    # Wait, no. Two regular hexagons of side 1 sharing an edge: the centers are separated by 2 * apothem = sqrt(3). Yes.
    # But that means they overlap if we just place centers at distance sqrt(3)? No, they share the edge, so they're tangent.
    # Actually for non-overlapping, centers must be at least sqrt(3) apart if aligned to share an edge, but could be more if rotated.
    # For our purposes, we'll use a hex grid where each hexagon is placed at a lattice point, and they tile perfectly.
    
    SQRT3 = math.sqrt(3)
    
    def axial_to_cart(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + 0.5 * q)
        return (x, y)
    
    # Generate the 19-hex cluster (side 3 hexagon): all (q,r) with max(|q|,|r|,|q+r|) <= 2
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q + r)) <= 2:
                all_centers.append((q, r))
    
    # We need to pick 12 of these 19 to minimize the bounding regular hexagon.
    # The 19-hex cluster has circumradius 3 (in terms of hex units), meaning the outer hexagon 
    # has side length 3 in the same grid.
    # 
    # Strategy: Remove 7 hexagons from the periphery to get a compact shape of 12.
    # The 19 hexagons arranged in a hexagon of side 3:
    #   Row r=-2: q=0,1,2 (3 hexes)
    #   Row r=-1: q=-1,0,1,2 (4 hexes)  
    #   Row r=0:  q=-2,-1,0,1,2 (5 hexes)
    #   Row r=1:  q=-2,-1,0,1 (4 hexes)
    #   Row r=2:  q=-2,-1,0 (3 hexes)
    #
    # Let's identify which to remove. The corners of the big hexagon are at:
    # (2,0), (1,1), (0,2), (-2,0), (-1,-1), (0,-2) in (q,r) -- wait let me recheck.
    # For max(|q|,|r|,|q+r|) <= 2, the 6 corners are:
    # (2,0), (2,-2)?? No. |q+r| = 0 for (2,0). Let me list all boundary points.
    
    # Let's just try a known compact 12-hex arrangement.
    # A "hexagon of side 2" has 7 hexes. We need 5 more.
    # Place 5 hexes around it in a ring, leaving one gap.
    # 
    # Core 7: max(|q|,|r|,|q+r|) <= 1
    # Ring of 12: max(|q|,|r|,|q+r|) <= 2 but > 1
    # Total 19. We pick all 7 core + 5 from the ring.
    
    core = set()
    ring = set()
    for q, r in all_centers:
        if max(abs(q), abs(r), abs(q + r)) <= 1:
            core.add((q, r))
        else:
            ring.add((q, r))
    
    # core has 7, ring has 12. Pick 5 from ring to minimize spread.
    # The ring hexagons are at distance 2 from center (in hex units).
    # Let's pick 5 that are roughly evenly spaced or clustered to minimize bounding box.
    
    # Ring positions (q, r) with max(|q|,|r|,|q+r|) == 2:
    ring_list = sorted(ring)
    
    # Try picking 5 from ring. For now, pick 5 specific ones.
    # Let's pick the ones at "corners" minus one to create an asymmetric shape.
    # Actually, let's just try: take all 7 core, plus 5 ring hexes.
    # Which 5? Let's try: (2,0), (1,1), (0,2), (-1,2), (-2,1) -- that's 5 consecutive on one side.
    # Or spread them: (2,0), (0,2), (-2,0), (0,-2), and one more.
    
    # Let me just compute the bounding hexagon for a candidate and check.
    
    def hex_distance_to_edge(x, y, outer_center, outer_side, outer_angle):
        """Minimum distance from point (x,y) to the boundary of a regular hexagon."""
        # Transform to outer hexagon's frame
        dx = x - outer_center[0]
        dy = y - outer_center[1]
        # Rotate by -outer_angle
        ca = math.cos(-math.radians(outer_angle))
        sa = math.sin(-math.radians(outer_angle))
        rx = ca * dx - sa * dy
        ry = sa * dx + ca * dy
        # In the hexagon's frame, a regular hexagon with side s has vertices at distance s from center
        # at angles 0, 60, 120, 180, 240, 300.
        # The distance from center to each edge (apothem) is s * sqrt(3) / 2.
        # For a point at (rx, ry), the distance to the boundary along the direction of the point:
        # We need the minimum t > 0 such that t*(rx, ry) is on the boundary.
        # For a regular hexagon, the boundary in polar coords (relative to center) has
        # r(theta) = apothem / cos((theta - k*60) mod 60 - 30) for the appropriate sector.
        # More precisely, for angle theta, find the nearest multiple of 60 degrees offset by 30.
        # The apothem is the distance to the edge midpoint.
        # Let's use: for each of the 6 edges, compute distance from point to that edge line,
        # and take the minimum positive one (inside the hexagon).
        
        apothem = outer_side * SQRT3 / 2
        
        # Edge midpoints are at angles 30, 90, 150, 210, 270, 330 (for flat-top with vertex at 0)
        # Actually for a hexagon with vertex at angle 0, edges are centered at 30, 90, etc.
        # Edge normal directions: 30, 90, 150, 210, 270, 330 degrees.
        # The constraint is: rx * cos(n) + ry * sin(n) <= apothem for each normal n.
        # Distance to boundary along direction (rx, ry): 
        #   For the relevant edge, t = apothem / (rx * cos(n) + ry * sin(n))
        
        theta = math.atan2(ry, rx)
        best_t = float('inf')
        for k in range(6):
            edge_angle = math.radians(30 + 60 * k)
            denom = rx * math.cos(edge_angle) + ry * math.sin(edge_angle)
            if denom > 1e-9:
                t = apothem / denom
                if t < best_t:
                    best_t = t
        return best_t
    
    def check_packing(inner_positions, outer_center, outer_side, outer_angle):
        """Check if all inner hexagons fit inside the outer hexagon and don't overlap each other.
        Return True if valid, False otherwise."""
        
        # Check outer containment: each inner hexagon's farthest vertex from outer center
        # must be within the outer hexagon.
        # Actually, we need every point of every inner hexagon to be inside the outer hexagon.
        # Conservative check: for each inner hexagon, check its 6 vertices against the outer hexagon.
        
        for (cx, cy, ang) in inner_positions:
            # Get vertices of this inner hexagon
            verts = []
            for (vx, vy) in base_vertices:
                # Rotate by ang
                ca = math.cos(math.radians(ang))
                sa = math.sin(math.radians(ang))
                rx = ca * vx - sa * vy
                ry = sa * vx + ca * vy
                verts.append((cx + rx, cy + ry))
            
            # Check each vertex is inside outer hexagon
            for (vx, vy) in verts:
                dx = vx - outer_center[0]
                dy = vy - outer_center[1]
                ca = math.cos(-math.radians(outer_angle))
                sa = math.sin(-math.radians(outer_angle))
                rx = ca * dx - sa * dy
                ry = sa * dx + ca * dy
                
                apothem = outer_side * SQRT3 / 2
                for k in range(6):
                    edge_angle = math.radians(30 + 60 * k)
                    val = rx * math.cos(edge_angle) + ry * math.sin(edge_angle)
                    if val > apothem + 1e-9:
                        return False
        
        # Check pairwise non-overlap: for each pair, minimum distance between hexagons >= 0
        # Two convex hexagons don't overlap if no vertex of one is inside the other 
        # AND no edges cross. Conservative: check if any vertex of either is inside the other,
        # and if edge midpoints are too close.
        # Simpler: for each pair, check the 36 vertex-edge distances.
        # Actually, for regular hexagons, they overlap iff the distance between centers
        # is less than some threshold depending on relative rotation.
        # Minimum distance between two unit hexagons with relative angle delta:
        # If aligned (delta=0): min center distance = sqrt(3) (sharing edge)
        # For general delta, it's more complex.
        # Let's do a vertex-in-hex test for each pair.
        
        n_inner = len(inner_positions)
        for i in range(n_inner):
            for j in range(i+1, n_inner):
                ci, cj = inner_positions[i], inner_positions[j]
                # Check if any vertex of i is inside j, or any vertex of j is inside i
                verts_i = []
                for (vx, vy) in base_vertices:
                    ca = math.cos(math.radians(ci[2]))
                    sa = math.sin(math.radians(ci[2]))
                    verts_i.append((ci[0] + ca*vx - sa*vy, ci[1] + sa*vx + ca*vy))
                
                verts_j = []
                for (vx, vy) in base_vertices:
                    ca = math.cos(math.radians(cj[2]))
                    sa = math.sin(math.radians(cj[2]))
                    verts_j.append((cj[0] + ca*vx - sa*vy, cj[1] + sa*vx + ca*vy))
                
                # Point-in-convex-polygon test
                def point_in_hex(px, py, center, angle):
                    # Transform to hex's local frame
                    dx = px - center[0]
                    dy = py - center[1]
                    ca = math.cos(-math.radians(angle))
                    sa = math.sin(-math.radians(angle))
                    rx = ca * dx - sa * dy
                    ry = sa * dx + ca * dy
                    # Check all 6 edges
                    for k in range(6):
                        edge_angle = math.radians(30 + 60 * k)
                        val = rx * math.cos(edge_angle) + ry * math.sin(edge_angle)
                        if val > 1 + 1e-9:  # apothem of unit hex = sqrt(3)/2... wait
                            # For unit hex, apothem = sqrt(3)/2
                            pass
                    # Actually for unit hex, the constraint is val <= sqrt(3)/2
                    # Let me redo:
                    apothem_unit = SQRT3 / 2
                    for k in range(6):
                        edge_angle = math.radians(30 + 60 * k)
                        val = rx * math.cos(edge_angle) + ry * math.sin(edge_angle)
                        if val > apothem_unit + 1e-9:
                            return False
                    return True
                
                overlap = False
                for v in verts_i:
                    if point_in_hex(v[0], v[1], (cj[0], cj[1]), cj[2]):
                        overlap = True
                        break
                if not overlap:
                    for v in verts_j:
                        if point_in_hex(v[0], v[1], (ci[0], ci[1]), ci[2]):
                            overlap = True
                            break
                if overlap:
                    return False
        
        return True
    
    # Let's use a known good configuration: the 12 hexagons from the hexagonal grid.
    # Place them in a "hexagon of side 2" (7 hexes) plus 5 more.
    # 
    # Actually, let me just use a simple arrangement and compute the minimum enclosing hexagon.
    
    # Simple approach: place 12 hexagons in a 2x3 grid pattern (hex grid).
    # Axial coords: q in [0,1,2,3,4,5], r = 0 -- that's a line, bad.
    # Better: two rows of 6.
    # Row 1 (r=0): q = -2, -1, 0, 1, 2, 3
    # Row 2 (r=1): q = -3, -2, -1, 0, 1, 2
    # This forms a parallelogram-ish shape.
    
    # Let me try the compact 12: core 7 + 5 ring.
    selected = list(core)
    # Pick 5 from ring
    ring_list = sorted(list(ring))
    # Pick specific 5: let's take (2,0), (1,1), (0,2), (-2,1), (-1,0) -- wait (-1,0) is in core.
    # Ring points: let me list them.
    for q, r in sorted(ring):
        pass  # just for reference
    
    # Ring points (q,r) where max(|q|,|r|,|q+r|)==2:
    # (2,0), (2,-1), (2,-2), (1,1), (1,2)?? No, |1+2|=3>2. 
    # Let me just enumerate properly.
    
    ring_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2:
                ring_points.append((q, r))
    
    # From the 12 ring points, pick 5. Let's try picking every other one for even spacing.
    # Sort by angle from center.
    def angle_of(q, r):
        x, y = axial_to_cart(q, r)
        return math.atan2(y, x)
    
    ring_points_sorted = sorted(ring_points, key=lambda p: angle_of(*p))
    
    # Pick 5 roughly evenly spaced from the 12
    picks = [ring_points_sorted[i] for i in [0, 2, 4, 6, 8]]
    
    selected = list(core) + picks
    
    # Convert to Cartesian
    inner = []
    for q, r in selected:
        x, y = axial_to_cart(q, r)
        inner.append([x, y, 0.0])
    
    # Compute minimum enclosing regular hexagon.
    # For a set of points, the minimum enclosing regular hexagon (with arbitrary rotation) 
    # is non-trivial. Let's use a simpler bound: 
    # The farthest point from the centroid determines a lower bound on L.
    # For each direction, L >= (max projection) / (sqrt(3)/2) if flat-side facing,
    # or L >= max projection if vertex facing.
    
    # Centroid of inner hexagon centers:
    cx = sum(p[0] for p in inner) / len(inner)
    cy = sum(p[1] for p in inner) / len(inner)
    
    # For a given outer hexagon center (cx, cy) and rotation theta, 
    # the required side length L must satisfy:
    # For each inner hexagon center at (px, py), the farthest vertex of the inner hexagon
    # from the outer center must be within the outer hexagon.
    # 
    # Approximate: the inner hexagon center is at distance d from outer center.
    # The inner hexagon extends up to 1 (circumradius) further in the worst direction.
    # So the effective distance is d + 1.
    # The outer hexagon has apothem = L * sqrt(3) / 2.
    # For the inner hexagon to fit, we need the inner hexagon's boundary to not cross the outer.
    # 
    # Conservative: L * sqrt(3) / 2 >= max over all inner vertices of (distance from outer center in the normal direction).
    
    # Let's just compute for a few candidate L values and check.
    
    # Lower bound: max distance from centroid to any inner vertex, divided by (sqrt(3)/2) for flat-side.
    max_dist = 0
    for (px, py, ang) in inner:
        for (vx, vy) in base_vertices:
            ca = math.cos(math.radians(ang))
            sa = math.sin(math.radians(ang))
            rx = px + ca*vx - sa*vy
            ry = py + sa*vx + ca*vy
            d = math.sqrt((rx-cx)**2 + (ry-cy)**2)
            max_dist = max(max_dist, d)
    
    # For a regular hexagon, if a point is at distance d from center along a vertex direction,
    # the required L is d. If along an edge-normal direction, required L is d / (sqrt(3)/2) = 2d/sqrt(3).
    # The worst case is the edge-normal direction: L >= 2 * max_dist / sqrt(3).
    L_lower = 2 * max_dist / SQRT3
    
    # Try L values from L_lower upward
    L = math.ceil(L_lower * 100) / 100.0
    L += 0.01  # small buffer
    
    # Check if this works with the current configuration
    # For now, just return a valid (if not optimal) solution.
    
    # Let's use a larger L to be safe, then we can optimize later.
    L = max_dist / (SQRT3 / 2) + 0.1  # apothem = L * sqrt(3)/2 >= max_dist
    
    # Verify
    valid = check_packing(inner, (cx, cy), L, 0.0)
    
    if not valid:
        # Increase L until valid
        for _ in range(100):
            L += 0.05
            if time.time() > deadline:
                break
            valid = check_packing(inner, (cx, cy), L, 0.0)
            if valid:
                break
    
    return inner, (cx, cy), L, 0.0

# EVOLVE-BLOCK-END
