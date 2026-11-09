"""Packing 12 unit hexagons in minimal regular hexagon using hexagonal cluster."""
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 1. Generate centers for a tight cluster of 12 hexagons.
    # A full cluster of radius 1 has 7 hexagons (center + 6 neighbors).
    # We need 12. We can take the 7 and add 5 more from the next ring.
    # To keep symmetry and compactness, we pick a subset of the ring-2 hexagons.
    # However, a simpler robust approach for "minimizing outer side length" 
    # with regular hexagons is often to use the densest packing locally.
    
    # Let's use the centers of a hexagonal grid.
    # Axial coordinates (q, r).
    # Center: (0,0)
    # Ring 1: 6 hexagons.
    # Ring 2: 12 hexagons.
    # Total 19. We pick 12.
    # To minimize the bounding box, we should pick a symmetric subset.
    # A known good configuration for small N is often a "hexagonal number" shape 
    # if N is a hex number (1, 7, 19...). 12 is not.
    # Closest is 7 or 19. 
    # If we use 12, we might use the 7 center ones + 5 from the next ring.
    # Or, we can use a rectangular-ish arrangement if it fits better?
    # No, hexagonal packing is generally denser.
    
    # Let's define the 12 centers.
    # We will use the 7 centers of the radius-1 hexagon.
    # And 5 centers from the radius-2 hexagon, chosen to maintain some symmetry.
    # Actually, let's just generate the 12 smallest distance from center?
    # No, that's 1 (dist 0), 6 (dist 1), 12 (dist 2). 
    # The 12th hexagon is in the second ring.
    # Let's pick the 7 from ring 0/1 and 5 from ring 2.
    
    sqrt3 = math.sqrt(3.0)
    
    # Convert axial (q, r) to cartesian (x, y) for flat-top hexagons
    def axial_to_cart(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return x, y

    # Get all centers in rings 0, 1, and 2
    candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Distance in hex steps is max(|q|, |r|, |q+r|)
            dist = max(abs(q), abs(r), abs(q+r))
            if dist <= 2:
                x, y = axial_to_cart(q, r)
                candidates.append((dist, q, r, x, y))
                
    # Sort by distance, then by angle to break ties consistently?
    # We want the 12 closest.
    # There are 1 at dist 0, 6 at dist 1, 12 at dist 2.
    # We need 12 total. So we take all 7 from dist <= 1.
    # And we need 5 more from dist 2.
    # Which 5? To minimize the bounding box, we should probably pick them symmetrically.
    # The 12 in ring 2 form a hexagon. Picking 5 breaks symmetry.
    # Maybe a better packing for 12 is not a subset of the infinite grid?
    # But for a first strong move, using a subset of the grid is safe and valid.
    # Let's pick the 5 that are "most inward" or just the first 5 in sorted order?
    # Actually, any 5 will work, but the bounding box will be determined by the furthest ones.
    # To minimize the max radius, we just need to ensure we don't pick outliers if possible, 
    # but all dist 2 are same distance from origin.
    # However, the *shape* matters for the enclosing hexagon.
    
    # Let's try a different specific configuration often cited for N=12:
    # It might be better to use a 2x2x3 block or similar?
    # Let's stick to the grid subset for simplicity and correctness.
    # We'll select the 7 closest, and then 5 more.
    # To be safe and symmetric, let's just take the first 12 from a sorted list by (dist, angle).
    
    candidates.sort(key=lambda c: (c[0], math.atan2(c[4], c[3])))
    selected = candidates[:12]
    
    inner = []
    for c in selected:
        # c is (dist, q, r, x, y)
        inner.append([c[3], c[4], 0.0])
        
    # Now compute the minimal enclosing regular hexagon.
    # The outer hexagon has side length L.
    # The inner hexagons have side length 1.
    # We need to find L such that all inner hexagons fit inside the outer one.
    # Since rotations are allowed for inner, and we fixed them at 0, 
    # and outer rotation is free (we can set to 0 wlog if we align), 
    # we can assume outer is aligned with inner (angle 0).
    # Wait, the outer hexagon can rotate. 
    # For a set of points, the minimal enclosing regular hexagon side length 
    # is related to the width of the set in the 3 principal directions of the hexagon.
    
    # A regular hexagon of side L contains a point (x,y) if:
    # |x| <= L
    # |y| <= L * sqrt(3) / 2  ? No.
    # Let's derive the containment condition.
    # A regular hexagon with side L, centered at origin, flat-top:
    # Vertices at (L, 0), (L/2, L*sqrt(3)/2), (-L/2, L*sqrt(3)/2), (-L, 0), (-L/2, -L*sqrt(3)/2), (L/2, -L*sqrt(3)/2).
    # The boundaries are lines:
    # x = L
    # x = -L
    # y = L * sqrt(3) / 2  (top edge? No, top vertex is y_max)
    # Actually, the width (distance between parallel sides) is L * sqrt(3).
    # The "radius" (center to vertex) is L.
    # The "inradius" (center to side midpoint) is L * sqrt(3) / 2.
    
    # For a point (x, y) to be inside a flat-top hexagon of side L:
    # It must satisfy:
    # 1. |x| <= L
    # 2. |y| <= L * sqrt(3) / 2  (This is not sufficient, need to check corners)
    # 3. The slanted sides:
    #    y <= sqrt(3) * (L - |x|) / 2  ? No.
    #    Let's use the standard inequalities for a hexagon defined by 3 pairs of parallel lines.
    #    Directions of normals: 0, 60, 120 degrees.
    #    Normal 0 (x-axis): |x| <= L * sqrt(3) / 2 ? No, that's for pointy-top or different orientation.
    
    # Let's use the property: A regular hexagon of side L is the intersection of 3 strips of width L*sqrt(3).
    # The strips are perpendicular to the vectors (1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2).
    # Wait, the distance between opposite sides is L*sqrt(3).
    # So for direction u, the projection of any point p must satisfy |p . u| <= L * sqrt(3) / 2.
    # The 3 directions u are:
    # u1 = (1, 0)
    # u2 = (1/2, sqrt(3)/2)
    # u3 = (-1/2, sqrt(3)/2)
    # Wait, are these the normals?
    # For a flat-top hexagon, the sides are horizontal. So one normal is (0,1).
    # The side length is L. The distance from center to horizontal side is L * sqrt(3) / 2.
    # So |y| <= L * sqrt(3) / 2.
    # The other sides are slanted.
    # Normal to slanted side: (sqrt(3)/2, 1/2) ?
    # Let's check the vertex (L, 0).
    # Distance from origin to line passing through (L,0) with normal (sqrt(3)/2, 1/2):
    # L * sqrt(3)/2. Correct.
    # So the conditions for a point (x,y) to be inside a flat-top hexagon of side L are:
    # 1. |y| <= L * sqrt(3) / 2
    # 2. |x * sqrt(3)/2 + y * 1/2| <= L * sqrt(3) / 2
    # 3. |x * sqrt(3)/2 - y * 1/2| <= L * sqrt(3) / 2
    
    # We need to find L such that for ALL inner hexagons, ALL their vertices satisfy these.
    # Since inner hexagons are rotated by 0, their vertices are fixed relative to their center.
    # Vertices of a unit flat-top hexagon at (cx, cy):
    # (cx + 1, cy)
    # (cx + 0.5, cy + sqrt(3)/2)
    # (cx - 0.5, cy + sqrt(3)/2)
    # (cx - 1, cy)
    # (cx - 0.5, cy - sqrt(3)/2)
    # (cx + 0.5, cy - sqrt(3)/2)
    
    # We can simply calculate the required L for each vertex of each inner hexagon,
    # taking the maximum over all vertices and all hexagons.
    # For a given vertex (vx, vy), the required L is determined by the max of the 3 constraints:
    # L >= 2 * |vy| / sqrt(3)
    # L >= 2 * |vx * sqrt(3)/2 + vy * 1/2| / sqrt(3)
    # L >= 2 * |vx * sqrt(3)/2 - vy * 1/2| / sqrt(3)
    
    min_L = 0.0
    for hx, hy, ha in inner:
        # ha is 0.0, so we use standard vertices
        verts = [
            (hx + 1.0, hy),
            (hx + 0.5, hy + sqrt3 / 2.0),
            (hx - 0.5, hy + sqrt3 / 2.0),
            (hx - 1.0, hy),
            (hx - 0.5, hy - sqrt3 / 2.0),
            (hx + 0.5, hy - sqrt3 / 2.0)
        ]
        for vx, vy in verts:
            c1 = 2.0 * abs(vy) / sqrt3
            c2 = 2.0 * abs(vx * sqrt3 / 2.0 + vy * 0.5) / sqrt3
            c3 = 2.0 * abs(vx * sqrt3 / 2.0 - vy * 0.5) / sqrt3
            L_req = max(c1, c2, c3)
            if L_req > min_L:
                min_L = L_req
                
    # The outer hexagon can be centered at (0,0) and rotated 0.
    # We could potentially improve by rotating the outer hexagon, 
    # but for this specific symmetric cluster, 0 is likely optimal or close.
    # Given time constraints, we return this analytically derived L.
    
    return inner, (0.0, 0.0), min_L, 0.0
# EVOLVE-BLOCK-END
