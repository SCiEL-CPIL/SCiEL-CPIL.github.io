"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 120.0  # safety margin

    # Geometry helpers
    def hex_vertices(cx, cy, s, angle_deg=0.0):
        """Return vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + i * math.pi / 3.0
            verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
        return verts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4 (including endpoints)."""
        c1 = cross(p1, p2, p3)
        c2 = cross(p1, p2, p4)
        c3 = cross(p3, p4, p1)
        c4 = cross(p3, p4, p2)
        if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
           ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
            return True
        # Collinear cases
        if abs(c1) < 1e-9 and on_segment(p1, p2, p3): return True
        if abs(c2) < 1e-9 and on_segment(p1, p2, p4): return True
        if abs(c3) < 1e-9 and on_segment(p3, p4, p1): return True
        if abs(c4) < 1e-9 and on_segment(p3, p4, p2): return True
        return False

    def on_segment(a, b, p):
        """Check if point p lies on segment a-b."""
        return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
                min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)

    def point_in_polygon(pt, poly):
        """Ray casting algorithm."""
        x, y = pt
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def hexagons_overlap(v1, v2, tol=1e-6):
        """Check if two hexagons (given as vertex lists) overlap."""
        # Check if any vertex of one is inside the other
        for pt in v1:
            if point_in_polygon(pt, v2):
                return True
        for pt in v2:
            if point_in_polygon(pt, v1):
                return True
        # Check edge intersections
        for i in range(6):
            for j in range(6):
                if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                    return True
        return False

    def all_contained(inner_verts_list, outer_verts):
        """Check if all inner hexagons are contained in the outer hexagon."""
        for iv in inner_verts_list:
            for pt in iv:
                if not point_in_polygon(pt, outer_verts):
                    return False
        # Also check that no edges cross
        for iv in inner_verts_list:
            for i in range(6):
                for j in range(6):
                    if segments_intersect(iv[i], iv[(i+1)%6], outer_verts[j], outer_verts[(j+1)%6]):
                        return False
        return True

    def no_overlap(inner_verts_list):
        """Check that no two inner hexagons overlap."""
        for i in range(len(inner_verts_list)):
            for j in range(i+1, len(inner_verts_list)):
                if hexagons_overlap(inner_verts_list[i], inner_verts_list[j]):
                    return False
        return True

    def validate(positions, angles, outer_center, outer_side, outer_angle=0.0):
        """Full validation: containment + no overlap."""
        outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
        inner_verts = [hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(len(positions))]
        if not all_contained(inner_verts, outer_verts):
            return False
        if not no_overlap(inner_verts):
            return False
        return True

    def compute_min_outer_side(positions, angles):
        """Compute the minimum side length of a hexagon (axis-aligned or optimally rotated) that contains all inner hexagons."""
        # We'll try several rotations of the outer hexagon and find the minimum side length
        best_side = float('inf')
        for outer_angle in range(0, 30, 1):  # Try 0,1,2,...,29 degrees
            # For a given outer hexagon orientation, we need to find the minimum side length
            # such that all vertices of all inner hexagons are inside.
            # For a regular hexagon with center (cx,cy), side s, rotation a:
            # A point (x,y) is inside if it satisfies 6 half-plane inequalities.
            # The distance from center to each edge is s*sqrt(3)/2.
            # The edge normals are at angles a + 30° + 60°*k for k=0..5
            
            a = math.radians(outer_angle)
            # Edge normals (outward) for a flat-top hexagon:
            # Actually, for a regular hexagon with vertices at angles a, a+60, ..., a+300:
            # The edges are between consecutive vertices. The outward normal of the edge
            # between vertex at angle theta1 and theta2 is at angle (theta1+theta2)/2 + 90°
            # For vertex at angle a + i*60°, the edge to next vertex has normal at a + i*60° + 90° + 30° = a + i*60° + 120°
            # Wait, let me think more carefully.
            
            # For a regular hexagon centered at origin with side s, vertices at angles a + i*60°, i=0..5:
            # Vertex i: (s*cos(a+i*pi/3), s*sin(a+i*pi/3))
            # Edge i (from vertex i to vertex i+1): midpoint at angle a + i*pi/3 + pi/6, distance s*sqrt(3)/2 from center
            # Outward normal direction: a + i*pi/3 + pi/6
            
            # For a point P to be inside: dot(P - center, normal_i) <= s*sqrt(3)/2 for all i
            
            # The minimum s such that all points are inside:
            # s >= max over all points and all i of dot(P - center, normal_i) / (sqrt(3)/2)
            # But we also need to choose the center. For now, let's fix center at origin and find min s.
            # Actually, we should optimize center too. But for simplicity, let's use the centroid of all vertices.
            
            # Collect all vertices of all inner hexagons
            all_pts = []
            for i in range(len(positions)):
                iv = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
                all_pts.extend(iv)
            
            # For a given orientation, the minimal enclosing hexagon (centered optimally) 
            # has side length = max over edges of (max dot product - min dot product) / (sqrt(3))
            # Because the width in direction normal_i is 2*s*sqrt(3)/2 = s*sqrt(3)
            # So s >= width_i / sqrt(3) for each i
            # And the optimal center shifts to equalize the max and min.
            
            normals = []
            for i in range(6):
                na = a + i * math.pi / 3.0 + math.pi / 6
                normals.append((math.cos(na), math.sin(na)))
            
            max_s = 0.0
            for nx, ny in normals:
                dots = [px * nx + py * ny for px, py in all_pts]
                width = max(dots) - min(dots)
                s_needed = width / math.sqrt(3.0)
                max_s = max(max_s, s_needed)
            
            best_side = min(best_side, max_s)
        
        return best_side

    # Honeycomb initial arrangement
    # For unit hexagons (side 1), the distance between centers of adjacent hexagons is 2*sqrt(3)/2 = sqrt(3)
    # Wait: for a regular hexagon of side s, the distance from center to edge is s*sqrt(3)/2.
    # Two hexagons sharing an edge have centers 2 * s*sqrt(3)/2 = s*sqrt(3) apart.
    # For s=1, that's sqrt(3) ≈ 1.732.
    
    # Honeycomb lattice vectors:
    # d1 = (sqrt(3), 0)
    # d2 = (sqrt(3)/2, 1.5)  -- this gives 60 degree angle
    # Actually, for a honeycomb of flat-top hexagons:
    # Adjacent centers are at distance sqrt(3) in directions 0°, 60°, 120°, 180°, 240°, 300°
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate honeycomb positions: center + 6 neighbors + 4 more
    positions = []
    angles = []
    
    # Central hexagon
    positions.append((0.0, 0.0))
    angles.append(0.0)
    
    # 6 neighbors at distance sqrt(3) in 6 directions
    for k in range(6):
        ang = k * math.pi / 3.0
        positions.append((sqrt3 * math.cos(ang), sqrt3 * math.sin(ang)))
        angles.append(0.0)
    
    # 4 more to make 11: add some from the next ring
    # Next ring positions are at 2*sqrt(3) in directions 0,60,... or sqrt(3) in directions 30,90,...
    # Let's add 4 more strategically
    # Add at (2*sqrt(3), 0), (0, 2*sqrt(3)), (-2*sqrt(3), 0), (0, -2*sqrt(3))
    for k in range(4):
        ang = k * math.pi / 2.0
        positions.append((2*sqrt3 * math.cos(ang), 2*sqrt3 * math.sin(ang)))
        angles.append(0.0)
    
    # Compute initial outer side
    outer_side = compute_min_outer_side(positions, angles)
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    best_side = outer_side
    best_positions = positions[:]
    best_angles = angles[:]
    best_outer_center = outer_center
    best_outer_angle = outer_angle
    
    # Simulated annealing
    import random
    random.seed(42)
    
    T = 0.1
    cool_rate = 0.999
    min_T = 0.0001
    
    step = 0.1
    
    while time.time() < deadline - 10.0:  # Leave 10s margin
        if T < min_T:
            T = min_T
        
        # Pick a random hexagon and perturb it
        idx = random.randrange(len(positions))
        new_positions = positions[:]
        new_angles = angles[:]
        
        dx = (random.random() - 0.5) * step
        dy = (random.random() - 0.5) * step
        da = (random.random() - 0.5) * step * 10
        
        new_positions[idx] = (positions[idx][0] + dx, positions[idx][1] + dy)
        new_angles[idx] = angles[idx] + da
        
        # Validate the new configuration
        if validate(new_positions, new_angles, outer_center, outer_side, outer_angle):
            new_side = compute_min_outer_side(new_positions, new_angles)
            
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions[:]
                best_angles = new_angles[:]
                best_outer_center = outer_center
                best_outer_angle = outer_angle
            
            # Accept if better or with probability exp(-delta/T)
            delta = new_side - outer_side
            if delta < 0 or random.random() < math.exp(-delta / T):
                positions = new_positions
                angles = new_angles
                outer_side = new_side
        else:
            # Reject: too costly to recompute, just skip
            pass
        
        T *= cool_rate
    
    # Return the best configuration found
    # Convert positions to the required format: [x, y, angle_degrees]
    inner = []
    for i in range(len(best_positions)):
        inner.append([best_positions[i][0], best_positions[i][1], math.degrees(best_angles[i])])
    
    return inner, best_outer_center, best_side, math.degrees(best_outer_angle)
# EVOLVE-BLOCK-END

