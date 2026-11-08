"""Strategy: Force-directed relaxation with binary search on outer side length."""
import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Helper: get vertices of a unit regular hexagon centered at (cx, cy) with rotation angle
    def hex_vertices(cx, cy, angle):
        # Vertices of unit hexagon (side 1) are at distance 1 from center
        # Standard orientation: flat top/bottom or pointy top?
        # Regular hexagon side length s. Distance from center to vertex is s.
        # Distance from center to edge (inradius) is s * sqrt(3)/2.
        # Let's use standard math angles: 0, 60, 120, 180, 240, 300 degrees
        # Rotated by angle.
        angles = np.deg2rad([0, 60, 120, 180, 240, 300] + [angle * 60]) # Wait, angle input is degrees
        # Let's just rotate the base angles
        base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
        rot = angle * np.pi / 180.0
        final_angles = base_angles + rot
        xs = cx + np.cos(final_angles)
        ys = cy + np.sin(final_angles)
        return np.column_stack((xs, ys))

    # Helper: check if a set of points (hexagon centers/vertices) is inside the outer hexagon
    # Outer hexagon: center (0,0), side length S, rotation 0 (flat top? or pointy?)
    # Let's assume outer hexagon is aligned with inner ones for simplicity, or we can just check distance.
    # A point (x,y) is inside a regular hexagon of side S centered at origin (flat top, i.e., vertices at 0, 60...)
    # if max(|x|, |y|*sqrt(3)/2 + |x|/2) <= S * sqrt(3)/2 ? No, that's for specific orientations.
    
    # Let's define the outer hexagon with "pointy top" (vertices at 90, 150, 210, 270, 330, 30) or "flat top"?
    # The problem says "regular hexagon". Orientation of outer doesn't matter for score, only side length.
    # Let's fix outer orientation to 0 (vertex at angle 0).
    # Inradius of outer hexagon with side S is R_out = S * sqrt(3) / 2.
    # Distance from center to any edge is R_out.
    # A point is inside if its distance to each of the 6 edges is <= R_out.
    
    def is_inside_outer(x, y, S):
        # Outer hexagon vertices at angles 0, 60, ...
        # Edges are perpendicular to angles 30, 90, 150, ...
        # Distance from origin to edge i is S * sqrt(3) / 2.
        # Edge normal angles: 30, 90, 150, 210, 270, 330 degrees.
        normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
        nx = np.cos(normals)
        ny = np.sin(normals)
        # Project point (x,y) onto normals
        dists = x * nx + y * ny
        # All distances must be <= R_out
        R_out = S * np.sqrt(3) / 2
        return np.all(dists <= R_out + 1e-9)

    # Helper: minimum distance between two convex polygons (approximated by vertices)
    # For hexagons, checking vertex-to-edge and edge-to-edge is robust, but for speed,
    # we can check distance between all pairs of vertices and see if it's less than the "effective" separation.
    # Actually, two unit hexagons are disjoint if the distance between their centers is sufficient?
    # No, rotation matters.
    # Simplest robust check: compute distance between all vertices of hex A and edges of hex B, and vice versa.
    # Or, even simpler for small N: check distance between all pairs of vertices. If min dist < 0, they overlap?
    # No, vertices can be inside without crossing.
    # Let's use a simpler proxy: The "radius" of a hexagon is 1. If distance between centers > 2, they are disjoint.
    # But they can be closer if rotated.
    # To be safe and fast, let's check the distance between the two sets of vertices.
    # If min distance between any vertex of A and any vertex of B is < epsilon, they might overlap, but not necessarily.
    # Better: check if any vertex of A is inside B or any vertex of B is inside A.
    # AND check if edges intersect.
    # For N=11, this is small. Let's implement a basic SAT (Separating Axis Theorem) or just point-in-poly + edge intersection.
    
    def point_in_hex(px, py, vertices):
        # vertices: (6, 2) array
        # Ray casting
        inside = False
        n = 6
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def edges_intersect(p1, p2, p3, p4):
        # Check if segment p1-p2 intersects p3-p4
        def cross(o, a, b):
            return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        # Collinear cases (rare in floating point relaxation, but good to handle)
        if d1 == 0 and point_on_segment(p3, p4, p1): return True
        if d2 == 0 and point_on_segment(p3, p4, p2): return True
        if d3 == 0 and point_on_segment(p1, p2, p3): return True
        if d4 == 0 and point_on_segment(p1, p2, p4): return True
        return False

    def point_on_segment(p, q, r):
        # r is on pq
        if p[0] <= r[0] <= q[0] or q[0] <= r[0] <= p[0]:
            if p[1] <= r[1] <= q[1] or q[1] <= r[1] <= p[1]:
                return True
        return False

    def are_disjoint(hex1_verts, hex2_verts):
        # Check if any vertex of hex1 is inside hex2 or vice versa
        for v in hex1_verts:
            if point_in_hex(v[0], v[1], hex2_verts):
                return False
        for v in hex2_verts:
            if point_in_hex(v[0], v[1], hex1_verts):
                return False
        
        # Check edge intersections
        for i in range(6):
            p1 = hex1_verts[i]
            p2 = hex1_verts[(i+1)%6]
            for j in range(6):
                p3 = hex2_verts[j]
                p4 = hex2_verts[(j+1)%6]
                if edges_intersect(p1, p2, p3, p4):
                    return False
        return True

    def is_valid_packing(positions, angles, S):
        # positions: (N, 2)
        # angles: (N,)
        verts_list = []
        for i in range(N):
            verts = hex_vertices(positions[i,0], positions[i,1], angles[i])
            verts_list.append(verts)
            
        # Check inside outer
        for i in range(N):
            # Check all vertices of inner hex i
            for v in verts_list[i]:
                if not is_inside_outer(v[0], v[1], S):
                    return False
            # Check center? Not strictly necessary if vertices are inside and convex, but good for sanity
            if not is_inside_outer(positions[i,0], positions[i,1], S):
                return False

        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not are_disjoint(verts_list[i], verts_list[j]):
                    return False
        return True

    def relax(positions, angles, S, max_iters=200, step_size=0.05):
        # Force directed relaxation
        # Forces:
        # 1. Repulsion between overlapping or close hexagons
        # 2. Attraction to center (to keep them compact)
        # 3. Repulsion from outer boundary
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            forces = np.zeros_like(positions)
            verts_list = [hex_vertices(positions[i,0], positions[i,1], angles[i]) for i in range(N)]
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i+1, N):
                    # Calculate minimum distance between the two hexagons
                    # Approximation: distance between centers minus 2? No.
                    # Let's just use a strong repulsion if they are close.
                    # Distance between centers
                    dx = positions[j,0] - positions[i,0]
                    dy = positions[j,1] - positions[i,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    # If they are close, push apart.
                    # Threshold: if dist < 2.0, they might overlap.
                    # Let's use a smooth potential.
                    if dist < 2.5:
                        # Force magnitude increases as dist decreases
                        f = 10.0 * (2.5 - dist) / 2.5
                        # Normalize direction
                        if dist > 1e-6:
                            fx = f * dx / dist
                            fy = f * dy / dist
                        else:
                            # Random push if exactly overlapping
                            fx = np.random.randn() * 0.1
                            fy = np.random.randn() * 0.1
                        forces[i,0] -= fx
                        forces[i,1] -= fy
                        forces[j,0] += fx
                        forces[j,1] += fy
            
            # Boundary repulsion
            for i in range(N):
                # Check distance to boundary for each vertex
                # If a vertex is close to boundary, push center inward
                for v in verts_list[i]:
                    # Check if v is close to boundary
                    # We can compute the distance from v to the outer hexagon boundary.
                    # Approximation: check is_inside_outer. If it fails or is close, push.
                    # To be efficient, let's just check the 6 edge distances.
                    normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
                    nx = np.cos(normals)
                    ny = np.sin(normals)
                    dists = v[0] * nx + v[1] * ny
                    R_out = S * np.sqrt(3) / 2
                    margin = 0.1
                    for k in range(6):
                        if dists[k] > R_out - margin:
                            # Push inward
                            # Direction is opposite to normal
                            push = 5.0 * (dists[k] - (R_out - margin))
                            forces[i,0] -= push * nx[k]
                            forces[i,1] -= push * ny[k]

            # Center attraction (weak)
            forces -= 0.01 * positions
            
            # Update positions
            positions += step_size * forces
            # Decay step size
            step_size *= 0.99

        return positions, angles

    # Initial guess for S
    # N=11. Hexagonal number 1, 7, 19.
    # 11 is between 7 and 19.
    # A 3x3 grid of hexagons (side 1) has width roughly 5?
    # Let's start with S=5.0 and go down.
    
    best_S = 10.0
    best_positions = None
    best_angles = None
    
    # Binary search for minimal S
    S_low = 4.0
    S_high = 10.0
    
    # We will try to find a valid packing for a given S.
    # If valid, we can try smaller S.
    
    # To make it faster, we will run relaxation for a few seconds, check validity.
    # If valid, update best_S.
    # If not, increase S slightly? Or just use binary search assuming monotonicity (which holds: if it fits in S, it fits in S+eps).
    
    # However, relaxation might fail to find a packing even if one exists.
    # So we should use multiple restarts or better initialization.
    
    # Initialization: Hexagonal lattice
    # Centers of a hexagonal packing of unit hexagons.
    # Distance between centers in a honeycomb is 2 * inradius = sqrt(3)? No.
    # Side length 1. Distance between centers of adjacent hexagons is 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732.
    # Let's generate a lattice.
    
    def generate_lattice():
        # 11 hexagons.
        # 1 in center, 6 around, 4 more?
        # Or a 3x3 block minus corners?
        # Let's just place them in a grid.
        pos = []
        # Row 0: 3
        # Row 1: 3
        # Row 2: 3
        # Row 3: 2
        # Total 11.
        
        dx = np.sqrt(3) # horizontal distance between centers in same row? No, that's for pointy top.
        # Let's use the distance between centers of touching hexagons: sqrt(3).
        # If we align them flat-top:
        # Horizontal spacing: 1.5
        # Vertical spacing: sqrt(3)
        # Wait, for flat top hexagons:
        # Width = 2 * side = 2.
        # Height = sqrt(3) * side = sqrt(3).
        # Horizontal distance between centers of adjacent columns: 1.5 * side? No.
        # Let's just use the vertex distance.
        # Center to vertex is 1.
        # If two hexagons touch, distance between centers is 2 * inradius = sqrt(3).
        
        # Let's try a simple grid of centers spaced by sqrt(3).
        # 3x4 grid = 12. Remove one.
        coords = []
        for i in range(3):
            for j in range(4):
                x = i * 1.5 * 2 # ?
                # Let's stick to the known fact: distance between centers is sqrt(3).
                # If we arrange in a rectangular grid, we need to offset rows.
                pass
        
        # Simpler: just scatter randomly near center.
        # Random restarts are more robust for force-directed methods.
        return None

    # Strategy:
    # 1. Set S to a reasonable upper bound (e.g., 6.0).
    # 2. Run relaxation with random initialization.
    # 3. If valid, record S.
    # 4. Binary search S.
    
    # Given the time limit, let's just do a few random restarts with a fixed S that is likely small enough,
    # or a quick binary search.
    
    # Let's try S = 5.0 first.
    # If we can't find it, try 5.5, 6.0.
    # If we find it, try 4.5, 4.0.
    
    # To maximize score improvement, we want the smallest S.
    
    # Let's do a coarse search:
    S_candidates = [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.5, 6.0]
    
    found_valid = False
    best_result = None
    
    for S in S_candidates:
        if time.time() > deadline - 5.0:
            break
            
        # Try a few restarts for this S
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization
            positions = np.random.randn(N, 2) * 1.0
            angles = np.random.uniform(0, 60, N) # Random rotations
            
            # Relax
            positions, angles = relax(positions, angles, S, max_iters=100, step_size=0.1)
            
            # Check validity
            if is_valid_packing(positions, angles, S):
                found_valid = True
                best_result = (positions, angles, S)
                # Since we are iterating S in increasing order, the first valid S is the best (smallest)
                # So we can break out of both loops if we found a valid one for the smallest S?
                # No, we want the SMALLEST S. So if we start from small S and go up, the first one we find is the best.
                break
        if found_valid:
            break
            
    if not found_valid:
        # Fallback: return a large S to ensure validity, even if score is bad.
        # Or return the seed program's result.
        # Let's try one more time with a larger S and more iterations.
        S = 8.0
        positions = np.random.randn(N, 2) * 2.0
        angles = np.zeros(N)
        positions, angles = relax(positions, angles, S, max_iters=500, step_size=0.1)
        if is_valid_packing(positions, angles, S):
            best_result = (positions, angles, S)
        else:
            # Give up, return seed-like result
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0

    positions, angles, S = best_result
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([positions[i,0], positions[i,1], angles[i]])
    
    return inner, (0.0, 0.0), S, 0.0
