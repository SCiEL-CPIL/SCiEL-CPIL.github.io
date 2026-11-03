import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint(poly1, poly2):
        """
        Fast check for disjointness of two convex hexagons.
        Returns True if disjoint, False if overlapping.
        Uses Separating Axis Theorem (SAT) logic implicitly via vertex/edge checks.
        For speed, we first do a bounding box check.
        """
        min_x1, min_y1 = np.min(poly1, axis=0)
        max_x1, max_y1 = np.max(poly1, axis=0)
        min_x2, min_y2 = np.min(poly2, axis=0)
        max_x2, max_y2 = np.max(poly2, axis=0)
        
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
            return True

        # Point in Polygon check (Ray casting)
        def point_in_poly(pt, poly):
            x, y = pt
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                    inside = not inside
            return inside

        # Check vertices of v1 in v2
        for pt in poly1:
            if point_in_poly(pt, poly2):
                return False
        
        # Check vertices of v2 in v1
        for pt in poly2:
            if point_in_poly(pt, poly1):
                return False

        # Check edge intersections
        def segments_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        for i in range(6):
            p1 = poly1[i]
            p2 = poly1[(i+1)%6]
            for j in range(6):
                p3 = poly2[j]
                p4 = poly2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        
        return True

    # Strategy:
    # 1. Generate candidate configurations using a structured search.
    # 2. For each configuration, verify validity and calculate score.
    # 3. Keep the best valid configuration.
    # 4. Perform a local refinement on the best configuration using Nelder-Mead
    #    but with a strict penalty for overlap to ensure validity.

    best_score = float('inf')
    best_positions = None
    best_angles = None

    # Candidate angles for inner hexagons.
    # Due to symmetry, 0, 30, 60, 90, 120, 150 are the distinct orientations modulo 60.
    # However, 0 and 60 are the same. 30 and 90 are the same.
    # So we really just need to consider relative rotations.
    # Let's try a few global rotations for the whole set, and relative rotations for subsets.
    
    # Pattern 1: All aligned (0 deg)
    # Pattern 2: Alternating 0 and 30
    # Pattern 3: Honeycomb lattice natural orientation (0)
    
    # Let's use a grid search over positions on a hexagonal lattice, 
    # and try a few angle patterns.
    
    sqrt3 = math.sqrt(3)
    
    # Generate a dense set of candidate positions on a hex lattice
    # Spacing 2 (touching) is the minimum. We can allow slight compression? No, disjoint means no overlap.
    # So centers must be at least distance 2 apart (for aligned hexes).
    # If rotated, distance can be slightly less? No, min distance between centers of disjoint unit hexes is 2.
    # Wait, if they are rotated, the "width" changes.
    # The minimum distance between centers of two disjoint unit regular hexagons is 2.
    # This is true regardless of rotation, as the inradius is sqrt(3)/2 and circumradius is 1.
    # Actually, the distance between centers of two touching hexes is 2.
    # If they are rotated, they can still touch at distance 2.
    # Can they be closer? No, the maximum width is 2 (flat-to-flat) and 2 (vertex-to-vertex is 2? No, vertex-to-vertex is 2).
    # Wait, distance between parallel flats is sqrt(3). Distance between opposite vertices is 2.
    # So the "radius" is 1.
    # Two circles of radius 1 need distance 2.
    # Hexagons are inside circles of radius 1.
    # So centers must be at least distance 2 apart?
    # Not necessarily. If one hexagon is rotated 30 degrees relative to the other,
    # the "corners" of one might fit into the "flat" sides of the other?
    # No, the hexagon contains the incircle of radius sqrt(3)/2 ~ 0.866.
    # And is contained in the circumcircle of radius 1.
    # So if centers are distance d apart, and d < 2, the circumcircles overlap.
    # But the hexagons might not.
    # However, for regular hexagons, the minimum distance between centers for disjointness is indeed 2.
    # Proof: The hexagon is the intersection of 6 half-planes.
    # The support function h(u) = max(v.u).
    # For unit hex, h(u) = 1 if u is normal to a flat, h(u) = sqrt(3)/2 if u is along a vertex?
    # No.
    # Let's assume distance 2 is the hard constraint for aligned.
    # For rotated, it might be slightly different, but 2 is a safe lower bound for "touching".
    # Actually, if you rotate one by 30 deg, the "width" in some directions decreases?
    # No, the width of a regular hexagon is constant in all directions? No.
    # Width is distance between parallel support lines.
    # For a regular polygon, width varies.
    # Min width = sqrt(3) (flat to flat). Max width = 2 (vertex to vertex).
    # So if you align flats, you need distance sqrt(3) + sqrt(3)/2? No.
    # If two hexes are aligned, min distance is 2 (vertex to vertex) or sqrt(3) (flat to flat)?
    # If they touch flat-to-flat, distance is sqrt(3).
    # If they touch vertex-to-vertex, distance is 2.
    # So they can be closer than 2 if they touch flat-to-flat.
    # Distance sqrt(3) ~ 1.732.
    # So the lattice spacing of 2 is NOT the minimum.
    # The minimum center-to-center distance for disjoint unit hexagons is sqrt(3) (when touching flat-to-flat).
    
    # So we should use a lattice with spacing sqrt(3) in the "flat" direction?
    # The honeycomb lattice of touching hexagons has center-to-center distance 2?
    # Let's check.
    # Hexagon side 1.
    # Distance from center to vertex: 1.
    # Distance from center to edge: sqrt(3)/2.
    # If two hexes touch edge-to-edge, the distance between centers is 2 * (sqrt(3)/2) = sqrt(3).
    # Wait, is the honeycomb lattice spacing 2 or sqrt(3)?
    # In a standard honeycomb, each hex touches 6 others.
    # The distance between centers of touching hexes is 2 * apothem = sqrt(3).
    # Let's re-verify.
    # Apothem = sqrt(3)/2.
    # Two apothems = sqrt(3).
    # So the lattice vectors have length sqrt(3).
    # My previous code used spacing 2. That was for "vertex to vertex" touching?
    # Or was it wrong?
    # If I place hexes at distance 2, they are NOT touching. They have a gap.
    # If I place them at distance sqrt(3), they touch flat-to-flat.
    # This is the densest packing.
    
    # So, let's generate a lattice with spacing sqrt(3).
    # Basis vectors for hex lattice (flat-top? pointy-top?):
    # Let's use pointy-top (vertex at 0 deg).
    # Flats are at 30, 90, 150...
    # Neighbors are at angles 0, 60, 120... (vertices) or 30, 90... (flats)?
    # If I touch flat-to-flat, the neighbor is in the direction of the normal to the flat.
    # Normals are at 30, 90, 150, 210, 270, 330.
    # So neighbors are at distance sqrt(3) in these directions.
    
    # Let's generate points on this lattice.
    
    lattice_points = []
    # Basis vectors for the lattice of centers of touching pointy-top hexes:
    # v1 = (0, sqrt(3))
    # v2 = (3/2, sqrt(3)/2)
    # Wait, distance is sqrt(3).
    # Angle of v1 is 90 deg. Length sqrt(3).
    # Angle of v2 is 30 deg. Length sqrt(3).
    # v2 = (sqrt(3)*cos(30), sqrt(3)*sin(30)) = (sqrt(3)*sqrt(3)/2, sqrt(3)*1/2) = (1.5, sqrt(3)/2).
    # v1 = (0, sqrt(3)).
    
    # Let's generate a grid of these.
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5 + j * 0.0 # Wait, v1 is (0, sqrt3), v2 is (1.5, sqrt3/2)
            # x = i * v2x + j * v1x = i * 1.5 + j * 0
            # y = i * v2y + j * v1y = i * sqrt3/2 + j * sqrt3
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
            
    # Sort by distance to origin
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # We need to select 12 points that are mutually disjoint.
    # Since the lattice points are exactly at distance sqrt(3) (touching), any subset is disjoint (touching is allowed? "disjoint" usually means interiors don't overlap. Touching boundaries is okay).
    # If "disjoint" means strictly no intersection, we need to add a tiny epsilon.
    # The prompt says "disjoint regular hexagons". In packing problems, touching is usually allowed.
    # Let's assume touching is allowed.
    
    # Candidate 1: The 12 closest points to the origin on this lattice.
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    # Check validity
    polys = [get_hex_verts(init_positions[i][0], init_positions[i][1], 1.0, 0.0) for i in range(N)]
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not check_disjoint(polys[i], polys[j]):
                valid = False
                break
        if not valid: break
    
    if valid:
        score = calc_min_outer_side(init_positions, init_angles, 0.0)
        if score < best_score:
            best_score = score
            best_positions = init_positions.copy()
            best_angles = init_angles.copy()
            
    # Try other angle patterns
    # Pattern 2: Rotate every other hex by 30 degrees?
    # This might allow denser packing?
    # If we rotate by 30, the "width" in the direction of the normal changes?
    # No, the hexagon is the same shape.
    # But relative rotation might allow "interlocking"?
    # No, convex polygons don't interlock.
    # However, if we rotate, the "touching" distance might change?
    # No, the minimum distance for disjointness is determined by the support functions.
    # For two congruent convex bodies K, min distance is min_u (h_K(u) + h_K(-u)) = min_u (h_K(u) + h_K(u)) = 2 * min_u h_K(u)?
    # No.
    # The minimum distance between two disjoint translates of K is the width of K in the direction of separation.
    # For a regular hexagon, the width varies between sqrt(3) and 2.
    # The minimum width is sqrt(3).
    # So the minimum distance between centers is sqrt(3), achieved when separating along a normal to a flat.
    # This is independent of rotation.
    # So rotating doesn't change the minimum distance.
    # It might change the "wasted space" around the hexes?
    # The outer hexagon is regular.
    # If we align the inner hexes with the outer hex (0 deg), the "corners" of the inner hexes point to the "corners" of the outer hex?
    # Outer hex normals are at 30, 90...
    # Inner hex (0 deg) normals are at 30, 90...
    # So they are aligned.
    # This is likely optimal.
    
    # Let's try to refine the best configuration found so far.
    if best_positions is not None:
        # Local refinement using Nelder-Mead
        # We will optimize positions and angles.
        # To ensure validity, we use a penalty function.
        
        def objective(params):
            if time.time() > deadline - 5:
                return 1e6
                
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    if penalty > 1.0:
                        break
                    if not check_disjoint(polys[i], polys[j]):
                        # Estimate overlap depth?
                        # Just add a large penalty.
                        # To make it smooth, we could calculate min distance, but that's hard.
                        # Let's just use a fixed penalty for now.
                        penalty += 10.0
                if penalty > 1.0:
                    break
            
            if penalty > 0:
                return penalty + calc_min_outer_side(positions, angles, 0.0)
                
            return calc_min_outer_side(positions, angles, 0.0)
            
        # Initial guess
        x0 = np.concatenate([best_positions.flatten(), best_angles])
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            
            if res.fun < best_score:
                # Verify validity of the result
                new_positions = res.x[:2*N].reshape(N, 2)
                new_angles = res.x[2*N:]
                
                polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint(polys[i], polys[j]):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_score = res.fun
                    best_positions = new_positions
                    best_angles = new_angles
        except Exception as e:
            pass

    # Fallback if no valid configuration was found (should not happen)
    if best_positions is None:
        # Use the simple honeycomb fallback
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        best_positions = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        best_angles = np.zeros(N)
        best_score = calc_min_outer_side(best_positions, best_angles, 0.0)

    final_positions = best_positions
    final_angles = best_angles
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
