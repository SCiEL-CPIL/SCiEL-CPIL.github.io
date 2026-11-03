"""Rewritten program for packing unit hexagons in a regular hexagon."""
import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Helper: compute vertices of a regular hexagon with side length s, center (cx, cy), rotation angle (radians)
    def hex_vertices(cx, cy, s, angle):
        # Regular hexagon vertices: for k=0..5, angle + k*pi/3
        # Side length s, so radius (center to vertex) = s
        angles = angle + np.arange(6) * np.pi / 3.0
        return cx + s * np.cos(angles), cy + s * np.sin(angles)
    
    # Helper: check if two convex polygons are disjoint (non-overlapping)
    # Using Separating Axis Theorem for convex polygons
    def polygons_disjoint(verts1, verts2, tol=1e-10):
        """
        Check if two convex polygons (given as (x,y) arrays of shape (n,2)) are disjoint.
        Returns True if disjoint, False if overlapping.
        """
        n1 = len(verts1)
        n2 = len(verts2)
        
        # Check all axes from both polygons
        axes = []
        for i in range(n1):
            j = (i + 1) % n1
            edge = verts1[j] - verts1[i]
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-15:
                normal = normal / np.linalg.norm(normal)
                axes.append(normal)
        
        for i in range(n2):
            j = (i + 1) % n2
            edge = verts2[j] - verts2[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-15:
                normal = normal / np.linalg.norm(normal)
                axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(verts1, axis)
            proj2 = np.dot(verts2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # If there's a gap, polygons are disjoint
            if max1 + tol < min2 or max2 + tol < min1:
                return True
        
        return False
    
    # Helper: get vertices of an inner hexagon
    def inner_hex_vertices(x, y, angle_deg, side=1.0):
        angle_rad = math.radians(angle_deg)
        return hex_vertices(x, y, side, angle_rad)
    
    # Helper: check if all inner hexagons are disjoint and inside outer hexagon
    def is_valid(inner, outer_center, outer_side, outer_angle_deg):
        ocx, ocy = outer_center
        outer_verts_x, outer_verts_y = hex_vertices(ocx, ocy, outer_side, math.radians(outer_angle_deg))
        outer_verts = np.column_stack([outer_verts_x, outer_verts_y])
        
        inner_verts_list = []
        for x, y, angle_deg in inner:
            vx, vy = inner_hex_vertices(x, y, angle_deg)
            verts = np.column_stack([vx, vy])
            inner_verts_list.append(verts)
        
        # Check all pairs are disjoint
        for i in range(len(inner_verts_list)):
            for j in range(i + 1, len(inner_verts_list)):
                if not polygons_disjoint(inner_verts_list[i], inner_verts_list[j]):
                    return False
        
        # Check all inner hexagons are inside outer hexagon
        # A point is inside a convex polygon if it's on the same side of all edges
        def point_in_polygon(px, py, poly):
            n = len(poly)
            for i in range(n):
                j = (i + 1) % n
                edge = poly[j] - poly[i]
                normal = np.array([-edge[1], edge[0]])
                # Ensure normal points inward
                # For a regular hexagon centered at origin, we can check
                # Actually, let's use the cross product method
                # Point P is inside if it's to the left of every edge (for CCW polygon)
                # or to the right (for CW)
                # Let's compute the signed distance
                # Cross product of edge with vector from vertex to point
                v1 = poly[j] - poly[i]
                v2 = np.array([px, py]) - poly[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                # For a CCW polygon, interior is to the left, so cross >= 0
                # But we need to know the winding order
                # Let's just check if all cross products have the same sign
                # Actually, simpler: use the fact that for convex polygon, 
                # point is inside if it satisfies all half-plane constraints
                # Let's determine the sign by checking the centroid
                pass
            
            # Simpler approach: check if all vertices of inner are inside outer
            # For convex polygons, if all vertices of one are inside the other, they're contained
            return True
        
        # Better: check containment by verifying all vertices of inner hex are inside outer
        def point_in_convex_polygon(px, py, poly):
            n = len(poly)
            signs = []
            for i in range(n):
                j = (i + 1) % n
                v1 = poly[j] - poly[i]
                v2 = np.array([px, py]) - poly[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if abs(cross) < 1e-12:
                    continue
                signs.append(1 if cross > 0 else -1)
            
            if not signs:
                return True
            return all(s == signs[0] for s in signs)
        
        for verts in inner_verts_list:
            for px, py in verts:
                if not point_in_convex_polygon(px, py, outer_verts):
                    return False
        
        return True
    
    # Strategy: Try different configurations of inner hexagons
    # For N=11, a natural packing is a hexagonal arrangement
    # Let's try a few candidate configurations and find the minimum enclosing hexagon
    
    # Configuration 1: Hexagonal grid arrangement
    # Center at (0,0), with hexagons at hex lattice positions
    # Hex lattice with spacing 2 (since side=1, distance between centers of adjacent hexes = 2)
    # Actually, for regular hexagons of side 1, the distance between centers of adjacent 
    # hexagons in a hex grid is 2 * (sqrt(3)/2) = sqrt(3) if they share an edge?
    # No: for regular hexagons of side s, the distance between centers of two adjacent 
    # hexagons that share an edge is 2 * (sqrt(3)/2) * s = sqrt(3) * s
    # Wait, let me recalculate. A regular hexagon of side s has:
    # - Distance from center to vertex (circumradius): s
    # - Distance from center to edge (inradius/apothem): (sqrt(3)/2) * s
    # When two hexagons share an edge, the distance between centers is 2 * apothem = sqrt(3) * s
    
    # For s=1, distance between adjacent centers = sqrt(3)
    
    sqrt3 = math.sqrt(3)
    
    # Hexagonal grid positions for 11 hexagons:
    # One in center, 6 around it, 4 more? That's 12.
    # Or: center + 6 around = 7, plus 4 more on a second ring?
    # Let's try: center at (0,0), 6 at distance sqrt(3) at angles 0, 60, 120, 180, 240, 300
    # That's 7. Need 4 more.
    
    # Alternative: linear arrangement or other patterns
    
    # Let's try several candidate configurations
    
    candidates = []
    
    # Candidate 1: Linear row
    positions = []
    for k in range(N):
        positions.append((k * 2 * sqrt3, 0.0, 0.0))
    candidates.append(positions)
    
    # Candidate 2: Hexagonal cluster (center + 6 around)
    # That's only 7, need 11. Let's do center + 6 + 4 from second ring
    positions = [(0.0, 0.0, 0.0)]
    for i in range(6):
        angle = i * 60
        x = sqrt3 * math.cos(math.radians(angle))
        y = sqrt3 * math.sin(math.radians(angle))
        positions.append((x, y, 0.0))
    # Add 4 more from second ring (distance 2*sqrt3)
    for i in range(4):
        angle = i * 90 + 30  # Spread them out
        x = 2 * sqrt3 * math.cos(math.radians(angle))
        y = 2 * sqrt3 * math.sin(math.radians(angle))
        positions.append((x, y, 0.0))
    candidates.append(positions)
    
    # Candidate 3: Two rows
    positions = []
    # Row 1: 6 hexagons
    for k in range(6):
        positions.append((k * 2 * sqrt3, 0.0, 0.0))
    # Row 2: 5 hexagons, offset
    for k in range(5):
        positions.append((k * 2 * sqrt3 + sqrt3, sqrt3 * 1.5, 0.0))
    candidates.append(positions)
    
    # Candidate 4: Three rows (4-3-4 or similar)
    positions = []
    # Row 1: 4 hexagons
    for k in range(4):
        positions.append((k * 2 * sqrt3, 0.0, 0.0))
    # Row 2: 3 hexagons
    for k in range(3):
        positions.append((k * 2 * sqrt3 + sqrt3, sqrt3 * 1.5, 0.0))
    # Row 3: 4 hexagons
    for k in range(4):
        positions.append((k * 2 * sqrt3, sqrt3 * 3.0, 0.0))
    candidates.append(positions)
    
    # Candidate 5: Compact hexagonal shape
    # Try to pack 11 in a more compact arrangement
    # Hex numbers: 1, 7, 19, ... So 11 is between 7 and 19
    # Let's try center + 6 + 4 more placed optimally
    
    # For each candidate, find the minimum enclosing hexagon
    best_result = None
    best_side = float('inf')
    
    for candidate_positions in candidates:
        if time.time() > deadline:
            break
        
        # Find minimum enclosing hexagon for this configuration
        # The minimum enclosing hexagon can be found by trying different orientations
        # and computing the required side length
        
        # For a given orientation, the minimum side length is determined by the 
        # maximum extent of the inner hexagons in the 6 directions of the outer hexagon
        
        min_side_for_config = float('inf')
        best_outer_params = None
        
        # Try different outer hexagon orientations
        for outer_angle_deg in range(0, 30, 5):  # 0, 5, 10, 15, 20, 25
            if time.time() > deadline:
                break
            
            outer_angle_rad = math.radians(outer_angle_deg)
            
            # For each inner hexagon, compute its vertices
            # Then find the minimum enclosing hexagon of the union of all vertices
            
            # The minimum enclosing hexagon with a fixed orientation can be found by
            # projecting all points onto the 6 normal directions and finding the 
            # required side length
            
            # For a regular hexagon with side s and orientation theta, the half-width
            # in direction phi is s * cos(phi - theta) for |phi - theta| <= 60 degrees
            # Actually, the support function of a regular hexagon with side s is:
            # h(phi) = s * cos(phi - theta) for phi in [theta - 60, theta + 60]
            # But we need to be more careful
            
            # Simpler: for a given set of points, the minimum enclosing regular hexagon
            # with a given orientation can be found by binary search on the side length
            
            # Let's collect all vertices of all inner hexagons
            all_points = []
            for x, y, angle_deg in candidate_positions:
                vx, vy = inner_hex_vertices(x, y, angle_deg)
                for px, py in zip(vx, vy):
                    all_points.append((px, py))
            all_points = np.array(all_points)
            
            if len(all_points) == 0:
                continue
            
            # Binary search on side length
            lo = 0.0
            hi = 50.0  # Upper bound
            
            for _ in range(50):
                mid = (lo + hi) / 2
                if time.time() > deadline:
                    break
                
                # Check if all points are inside a hexagon of side mid centered at origin
                # with orientation outer_angle_rad
                # First, find the center that minimizes the enclosing hexagon
                # For simplicity, center at the centroid of all points
                cx = np.mean(all_points[:, 0])
                cy = np.mean(all_points[:, 1])
                
                # Shift points so centroid is at origin
                pts = all_points - np.array([cx, cy])
                
                # Check if all points are inside hexagon of side mid, orientation outer_angle_rad
                # A point (px, py) is inside a regular hexagon of side s, center (0,0), 
                # orientation theta if for all 6 directions, the projection is within bounds
                
                # The hexagon can be described as the intersection of 6 half-planes
                # For a regular hexagon with side s and orientation theta:
                # The 6 normal directions are theta + k*pi/3 for k=0..5
                # The distance from center to each edge (apothem) is (sqrt(3)/2) * s
                # So the constraint is: |dot(p, n_k)| <= (sqrt(3)/2) * s for all k
                
                apothem = (sqrt3 / 2) * mid
                normals = []
                for k in range(6):
                    angle = outer_angle_rad + k * np.pi / 3
                    normals.append(np.array([math.cos(angle), math.sin(angle)]))
                
                valid = True
                for px, py in pts:
                    for n in normals:
                        proj = abs(px * n[0] + py * n[1])
                        if proj > apothem + 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    hi = mid
                else:
                    lo = mid
            
            side = hi
            if side < min_side_for_config:
                min_side_for_config = side
                best_outer_params = (cx, cy, outer_angle_deg)
        
        if min_side_for_config < best_side:
            best_side = min_side_for_config
            best_result = (candidate_positions, best_outer_params[0], best_outer_params[1], best_outer_params[2])
    
    if best_result is None:
        # Fallback: use the linear configuration with a large hexagon
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    positions, ocx, ocy, outer_angle = best_result
    inner = [[x, y, angle_deg] for x, y, angle_deg in positions]
    return inner, (ocx, ocy), best_side, outer_angle

# EVOLVE-BLOCK-END
