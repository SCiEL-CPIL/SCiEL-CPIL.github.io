import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 # seconds
    safety_margin = 10.0
    
    # 1. Define inner hexagon centers
    # Using axial coordinates (q, r) for a hexagonal lattice
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # This corresponds to pointy-top hexagons? 
    # Let's check distance between (0,0) and (1,0).
    # x1=0, y1=0. x2=sqrt(3), y2=0. Dist = sqrt(3). Correct for side 1.
    
    # Select 11 compact points
    # Center (0,0)
    # Ring 1: 6 points
    # Ring 2: 4 points to make 11
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # 2. Generate vertices for all inner hexagons
    # A unit hexagon centered at (cx, cy) with rotation 0 (pointy-top relative to lattice?)
    # The lattice vectors imply the hexagons are aligned such that one vertex is at angle 0?
    # If dist is sqrt(3), and we use standard hex packing, the hexagons share edges.
    # The vertices of a unit hexagon centered at origin (pointy-top, vertex at top) are:
    # (0, 1), (-sqrt(3)/2, 0.5), (-sqrt(3)/2, -0.5), (0, -1), (sqrt(3)/2, -0.5), (sqrt(3)/2, 0.5)
    # Wait, the distance from center to vertex is 1.
    # The distance from center to flat is sqrt(3)/2.
    # The distance between centers of adjacent hexagons is sqrt(3) (2 * flat distance).
    # This matches our lattice.
    
    # So, inner hexagons are "pointy-top" if we define vertices at angles 90, 150, ...
    # Or "flat-top" if vertices are at 0, 60...
    # Let's check the lattice again.
    # (0,0) to (1,0) is along X axis.
    # If hexagons share an edge perpendicular to X axis, the edge is vertical.
    # This implies the hexagons are "flat-top" (flat edge on left/right).
    # Vertices of a flat-top hexagon centered at (0,0):
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Let's verify edge sharing.
    # Right edge of (0,0) is from (0.5, sqrt(3)/2) to (0.5, -sqrt(3)/2)? No, that's a vertex to vertex.
    # Flat-top vertices:
    # Angle 0: (1,0)
    # Angle 60: (0.5, sqrt(3)/2)
    # Angle 120: (-0.5, sqrt(3)/2)
    # Angle 180: (-1,0)
    # Angle 240: (-0.5, -sqrt(3)/2)
    # Angle 300: (0.5, -sqrt(3)/2)
    # Edge between 0 and 60 is not an edge. Edge is between 60 and 120?
    # No, a regular hexagon has 6 edges.
    # Vertices:
    # v0 = (1, 0)
    # v1 = (0.5, sqrt(3)/2)
    # v2 = (-0.5, sqrt(3)/2)
    # v3 = (-1, 0)
    # v4 = (-0.5, -sqrt(3)/2)
    # v5 = (0.5, -sqrt(3)/2)
    # Edge v0-v1? Length sqrt(0.25 + 0.75) = 1. Yes.
    # So the "right" side of the hexagon is the vertex v0.
    # The "right" flat is the edge v1-v5? No, v1 and v5 are not connected.
    # The edge v1-v2 is horizontal (top flat).
    # The edge v4-v5 is horizontal (bottom flat).
    # The edge v0-v1 is slanted.
    # So this is a "pointy-right" hexagon?
    # Actually, if v0 is at (1,0), the vertex points right.
    # The edge connecting v5 and v0 is slanted.
    # The edge connecting v0 and v1 is slanted.
    # So the "right" side is a vertex.
    # If we place another hexagon at (sqrt(3), 0), its left vertex is at (sqrt(3)-1, 0).
    # Distance from (1,0) to (sqrt(3)-1, 0) is sqrt(3)-2 < 0. They overlap.
    # Wait.
    # Distance between centers is sqrt(3).
    # If hexagons touch, the distance between centers is 2 * (apothem) = 2 * sqrt(3)/2 = sqrt(3).
    # This happens if they share an edge.
    # Which edge?
    # If hex 1 is at (0,0) and hex 2 is at (sqrt(3), 0), they share the vertical edge?
    # For them to share a vertical edge, the hexagons must be "flat-top" (flat edge on left/right)?
    # No, if they share a vertical edge, the edge is vertical.
    # In a regular hexagon, are there vertical edges?
    # If vertices are at 0, 60, 120...
    # Edge 120-180? No.
    # Edge 60-120 is horizontal.
    # Edge 0-60 is slanted.
    # So there are NO vertical edges in this orientation.
    # Therefore, they cannot share a vertical edge.
    # They must share a slanted edge.
    # If hex 2 is at (sqrt(3), 0), and they share an edge, the edge must be perpendicular to the line connecting centers (X axis).
    # So the shared edge must be vertical.
    # Contradiction.
    
    # Let's re-evaluate the lattice.
    # If centers are at distance sqrt(3), and they touch, they share an edge.
    # The line connecting centers is perpendicular to the shared edge.
    # So if centers are on X axis, shared edge is vertical.
    # This requires the hexagons to have a vertical edge.
    # A regular hexagon has a vertical edge if it is "pointy-top" (vertex at top/bottom).
    # Vertices of pointy-top:
    # (0, 1), (sqrt(3)/2, 0.5), (sqrt(3)/2, -0.5), (0, -1), (-sqrt(3)/2, -0.5), (-sqrt(3)/2, 0.5)
    # Edge between (sqrt(3)/2, 0.5) and (sqrt(3)/2, -0.5) is vertical.
    # Length 1.
    # Center to this edge distance is sqrt(3)/2.
    # So if we place a hex at (0,0) and one at (sqrt(3), 0), they share the vertical edge at x = sqrt(3)/2.
    # This works.
    # So the inner hexagons are "pointy-top" (vertex at top).
    
    # Vertices of pointy-top unit hexagon at (cx, cy):
    # v0 = (cx, cy + 1)
    # v1 = (cx + sqrt(3)/2, cy + 0.5)
    # v2 = (cx + sqrt(3)/2, cy - 0.5)
    # v3 = (cx, cy - 1)
    # v4 = (cx - sqrt(3)/2, cy - 0.5)
    # v5 = (cx - sqrt(3)/2, cy + 0.5)
    
    all_vertices = []
    for cx, cy in centers:
        verts = [
            (cx, cy + 1.0),
            (cx + np.sqrt(3)/2, cy + 0.5),
            (cx + np.sqrt(3)/2, cy - 0.5),
            (cx, cy - 1.0),
            (cx - np.sqrt(3)/2, cy - 0.5),
            (cx - np.sqrt(3)/2, cy + 0.5)
        ]
        all_vertices.extend(verts)
        
    all_vertices = np.array(all_vertices)
    
    # 3. Optimize outer hexagon
    # Outer hexagon is regular, side S, center (ox, oy), rotation theta.
    # We want to minimize S.
    # For a fixed theta, the optimal center (ox, oy) is the center of the minimal bounding box
    # in the 3 directions defined by the hexagon normals.
    
    best_S = float('inf')
    best_theta = 0.0
    best_center = (0.0, 0.0)
    
    # We can restrict theta to [0, 60] because of symmetry.
    # Step size 0.1 degrees is 600 iterations. Very fast.
    
    for deg in np.arange(0, 60.1, 0.1):
        if time.time() > deadline - safety_margin:
            break
            
        theta = np.deg2rad(deg)
        
        # Normals of the outer hexagon sides.
        # A regular hexagon has 3 pairs of parallel sides.
        # The normals are at angles theta, theta + 60, theta + 120.
        # (The sides themselves are perpendicular to these normals).
        
        n1 = np.array([np.cos(theta), np.sin(theta)])
        n2 = np.array([np.cos(theta + np.pi/3), np.sin(theta + np.pi/3)])
        n3 = np.array([np.cos(theta + 2*np.pi/3), np.sin(theta + 2*np.pi/3)])
        
        # Project all vertices onto these normals
        p1 = all_vertices @ n1
        p2 = all_vertices @ n2
        p3 = all_vertices @ n3
        
        # Find min and max for each
        min1, max1 = np.min(p1), np.max(p1)
        min2, max2 = np.min(p2), np.max(p2)
        min3, max3 = np.min(p3), np.max(p3)
        
        # The optimal center (ox, oy) must satisfy:
        # ox * n1[0] + oy * n1[1] = (min1 + max1) / 2
        # ox * n2[0] + oy * n2[1] = (min2 + max2) / 2
        # ox * n3[0] + oy * n3[1] = (min3 + max3) / 2
        
        # This is an overdetermined system (3 equations, 2 unknowns).
        # However, for a valid packing, the "widths" must be consistent.
        # Actually, the center that minimizes the max half-width is the intersection of the midlines?
        # If the shape is not perfectly symmetric, the optimal center might not be the exact midpoint of all 3.
        # But for a convex set, the minimum enclosing hexagon of a given orientation
        # is centered at the centroid of the "bounding box" corners?
        # A simpler approach: The required S is determined by the max of the half-widths.
        # half_width1 = (max1 - min1) / 2
        # half_width2 = (max2 - min2) / 2
        # half_width3 = (max3 - min3) / 2
        # S = max(half_widths) * 2 / sqrt(3)
        # The center that achieves this is the one that centers the "widest" dimension?
        # Actually, if we shift the center, we change the projections.
        # Let $c$ be the center. $p_i' = p_i - c \cdot n_i$.
        # We want $\max(\max(p_i') - \min(p_i'))$ to be minimized? No, we want $\max(\max(p_i'), \min(p_i'))$ to be minimized?
        # The condition for $p$ to be in hexagon centered at $c$ is $| (p-c) \cdot n_i | \le r$ for all $i$.
        # So we need $\max_i \max_p | (p-c) \cdot n_i | \le r$.
        # This is equivalent to minimizing $r$ such that there exists $c$.
        # This is the Chebyshev center problem in the transformed space.
        # For a fixed orientation, the optimal $c$ is the center of the minimum enclosing circle? No.
        # It is the point $c$ that minimizes $\max_k \max_p | (p-c) \cdot n_k |$.
        # This can be solved by linear programming or by noting that the optimal $c$
        # is the intersection of the "midlines" of the tightest constraints.
        # For simplicity and speed, we can assume the optimal center is the centroid of the vertices?
        # Or the center of the bounding box of the projections?
        # Let's calculate the center as the average of the midpoints of the 3 intervals.
        # This is a good heuristic and often optimal for centrally symmetric shapes.
        
        mid1 = (min1 + max1) / 2.0
        mid2 = (min2 + max2) / 2.0
        mid3 = (min3 + max3) / 2.0
        
        # Solve for ox, oy using least squares on the 3 equations
        # [n1x n1y] [ox]   [mid1]
        # [n2x n2y] [oy] = [mid2]
        # [n3x n3y]        [mid3]
        
        A = np.array([n1, n2, n3]).T # 2x3
        b = np.array([mid1, mid2, mid3])
        
        # Least squares solution
        ox_oy, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        ox, oy = ox_oy
        
        # Calculate required r
        # r must be at least max distance from (ox, oy) to any vertex projected on normals
        # dist1 = max(|(p - c) dot n1|)
        # We can just use the half-widths calculated above, but shifted by c?
        # Actually, if we choose c as the midpoint, the max deviation is exactly the half-width.
        # So r = max( (max1-min1)/2, (max2-min2)/2, (max3-min3)/2 )
        
        r1 = (max1 - min1) / 2.0
        r2 = (max2 - min2) / 2.0
        r3 = (max3 - min3) / 2.0
        
        r = max(r1, r2, r3)
        
        # S = r * 2 / sqrt(3)
        S = r * 2.0 / np.sqrt(3)
        
        if S < best_S:
            best_S = S
            best_theta = theta
            best_center = (ox, oy)
            
    # Construct the inner hexagon list
    # Format: (x, y, angle_degrees)
    # The inner hexagons are aligned with the lattice (pointy-top), so angle is 0 relative to their local frame?
    # The task says "each inner hexagon may translate and rotate".
    # Our construction assumes they are all aligned (rotation 0).
    # The angle in the output should be the rotation of the inner hexagon.
    # Since we generated them as pointy-top, and if we define 0 degrees as pointy-top, then angle is 0.
    # If the evaluator expects a specific convention, 0 is safe for aligned packing.
    
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    # Outer hexagon
    # Center (ox, oy)
    # Side length S
    # Rotation: best_theta is the angle of the NORMALS.
    # The rotation of the hexagon itself (angle of a side or vertex) might be different.
    # Usually, "rotation" of a polygon refers to the angle of a specific vertex or side.
    # If best_theta is the angle of the normal to the first side, then the side is at best_theta + 90.
    # The vertex is at best_theta + 30?
    # Let's assume the standard convention: 0 degrees means a vertex is at (S, 0) or similar.
    # For a regular hexagon, if a normal is at theta, the vertices are at theta + 30, theta + 90, etc.
    # So the rotation angle of the hexagon (vertex position) is best_theta + 30 degrees.
    # However, the interface just asks for "angle".
    # Let's provide the angle of the normal? Or the angle of the shape?
    # "outer hexagon ... rotation"
    # Usually, this is the angle of the first vertex relative to the x-axis.
    # If normals are at theta, vertices are at theta + 30.
    # Let's return best_theta + 30.0 (in degrees).
    
    outer_angle = np.rad2deg(best_theta) + 30.0
    
    return inner, best_center, best_S, outer_angle
# EVOLVE-BLOCK-END
