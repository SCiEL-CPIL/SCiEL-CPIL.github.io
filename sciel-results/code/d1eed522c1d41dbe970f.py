import numpy as np
import time

N = 11

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Generate candidate positions for 11 hexagons
    # We'll use a lattice based on the honeycomb structure
    # Lattice vectors for flat-top hexagons
    # Distance between centers is sqrt(3)
    # Let's define a set of grid points (i, j) in the lattice basis
    # i, j can be integers or half-integers depending on the row
    
    # A better approach: define centers explicitly for a compact shape.
    # Shape: A hexagon of side 2 (7 hexes) + 4 more.
    # Let's try to extend the side 2 hexagon to side 3 in 4 directions.
    # The side 3 hexagon has rows of lengths 3, 4, 5, 4, 3 (total 19).
    # We want 11.
    # Maybe rows 3, 3, 3, 2?
    
    # Let's generate centers for a "3-3-3-2" arrangement.
    # Flat top hexagons.
    # Width = 1. Height = sqrt(3).
    # Horizontal spacing between centers in same row: 1.5? No.
    # If flat top, the horizontal distance between centers of touching hexes is 1.5?
    # Let's check.
    # Vertex at (0.5, sqrt(3)/2). Next hex center at (1.5, sqrt(3)/2)?
    # If I place hex at (0,0), its right vertex is at (1,0).
    # Neighbor to the right (sharing vertical edge) has left vertex at (1,0).
    # So center is at (1+1, 0) = (2,0)? No.
    # Flat top hex: vertices (1,0), (0.5, h), (-0.5, h), (-1,0), (-0.5, -h), (0.5, -h). h=sqrt(3)/2.
    # Right edge is vertical? No, the edge from (1,0) to (0.5, h) is slanted.
    # The vertical edges are NOT present in flat-top.
    # Pointy-top has vertical edges.
    
    # Let's use Pointy-Top Hexagons for easier lattice math?
    # Pointy-top: vertices (0, 1), (-h, 0.5), (-h, -0.5), (0, -1), (h, -0.5), (h, 0.5). h=sqrt(3)/2.
    # Vertical edges? No, the edges are slanted.
    # Wait, pointy-top has a vertex at top (0,1) and bottom (0,-1).
    # The "sides" are slanted.
    # The neighbors are at distance sqrt(3).
    # Lattice vectors:
    # a1 = (sqrt(3), 0.5)
    # a2 = (0, 1) ? No.
    # Standard pointy-top lattice:
    # Row 0: y=0. Centers at x = 0, sqrt(3), 2sqrt(3)...
    # Row 1: y=1.5. Centers at x = sqrt(3)/2, 3sqrt(3)/2...
    
    # Let's stick to the geometric optimization.
    # We will define a set of 11 centers.
    # Initial guess: The 7-cluster plus 4 more.
    
    # Centers for 7-cluster (pointy-top, side 1):
    # Center (0,0)
    # Neighbors at distance sqrt(3) = 1.732.
    # Angles: 0, 60, 120, 180, 240, 300.
    # This forms a hexagon.
    
    centers = []
    # 7 cluster
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        centers.append((x, y))
        
    # Add 4 more.
    # We can add them to the "corners" of the 7-cluster to make it more circular/hexagonal.
    # The 7-cluster has "bays" between the neighbors.
    # The distance between neighbor k and k+1 is sqrt(3).
    # We can place new hexes in the gaps.
    # The gap is a rhombus.
    # Let's add them at the midpoints of the arcs?
    # Or extend the 7-cluster to a larger hexagon.
    
    # Let's try a different configuration:
    # A 3x3 block of hexagons? No, that's 9.
    # Let's use a lattice grid.
    # Pointy-top lattice coordinates (col, row).
    # x = col * sqrt(3) + (row % 2) * sqrt(3)/2
    # y = row * 1.5
    
    # Let's pick 11 points from this lattice that form a compact shape.
    # Rows:
    # Row 0: 3 hexes (cols -1, 0, 1)
    # Row 1: 3 hexes (cols -1, 0, 1)
    # Row 2: 3 hexes (cols -1, 0, 1)
    # Row 3: 2 hexes (cols -1, 0)
    # Total 11.
    
    new_centers = []
    for r in range(4):
        if r < 3:
            for c in range(-1, 2):
                x = c * np.sqrt(3) + (r % 2) * np.sqrt(3)/2
                y = r * 1.5
                new_centers.append((x, y))
        else:
            for c in range(-1, 1):
                x = c * np.sqrt(3) + (r % 2) * np.sqrt(3)/2
                y = r * 1.5
                new_centers.append((x, y))
                
    # Use these centers
    centers = new_centers
    
    # Get vertices of all inner hexagons
    # Pointy-top unit hexagon vertices relative to center:
    # (0, 1), (-h, 0.5), (-h, -0.5), (0, -1), (h, -0.5), (h, 0.5)
    h = np.sqrt(3) / 2
    offsets = np.array([
        [0, 1],
        [-h, 0.5],
        [-h, -0.5],
        [0, -1],
        [h, -0.5],
        [h, 0.5]
    ])
    
    all_vertices = []
    for cx, cy in centers:
        verts = offsets + np.array([cx, cy])
        all_vertices.append(verts)
        
    all_vertices = np.vstack(all_vertices)
    
    # Find minimal enclosing hexagon
    # We search for angle theta and center (cx, cy)
    # For a fixed theta, the optimal center is the midpoint of the min/max projections?
    # Actually, for a regular hexagon, the side length L is determined by the max of the projections onto the 6 normals.
    # Let normals be n_i(theta).
    # proj_i = dot(vert, n_i)
    # L = max_i (max(proj_i) - min(proj_i)) / 2 * 2?
    # The width of the hexagon in direction n is 2L cos(0) ? No.
    # For a regular hexagon of side L, the distance between parallel sides is L * sqrt(3).
    # So if the range of projections is W, then L = W / sqrt(3).
    
    best_L = float('inf')
    best_center = (0,0)
    best_angle = 0
    
    # Search angles
    # Symmetry: 0 to 60 degrees.
    angles = np.linspace(0, np.pi/3, 100)
    
    for theta in angles:
        if time.time() > deadline - 5:
            break
            
        # Normals of the hexagon
        # Vertices of unit hexagon (side 1) are at angles 0, 60, ...
        # Normals are at 30, 90, ...
        # If we rotate the hexagon by theta, normals are at theta + 30 + k*60.
        # Let's define normals for a hexagon with flat top?
        # Pointy-top hexagon has normals at 0, 60, ...?
        # Vertex at 90 deg. Normal to side is 60 deg?
        # Side is between 30 and 90? No.
        # Let's just use the property that the width in direction alpha is W(alpha).
        # For a regular hexagon of side L, W(alpha) = L * (sqrt(3) * |cos(alpha - alpha_0)| + ...)?
        # Simpler: The projection of a hexagon of side L onto a line at angle phi is L * (sqrt(3) if phi is normal to side, 2L if phi is normal to vertex).
        # Actually, the width function of a regular hexagon is periodic with period 60 deg.
        # W(phi) = L * sqrt(3) / cos((phi - phi_0) % 30 - 15)? No.
        # Let's just compute the bounding box in the rotated frame.
        
        # Rotated normals
        base_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        normals = np.array([[np.cos(a + theta), np.sin(a + theta)] for a in base_angles])
        
        # Project all vertices
        # V is (N, 2)
        # N is (6, 2)
        # Projections = V @ N.T
        projs = all_vertices @ normals.T
        
        # For each normal direction, find min and max projection
        mins = np.min(projs, axis=0)
        maxs = np.max(projs, axis=0)
        
        # The width in each direction is maxs - mins
        widths = maxs - mins
        
        # The side length L of the enclosing hexagon must satisfy:
        # For a regular hexagon of side L, the width in direction theta_k is:
        # If theta_k is normal to a side, width = L * sqrt(3).
        # If theta_k is normal to a vertex, width = 2 * L.
        # Our normals are separated by 60 degrees.
        # Let's assume our normals are aligned with the "side normals" of the hexagon.
        # Then width = L * sqrt(3).
        # So L = width / sqrt(3).
        # However, the hexagon orientation is theta.
        # We need to ensure the hexagon covers the points.
        # The condition is that for all 6 directions, the point must be inside.
        # If we define the hexagon by its 6 supporting lines at distance d from center,
        # then d = L * sqrt(3) / 2.
        # So L = 2 * d / sqrt(3).
        # The required d is (max_proj - min_proj) / 2.
        # So L = (max_proj - min_proj) / sqrt(3).
        
        Ls = widths / np.sqrt(3)
        L_max = np.max(Ls)
        
        if L_max < best_L:
            best_L = L_max
            # The center of the enclosing hexagon is the midpoint of the projections
            # proj_center = (mins + maxs) / 2
            # This gives the projection of the center onto the normals.
            # We can recover the center by averaging the inverse projections?
            # Or simply, the center is the average of the midpoints of the bounding box in the rotated frame?
            # Let's just compute the center as the mean of the vertices? No, that's not optimal.
            # The optimal center is the one that minimizes the max deviation.
            # For a fixed orientation, the optimal center is the center of the minimal enclosing circle? No.
            # It is the center that equalizes the margins?
            # Actually, for a convex polygon, the minimal enclosing regular hexagon center is the intersection of the bisectors of the tightest constraints.
            # Approximation: The center is the average of the mins and maxs mapped back?
            # Let's just use the centroid of the vertices as a good initial guess, or solve a small LP?
            # Given time constraints, let's assume the center is the mean of the vertices.
            # This might not be optimal, but it's a start.
            cx = np.mean(all_vertices[:, 0])
            cy = np.mean(all_vertices[:, 1])
            best_center = (cx, cy)
            best_angle = np.rad2deg(theta)
            
    # Return the result
    # inner: list of [x, y, angle]
    # We used pointy-top hexagons. Angle 0 in our definition corresponds to pointy-top.
    # The task says "angle_degrees". Usually 0 is flat-top or pointy-top?
    # Standard: 0 is usually flat-top in some libraries, pointy-top in others.
    # Let's assume the evaluator handles the rotation.
    # We will return angle 0.0 for all inner hexagons since we placed them in a lattice.
    
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    return inner, best_center, best_L, best_angle

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Lattice parameters for pointy-top hexagons
    h = np.sqrt(3) / 2
    w = np.sqrt(3)
    
    # Define 11 centers in a compact shape
    # Rows of 3, 3, 3, 2
    centers = []
    for r in range(4):
        if r < 3:
            for c in range(-1, 2):
                x = c * w + (r % 2) * h
                y = r * 1.5
                centers.append((x, y))
        else:
            for c in range(-1, 1):
                x = c * w + (r % 2) * h
                y = r * 1.5
                centers.append((x, y))
                
    # Vertices of unit pointy-top hexagon
    offsets = np.array([
        [0, 1],
        [-h, 0.5],
        [-h, -0.5],
        [0, -1],
        [h, -0.5],
        [h, 0.5]
    ])
    
    all_vertices = []
    for cx, cy in centers:
        verts = offsets + np.array([cx, cy])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    best_L = float('inf')
    best_center = (0,0)
    best_angle = 0
    
    # Search over rotation angles
    # The optimal angle is likely 0 or 30 degrees relative to the lattice
    angles = np.linspace(0, np.pi/6, 50) # 0 to 30 degrees
    
    for theta in angles:
        if time.time() > deadline:
            break
            
        # Normals of the enclosing hexagon
        # We align the normals with the hexagon sides
        # Base normals at 0, 60, ...
        base_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        normals = np.array([[np.cos(a + theta), np.sin(a + theta)] for a in base_angles])
        
        projs = all_vertices @ normals.T
        mins = np.min(projs, axis=0)
        maxs = np.max(projs, axis=0)
        widths = maxs - mins
        Ls = widths / np.sqrt(3)
        L_max = np.max(Ls)
        
        if L_max < best_L:
            best_L = L_max
            # Estimate center
            # The center projection on normal i is (mins[i] + maxs[i]) / 2
            # We can solve for center (cx, cy) using least squares
            # normals @ [cx, cy].T = center_projs
            center_projs = (mins + maxs) / 2
            # Solve normals.T @ [cx, cy] = center_projs ? No.
            # normals is (6,2). We want [cx, cy] such that normals @ [cx, cy].T = center_projs.
            # This is an overdetermined system. Use lstsq.
            sol, residuals, rank, s = np.linalg.lstsq(normals, center_projs, rcond=None)
            cx, cy = sol
            best_center = (cx, cy)
            best_angle = np.rad2deg(theta)
            
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    return inner, best_center, best_L, best_angle
# EVOLVE-BLOCK-END
