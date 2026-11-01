import numpy as np
import time
from collections import deque
import itertools

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Axial coordinates for hexagonal grid
    # Directions: (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    def axial_to_xy(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    def get_neighbors(q, r):
        return [(q + dq, r + dr) for dq, dr in dirs]

    def generate_clusters(n_cells):
        """
        Generate all connected sets of n_cells on a hexagonal grid.
        To avoid duplicates, we enforce a canonical form:
        The cluster must contain the origin (0,0).
        We grow the cluster by adding cells adjacent to the existing set.
        We use a set of tuples of axial coords to represent the cluster.
        To avoid duplicates, we can use a canonical representation:
        Translate the cluster so that the minimum q is 0, and if min q is 0, min r is 0, etc.
        """
        # This is a standard polyomino-like enumeration but for hexagons.
        # For n=12, the number of clusters is manageable (few thousands).
        clusters = set()
        
        # Start with origin
        current = frozenset([(0, 0)])
        clusters.add(current)
        
        # We need to generate all connected sets of size n_cells.
        # A simple BFS on the set of clusters might be slow if not pruned.
        # Alternative: Recursive backtracking with pruning.
        
        # Given the small size (12), we can use a simpler approach:
        # Generate all subsets of a larger grid? No, too big.
        # Let's use the "growth" method with canonicalization.
        
        # To ensure we don't miss any, we grow from a seed.
        # However, pure growth from (0,0) might miss shapes that don't contain (0,0) in their final canonical form?
        # No, any connected shape can be translated to contain (0,0).
        # So we generate all connected shapes of size k that contain (0,0).
        
        # To avoid duplicates, we define a canonical form:
        # 1. Translate so min(q) = 0.
        # 2. If min(q) > 0, shift left.
        # 3. If min(q) == 0, shift down (min(r) = 0).
        # This might still have duplicates if the shape is symmetric.
        # We will store the canonical frozenset.
        
        def canonicalize(cells):
            # cells is a set of (q, r)
            min_q = min(q for q, r in cells)
            min_r = min(r for q, r in cells if q == min_q)
            # Shift so that min_q is 0, and among those, min_r is 0
            return frozenset((q - min_q, r - min_r) for q, r in cells)

        # BFS growth
        # State: frozenset of cells
        # We want size n_cells.
        
        # Optimization: Only add cells that are "adjacent" and maintain connectivity.
        # Also, to reduce branching, we can restrict the "frontier".
        
        # Let's try a recursive approach with pruning.
        # To keep it fast, we limit the search space.
        
        # Actually, for n=12, the number of hexagonal polyhexes is 2500+.
        # Generating them all is fast.
        
        # We will use a queue of (cells_set, frontier_set)
        # To avoid duplicates, we only add cells from the frontier that are "new" in a canonical sense?
        # Standard algorithm:
        # 1. Start with {(0,0)}.
        # 2. At each step, add a cell from the perimeter.
        # 3. To avoid duplicates, we enforce that the new cell is "above" or "right" of previous additions in some order?
        # Simpler: Just generate all, then deduplicate via canonical form.
        # With n=12, the number of raw generated trees is small enough if we prune disconnected ones.
        
        # Let's implement a simple backtracking generator.
        
        def backtrack(current_cells, size):
            if size == n_cells:
                clusters.add(canonicalize(current_cells))
                return
            
            # Get perimeter cells
            perimeter = set()
            for q, r in current_cells:
                for nq, nr in get_neighbors(q, r):
                    if (nq, nr) not in current_cells:
                        perimeter.add((nq, nr))
                        
            # To reduce duplicates, we can sort perimeter cells and only add those that are "valid"
            # A common trick: only add cells that are "greater than" the last added cell in some ordering?
            # This is tricky with 2D.
            # Let's just generate and deduplicate.
            # To prevent explosion, we can limit the bounding box size.
            # Max dimension for 12 hexagons is 12.
            
            for nq, nr in perimeter:
                # Pruning: ensure we don't go too far
                if abs(nq) > 15 or abs(nr) > 15:
                    continue
                # Add cell
                new_cells = set(current_cells)
                new_cells.add((nq, nr))
                backtrack(frozenset(new_cells), size + 1)
                
        backtrack(frozenset([(0, 0)]), 1)
        return list(clusters)

    def calculate_min_side(cluster_cells):
        """
        Calculate the minimum side length of a regular hexagon containing the cluster.
        Returns the side length.
        """
        # Get all vertices of all hexagons in the cluster
        all_vertices = []
        for q, r in cluster_cells:
            cx, cy = axial_to_xy(q, r)
            # Vertices of unit hexagon at (cx, cy) with angle 0
            # Angles: 0, 60, 120, 180, 240, 300
            for k in range(6):
                theta = np.radians(60 * k)
                vx = cx + np.cos(theta)
                vy = cy + np.sin(theta)
                all_vertices.append((vx, vy))
        
        all_vertices = np.array(all_vertices)
        
        # Normals to the 6 sides of a regular hexagon (angle 0)
        # Side 0 is between vertex 0 and 1.
        # Vertex 0: (1, 0), Vertex 1: (0.5, sqrt(3)/2)
        # Midpoint: (0.75, sqrt(3)/4)
        # Normal direction: (sqrt(3)/2, 1/2) ?
        # Let's use the standard normals for a hexagon with flat top/bottom or pointy top?
        # The seed uses:
        # L1 = x + y/sqrt3  -> Normal (1, 1/sqrt3) -> Angle 30 deg?
        # Let's check the seed's normals.
        # The seed's `required_outer_side` effectively calculates the max projection onto 6 directions.
        # The directions are the normals to the sides.
        # For a regular hexagon, the normals are at 30, 90, 150, 210, 270, 330 degrees?
        # Or 0, 60, 120...?
        # Let's derive from the seed:
        # L1 = x + y/sqrt3. Normal vector n1 = (1, 1/sqrt3). Normalized: (sqrt(3)/2, 1/2). Angle 30 deg.
        # L2 = 2y/sqrt3. Normal vector n2 = (0, 2/sqrt3). Normalized: (0, 1). Angle 90 deg.
        # L3 = -x + y/sqrt3. Normal vector n3 = (-1, 1/sqrt3). Angle 150 deg.
        # L4 = -x - y/sqrt3. Normal vector n4 = (-1, -1/sqrt3). Angle 210 deg.
        # L5 = -2y/sqrt3. Normal vector n5 = (0, -1). Angle 270 deg.
        # L6 = x - y/sqrt3. Normal vector n6 = (1, -1/sqrt3). Angle 330 deg.
        
        # So the normals are at 30, 90, 150, 210, 270, 330.
        # These are the normals to the sides of a "pointy-top" hexagon?
        # A pointy-top hexagon has vertices at 90, 30, -30, -90, -150, 150?
        # If vertices are at 90, 30, -30... then sides are at 60, 0, -60...
        # Wait, if the normal is at 90, the side is horizontal (top/bottom).
        # So the hexagon has horizontal top and bottom sides.
        # This corresponds to a "flat-top" hexagon in standard orientation?
        # Standard flat-top: Vertices at 0, 60, 120, 180, 240, 300.
        # Side between 0 and 60: Midpoint 30. Normal 30? No, normal is 30+90=120?
        # Let's stick to the seed's normals: 30, 90, 150, 210, 270, 330.
        
        normals = []
        for k in range(6):
            theta = np.radians(30 + 60 * k)
            normals.append((np.cos(theta), np.sin(theta)))
        normals = np.array(normals) # (6, 2)
        
        # Project all vertices onto each normal
        # Projections shape: (6, M) where M is number of vertices
        projections = all_vertices @ normals.T
        
        # For each direction i, we have a range [min_i, max_i].
        # The width in direction i is max_i - min_i.
        # The side length S of the outer hexagon must satisfy:
        # The distance between opposite sides is S * sqrt(3).
        # So S * sqrt(3) >= width_i for all i.
        # S >= width_i / sqrt(3).
        # S_min = max_i (width_i / sqrt(3)).
        
        mins = np.min(projections, axis=1)
        maxs = np.max(projections, axis=1)
        widths = maxs - mins
        
        max_width = np.max(widths)
        side_length = max_width / np.sqrt(3.0)
        
        return side_length

    # Generate all clusters
    clusters = generate_clusters(N)
    
    best_side = float('inf')
    best_cluster = None
    
    for cluster in clusters:
        if time.time() > deadline:
            break
            
        side = calculate_min_side(cluster)
        if side < best_side:
            best_side = side
            best_cluster = cluster

    # If no cluster found (shouldn't happen), fallback
    if best_cluster is None:
        # Fallback to a simple linear arrangement
        best_cluster = frozenset([(i, 0) for i in range(N)])
        best_side = calculate_min_side(best_cluster)

    # Construct the output
    # We need to return the inner hexagons' positions.
    # The cluster is defined in axial coordinates.
    # We need to convert to Cartesian (x, y).
    # We also need to center the configuration so that the outer hexagon (centered at 0,0) contains it optimally.
    
    # The optimal center for the outer hexagon is the midpoint of the projection ranges.
    # Let's compute the optimal shift.
    
    all_vertices = []
    for q, r in best_cluster:
        cx, cy = axial_to_xy(q, r)
        for k in range(6):
            theta = np.radians(60 * k)
            vx = cx + np.cos(theta)
            vy = cy + np.sin(theta)
            all_vertices.append((vx, vy))
    all_vertices = np.array(all_vertices)
    
    normals = []
    for k in range(6):
        theta = np.radians(30 + 60 * k)
        normals.append((np.cos(theta), np.sin(theta)))
    normals = np.array(normals)
    
    projections = all_vertices @ normals.T
    mins = np.min(projections, axis=1)
    maxs = np.max(projections, axis=1)
    
    # The optimal center C in Cartesian coordinates satisfies:
    # C . n_i = (min_i + max_i) / 2 for all i?
    # This is only possible if the system is consistent.
    # For a regular hexagon, the "center" of the bounding box in the 6 directions
    # corresponds to the centroid of the "corners" of the bounding box.
    # We can solve for C using least squares or just pick the average of the midpoints projected back?
    # Actually, the shift vector S_shift is such that the new projections are centered at 0.
    # new_proj_i = proj_i - C . n_i.
    # We want max(new_proj_i) - min(new_proj_i) to be minimized? No, the width is invariant to shift.
    # We want the outer hexagon centered at 0 to contain the shifted points.
    # Condition: | (p - C) . n_i | <= S * sqrt(3) / 2.
    # This means -S*sqrt(3)/2 <= (p-C).n_i <= S*sqrt(3)/2.
    # Let width_i = max(p.n_i) - min(p.n_i).
    # We know S = max(width_i)/sqrt(3).
    # We need to choose C such that for the critical direction i*, the range is centered.
    # C . n_i* = (max(p.n_i*) + min(p.n_i*)) / 2.
    # We can solve for C using 2 of the normals (e.g., n_0 and n_1) and the midpoints.
    
    # Let mid_i = (max_i + min_i) / 2.
    # We want C . n_i = mid_i.
    # This is an overdetermined system. We can solve it in least squares sense.
    # Or, since the geometry is regular, the optimal C is the centroid of the "bounding hexagon" vertices.
    # Simpler: Solve for C using n_0 and n_1.
    
    # n_0 = (cos(30), sin(30)), n_1 = (cos(90), sin(90)) = (0, 1)
    # C . n_0 = mid_0
    # C . n_1 = mid_1
    
    # Let C = (cx, cy).
    # cx * cos(30) + cy * sin(30) = mid_0
    # cy = mid_1
    
    # cx * cos(30) = mid_0 - mid_1 * sin(30)
    # cx = (mid_0 - mid_1 * 0.5) / cos(30)
    
    mid_0 = (mins[0] + maxs[0]) / 2.0
    mid_1 = (mins[1] + maxs[1]) / 2.0
    
    cos_30 = np.cos(np.radians(30))
    sin_30 = np.sin(np.radians(30))
    
    cx_opt = (mid_0 - mid_1 * sin_30) / cos_30
    cy_opt = mid_1
    
    # Shift all inner hexagons by -C_opt so they fit in the outer hexagon centered at 0.
    inner = []
    for q, r in best_cluster:
        cx, cy = axial_to_xy(q, r)
        x_final = cx - cx_opt
        y_final = cy - cy_opt
        inner.append([float(x_final), float(y_final), 0.0])
        
    # Final validation check using the seed's logic (optional but good for safety)
    # We trust the calculation, but let's verify the side length one last time.
    # The side length is best_side.
    
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
