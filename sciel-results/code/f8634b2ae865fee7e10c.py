"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Strategy: Use a boundary-anchored "peeling" heuristic.
    1. Place hexagons in a dense cluster (hexagonal lattice) to form a compact shape.
    2. Compute the minimal enclosing regular hexagon for this cluster via support functions.
    3. Optimize the placement by trying different rotations of the inner cluster relative to the outer hexagon.
    4. Refine the outer hexagon size and position to tightly enclose the inner set.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin
    
    # Unit regular hexagon vertices (side length 1, centered at origin, flat-top orientation)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    unit_hex_verts = np.array([[np.cos(np.pi/3 * i), np.sin(np.pi/3 * i)] for i in range(6)])
    
    def get_hex_vertices(center, side_length, angle_degrees):
        """Get vertices of a regular hexagon with given center, side length, and rotation."""
        angle = np.radians(angle_degrees)
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        return center + side_length * (unit_hex_verts @ rot.T)
    
    def support_function(points, direction):
        """Compute the maximum projection of points onto a direction."""
        return np.max(points @ direction)
    
    def min_enclosing_hexagon_size(centers, inner_angles, outer_angle_degrees):
        """
        Compute the minimal side length of a regular hexagon (with given rotation)
        that encloses all inner hexagons (each with their own center and rotation).
        
        For a convex container, the minimal enclosing hexagon side length can be found
        by computing the support function in each of the 6 normal directions of the outer hexagon.
        """
        outer_angle = np.radians(outer_angle_degrees)
        # The 6 outward normal directions of a regular hexagon
        # For a flat-top hexagon, normals are at angles 90, 150, 210, 270, 330, 30 degrees
        # Actually, let's use the direction from center to each edge midpoint
        # For a regular hexagon, the 6 edge normals are at angles: 90, 150, 210, 270, 330, 30
        # But it's easier to use the vertex directions rotated by 30 degrees
        
        # The support function of the outer hexagon in direction n is:
        # s_outer(n) = side_length * max_i (n . v_i) where v_i are the unit hex vertices
        # We need: max over all inner hex vertices of (p . n) <= s_outer(n) for all 6 normals n
        
        # Collect all vertices of all inner hexagons
        all_verts = []
        for c, ang in zip(centers, inner_angles):
            verts = get_hex_vertices(c, 1.0, ang)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # The 6 normal directions for the outer hexagon (pointing outward from each edge)
        # For a regular hexagon centered at origin with rotation outer_angle:
        # Edge normals are at angles: outer_angle + 90 + 60*k for k=0..5
        # Actually, for a flat-top hexagon (vertices at 0,60,...), the edge normals are at 90,150,210,270,330,30
        # Let's compute them properly.
        
        # The outer hexagon has vertices at angles: outer_angle + 60*k
        # The edge between vertex k and vertex k+1 has a normal pointing outward
        # The normal direction is at angle: outer_angle + 30 + 60*k
        
        normal_angles = outer_angle + np.pi/6 + 2*np.pi*np.arange(6)/6
        normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
        
        # For each normal direction, find the max projection of all inner vertices
        # The outer hexagon must satisfy: for each normal n, s_outer(n) >= max_p (p . n) - 0
        # where s_outer(n) = side_length * (n . v_max) where v_max is the vertex of unit hex in direction n
        
        # For a unit regular hexagon, the support function in direction n is:
        # s(n) = max_i (n . v_i) where v_i are the 6 vertices
        # This equals the distance from center to the edge in direction n, which is:
        # s(n) = cos(angle_between_n_and_nearest_vertex_normal) * apothem... 
        
        # Simpler: just compute max projection of unit hex vertices in each normal direction
        unit_projections = np.array([np.max(unit_hex_verts @ n) for n in normals])
        
        # For the outer hexagon with side length L centered at origin:
        # The support in direction n is L * unit_projections[k]
        # We need: L * unit_projections[k] >= max_p (p . n) for all k
        # So L >= max_p (p . n) / unit_projections[k] for all k
        # L_min = max_k (max_p (p . n_k) / unit_projections[k])
        
        max_projections = np.array([np.max(all_verts @ n) for n in normals])
        
        # L_min = max_k (max_projections[k] / unit_projections[k])
        ratios = max_projections / unit_projections
        L_min = np.max(ratios)
        
        return L_min
    
    def compute_optimal_center(centers, inner_angles, outer_angle_degrees):
        """
        Find the center of the outer hexagon that minimizes the side length.
        For a given set of inner points, the optimal outer center is the one that
        minimizes the max of (support_in_direction / unit_support_in_direction).
        
        This is equivalent to finding the smallest enclosing hexagon, which can be
        approximated by centering on the centroid and then refining.
        """
        # Start with centroid
        all_centers = np.array(centers)
        center = np.mean(all_centers, axis=0)
        
        # Refine by trying small perturbations
        best_L = min_enclosing_hexagon_size(centers, inner_angles, outer_angle_degrees)
        # But this assumes outer center is at origin. We need to shift.
        
        # Actually, let's reformulate: given inner vertices, find outer center c and side L
        # such that all inner vertices are inside the outer hexagon centered at c with side L.
        
        # The constraint is: for each normal n_k, (p - c) . n_k <= L * unit_proj[k] for all p
        # So: c . n_k >= min_p (p . n_k) ... no wait
        # p . n_k - c . n_k <= L * unit_proj[k]
        # c . n_k >= p . n_k - L * unit_proj[k] for all p
        # c . n_k >= max_p (p . n_k) - L * unit_proj[k]
        
        # Also: p . n_k - c . n_k >= -L * unit_proj[k] (for the opposite direction, which is -n_k)
        # Actually, the hexagon is defined by 6 half-planes. The normal n_k points outward.
        # The constraint is: (p - c) . n_k <= L * unit_proj[k]
        # And: (p - c) . (-n_k) <= L * unit_proj[k] => c . n_k - p . n_k <= L * unit_proj[k]
        
        # So for each k:
        # max_p (p . n_k) - c . n_k <= L * unit_proj[k]
        # c . n_k - min_p (p . n_k) <= L * unit_proj[k]
        
        # Let a_k = max_p (p . n_k), b_k = min_p (p . n_k)
        # Then: c . n_k >= a_k - L * unit_proj[k]
        #       c . n_k <= b_k + L * unit_proj[k]
        
        # So: a_k - L * unit_proj[k] <= c . n_k <= b_k + L * unit_proj[k]
        # For feasibility: a_k - L * unit_proj[k] <= b_k + L * unit_proj[k]
        # => a_k - b_k <= 2 * L * unit_proj[k]
        # => L >= (a_k - b_k) / (2 * unit_proj[k])
        
        # And c must satisfy the linear constraints.
        
        # Let's collect all vertices
        all_verts = []
        for c, ang in zip(centers, inner_angles):
            verts = get_hex_vertices(c, 1.0, ang)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        outer_angle = np.radians(outer_angle_degrees)
        normal_angles = outer_angle + np.pi/6 + 2*np.pi*np.arange(6)/6
        normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
        unit_projections = np.array([np.max(unit_hex_verts @ n) for n in normals])
        
        # Compute a_k and b_k
        projections = all_verts @ normals.T  # shape (num_verts, 6)
        a = np.max(projections, axis=0)  # max projection in each normal direction
        b = np.min(projections, axis=0)  # min projection in each normal direction
        
        # Lower bound on L
        L_lower = np.max((a - b) / (2 * unit_projections))
        
        # Now find a center c that satisfies the constraints
        # c . n_k >= a_k - L * unit_proj[k]
        # c . n_k <= b_k + L * unit_proj[k]
        
        # Try L = L_lower and see if a feasible c exists
        # This is a linear feasibility problem. We can use a simple approach:
        # The center c should be such that the constraints are tight in some sense.
        # A good guess is the center of the bounding box in each direction.
        
        # For each direction k, the "midpoint" constraint suggests:
        # c . n_k = (a_k + b_k) / 2 when L is large enough
        # But with finite L, we have a range.
        
        # Let's just try c = mean of all vertices and check if it works
        c_guess = np.mean(all_verts, axis=0)
        
        # Check if c_guess works with L = L_lower
        def is_feasible(c, L):
            for k in range(6):
                if c @ normals[k] < a[k] - L * unit_projections[k] - 1e-9:
                    return False
                if c @ normals[k] > b[k] + L * unit_projections[k] + 1e-9:
                    return False
            return True
        
        # If not feasible, increase L slightly
        L = L_lower
        if not is_feasible(c_guess, L):
            # Try to find a better center
            # The optimal center minimizes L, which is a convex optimization problem
            # Let's do a simple grid search around the centroid
            best_L = np.inf
            best_c = c_guess
            for dx in np.linspace(-1, 1, 11):
                for dy in np.linspace(-1, 1, 11):
                    c_try = c_guess + np.array([dx, dy])
                    # Check feasibility and compute required L
                    required_L = 0
                    for k in range(6):
                        # c . n_k >= a_k - L * unit_proj[k] => L >= (a_k - c . n_k) / unit_proj[k]
                        # c . n_k <= b_k + L * unit_proj[k] => L >= (c . n_k - b_k) / unit_proj[k]
                        req1 = (a[k] - c_try @ normals[k]) / unit_projections[k]
                        req2 = (c_try @ normals[k] - b[k]) / unit_projections[k]
                        required_L = max(required_L, req1, req2)
                    if required_L < best_L:
                        best_L = required_L
                        best_c = c_try
            
            return best_c, best_L
        
        return c_guess, L_lower
    
    # Generate hexagonal lattice positions for N hexagons
    def hex_lattice_positions(n):
        """Generate positions of n hexagons in a hexagonal lattice pattern."""
        positions = [(0, 0)]
        # Add rings
        ring = 1
        while len(positions) < n:
            # Ring of hexagons at distance ring
            for i in range(6 * ring):
                if len(positions) >= n:
                    break
                # Position on the ring
                # The ring consists of 6 sides of length ring
                side = i // ring
                pos_on_side = i % ring
                # Direction for this side
                angle = np.pi/3 * side
                # Start point of this side
                start_angle = np.pi/3 * side
                # The vertices of the ring are at distance ring * sqrt(3) from center? 
                # Actually, for a hexagonal lattice with spacing 2 (center-to-center for touching hexagons),
                # the ring of radius r has hexagons at specific positions.
                
                # Let's use axial coordinates
                # Center is (0,0). Ring r has positions:
                # (r, -r+k) for k=0..r  (side 0)
                # (r-k, -r) for k=0..r  (side 1)
                # ... etc
                
                # Convert axial to cartesian for flat-top hexagons:
                # x = 2 * q + r (if using q,r axial coords with spacing 2)
                # y = sqrt(3) * r
                # But we want center-to-center distance of 2 for touching unit hexagons.
                
                # Axial coordinates (q, r) to cartesian:
                # x = 2*q + r
                # y = sqrt(3)*r
                # Distance between adjacent hexagons is 2.
                
                # Ring r positions in axial coords:
                # Side 0: (r, -r) to (r, 0)  -> q=r, r_coord from -r to 0
                # Side 1: (r, 0) to (0, -r)  -> q from r to 0, r_coord = -q
                # Side 2: (0, -r) to (-r, 0) -> q from 0 to -r, r_coord = q - r... 
                
                # Let me just generate all axial coords in the ring
                # For ring r, the 6 corners are at:
                # (r, 0), (0, r), (-r, r), (-r, 0), (0, -r), (r, -r)
                # Wait, that's not right for a hexagonal lattice.
                
                # Standard hex ring:
                # Corner 0: (r, -r)
                # Corner 1: (r, 0)
                # Corner 2: (0, r)
                # Corner 3: (-r, r)
                # Corner 4: (-r, 0)
                # Corner 5: (0, -r)
                
                corners = [
                    (r, -r),
                    (r, 0),
                    (0, r),
                    (-r, r),
                    (-r, 0),
                    (0, -r)
                ]
                
                for s in range(6):
                    c1 = corners[s]
                    c2 = corners[(s+1) % 6]
                    dq = c2[0] - c1[0]
                    dr = c2[1] - c1[1]
                    for k in range(ring):
                        q = c1[0] + k * dq // ring
                        r_coord = c1[1] + k * dr // ring
                        # Convert to cartesian
                        x = 2*q + r_coord
                        y = np.sqrt(3) * r_coord
                        positions.append((x, y))
        
        return positions[:n]
    
    # Generate positions
    positions = hex_lattice_positions(N)
    centers = [np.array(p, dtype=float) for p in positions]
    inner_angles = [0.0] * N
    
    # Try different outer hexagon rotations to find the best fit
    best_result = None
    best_side = np.inf
    
    # Try various outer rotations
    for outer_rot in np.linspace(0, 60, 13):  # 0 to 60 degrees in 5 degree steps
        # Also try rotating the inner cluster
        for inner_rot in [0.0, 30.0]:
            rotated_centers = []
            rot_angle = np.radians(inner_rot)
            rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                                [np.sin(rot_angle), np.cos(rot_angle)]])
            for c in centers:
                rotated_centers.append(c @ rot_mat.T)
            
            rotated_angles = [(a + inner_rot) % 360 for a in inner_angles]
            
            # Compute optimal center and side length
            center, side = compute_optimal_center(rotated_centers, rotated_angles, outer_rot)
            
            if side < best_side:
                best_side = side
                best_result = (rotated_centers, rotated_angles, center, outer_rot)
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    if best_result is None:
        # Fallback: simple placement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    rotated_centers, rotated_angles, center, outer_rot = best_result
    
    # Format inner hexagons: (x, y, angle_degrees)
    inner = [[c[0], c[1], a] for c, a in zip(rotated_centers, rotated_angles)]
    
    return inner, (center[0], center[1]), best_side, outer_rot
# EVOLVE-BLOCK-END
