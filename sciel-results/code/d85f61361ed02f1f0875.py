import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def hexagon_packing_11():
    start_time = time.time()
    deadline = 625.0

    # Unit hexagon vertices at angle 0 (pointy-top: vertices at 0, 60, 120, 180, 240, 300)
    # Actually per spec: vertices at angle a + 60*i for i=0..5
    # So at angle 0: vertices at 0, 60, 120, 180, 240, 300 degrees
    hex_verts = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])

    # Edge normals: for a hexagon with vertices at angles 0,60,...,300,
    # the edges are between consecutive vertices. The normal to the edge
    # between vertices at angle a and a+60 points in direction a+30.
    # So edge normals are at 30, 90, 150, 210, 270, 330
    edge_normals = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(30, 360, 60)
    ])

    # Project hexagon vertices onto edge normals
    proj_hex = hex_verts @ edge_normals.T
    min_hex = np.min(proj_hex, axis=0)
    max_hex = np.max(proj_hex, axis=0)

    def get_hex_vertices(center, angle_deg):
        """Get vertices of a hexagon with given center and angle."""
        angle_rad = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
        return center + hex_verts @ rot.T

    def pair_gap(c1, c2, a1, a2):
        """Compute the gap between two hexagons. Positive means separated.
        Gap = minimum over all 12 edge normals of the separation.
        For the pair to be valid, gap must be > 1e-6 (no intersection).
        
        Actually, we need: for at least one of the 12 edge normals, the
        projected intervals are apart by more than 1e-6.
        Wait, re-reading: "on at least one of their 12 edge normals the projected
        intervals are apart by more than 1e-6 (touching or a smaller gap is an
        intersection)"
        
        This means: the hexagons are separated if there EXISTS an edge normal
        (from either hexagon, 6+6=12 total) such that the projections don't
        overlap (gap > 1e-6). If for ALL 12 normals the projections overlap
        (gap <= 1e-6), then they intersect.
        
        Actually I think the condition is: they are valid (non-intersecting)
        if on at least one of the 12 edge normals, the intervals are apart.
        This is the separating axis theorem.
        
        Let me reconsider. The 12 edge normals are the 6 normals of hexagon 1
        and the 6 normals of hexagon 2. For each normal, we project both
        hexagons and check if the intervals overlap. If for ALL 12 normals
        the intervals overlap, the hexagons intersect. If for ANY normal they
        are apart, they don't intersect.
        
        So gap = max over all 12 normals of (separation on that normal).
        If gap > 1e-6, they're separated.
        """
        # Normals for hexagon 1 (rotated by a1)
        a1_rad = np.deg2rad(a1)
        rot1 = np.array([[np.cos(a1_rad), -np.sin(a1_rad)],
                         [np.sin(a1_rad), np.cos(a1_rad)]])
        normals1 = edge_normals @ rot1.T
        
        # Normals for hexagon 2 (rotated by a2)
        a2_rad = np.deg2rad(a2)
        rot2 = np.array([[np.cos(a2_rad), -np.sin(a2_rad)],
                         [np.sin(a2_rad), np.cos(a2_rad)]])
        normals2 = edge_normals @ rot2.T
        
        all_normals = np.vstack([normals1, normals2])
        
        # Project hex1 vertices onto all normals
        verts1 = get_hex_vertices(c1, a1)
        proj1 = verts1 @ all_normals.T
        min1 = np.min(proj1, axis=0)
        max1 = np.max(proj1, axis=0)
        
        # Project hex2 vertices onto all normals
        verts2 = get_hex_vertices(c2, a2)
        proj2 = verts2 @ all_normals.T
        min2 = np.min(proj2, axis=0)
        max2 = np.max(proj2, axis=0)
        
        # Separation on each normal: max(0, min2 - max1, min1 - max2)
        # Actually separation = max(min2 - max1, min1 - max2, 0)
        # But we want the gap: if intervals don't overlap, gap = distance between them
        # gap_i = max(min2_i - max1_i, min1_i - max2_i)
        # If gap_i > 0, they're separated on that axis
        gap = np.maximum(min2 - max1, min1 - max2)
        
        return np.max(gap)

    def compute_side(centers, angles, theta):
        """Compute the side length of the outer hexagon that contains all inner hexagons."""
        # Outer hexagon has angle theta, centered at origin
        # Vertices at theta + 60*i
        # Edge normals at theta + 30 + 60*i
        outer_normals = np.array([
            [np.cos(theta + np.deg2rad(30 + 60 * i)),
             np.sin(theta + np.deg2rad(30 + 60 * i))] for i in range(6)
        ])
        
        # For each inner hexagon, find max projection onto each outer normal
        # The outer hexagon with side s has apothem = s * sqrt(3)/2
        # The distance from center to each edge is the apothem
        # For a hexagon with side s, the inradius (apothem) = s * sqrt(3)/2
        # So s = 2 * apothem / sqrt(3)
        
        max_projections = np.zeros(6)
        for k in range(11):
            verts = get_hex_vertices(centers[k], angles[k])
            proj = verts @ outer_normals.T
            max_projections = np.maximum(max_projections, np.max(proj, axis=0))
        
        # The outer hexagon must contain all inner hexagons
        # The apothem of the outer hexagon must be >= max projection
        # apothem = s * sqrt(3) / 2
        # s = 2 * apothem / sqrt(3)
        apothem = np.max(max_projections)
        side = 2.0 * apothem / np.sqrt(3.0)
        return side

    def is_valid(centers, angles, theta):
        """Check if the configuration is valid."""
        # Check all pairs are separated
        for i in range(11):
            for j in range(i + 1, 11):
                gap = pair_gap(centers[i], centers[j], angles[i], angles[j])
                if gap <= 1e-6:
                    return False
        
        # Check all inner hexagon vertices are inside the outer hexagon
        outer_verts = np.array([
            [side_val * np.cos(theta + np.deg2rad(60 * i)),
             side_val * np.sin(theta + np.deg2rad(60 * i))] 
            for i in range(6)
        ]) if False else None
        
        # Actually, let's check using the edge normals
        outer_normals = np.array([
            [np.cos(theta + np.deg2rad(30 + 60 * i)),
             np.sin(theta + np.deg2rad(30 + 60 * i))] for i in range(6)
        ])
        
        # Get outer side
        max_projections = np.zeros(6)
        for k in range(11):
            verts = get_hex_vertices(centers[k], angles[k])
            proj = verts @ outer_normals.T
            max_projections = np.maximum(max_projections, np.max(proj, axis=0))
        
        apothem = np.max(max_projections)
        side = 2.0 * apothem / np.sqrt(3.0)
        
        # Check each inner vertex is inside the outer hexagon
        # For each outer edge (from vertex i to vertex i+1), the inward normal
        # points toward the center. A point p is inside if it's on the correct
        # side of each edge.
        
        outer_verts = np.array([
            [side * np.cos(theta + np.deg2rad(60 * i)),
             side * np.sin(theta + np.deg2rad(60 * i))] 
            for i in range(6)
        ])
        
        for k in range(11):
            verts = get_hex_vertices(centers[k], angles[k])
            for p in verts:
                # Check against each outer edge
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i + 1) % 6]
                    # Edge direction
                    edge_dir = p2 - p1
                    # Inward normal (rotated 90 degrees clockwise from edge direction,
                    # which should point toward center)
                    # Actually, for a convex polygon, the inward normal of edge p1->p2
                    # is the 90-degree clockwise rotation of the edge direction
                    # if the polygon is CCW. Let's check: for a regular hexagon centered
                    # at origin with vertices CCW, the inward normal of each edge
                    # points toward the origin.
                    # Normal = (edge_dir_y, -edge_dir_x) for CW rotation
                    # Or (-edge_dir_y, edge_dir_x) for CCW rotation
                    # Let's use: normal = (p2.y - p1.y, -(p2.x - p1.x))
                    # This is the 90-degree clockwise rotation of edge_dir
                    nx = edge_dir[1]
                    ny = -edge_dir[0]
                    # Normalize
                    norm = np.sqrt(nx**2 + ny**2)
                    if norm > 1e-10:
                        nx /= norm
                        ny /= norm
                    # Check if p is on the inward side
                    # The inward side is where the center (0,0) is
                    # cross(edge_dir, p - p1) should have the same sign as
                    # cross(edge_dir, center - p1) = cross(edge_dir, -p1)
                    cross_p = edge_dir[0] * (p[1] - p1[1]) - edge_dir[1] * (p[0] - p1[0])
                    cross_c = edge_dir[0] * (0 - p1[1]) - edge_dir[1] * (0 - p1[0])
                    if cross_p * cross_c < -1e-6:
                        return False
        return True

    def objective(params):
        """Objective: minimize the side length of the outer hexagon."""
        if time.time() > deadline - 5:
            return 1e10
        
        # params: 11*2 + 11 + 1 = 34
        # centers (22), angles (11), theta (1)
        centers = params[:22].reshape(11, 2)
        angles = params[22:33]
        theta = params[33]
        
        # Check separation
        for i in range(11):
            for j in range(i + 1, 11):
                gap = pair_gap(centers[i], centers[j], angles[i], angles[j])
                if gap <= 1e-6:
                    return 1e8 + (1e-6 - gap)
        
        side = compute_side(centers, angles, theta)
        
        # Check containment
        outer_normals = np.array([
            [np.cos(theta + np.deg2rad(30 + 60 * i)),
             np.sin(theta + np.deg2rad(30 + 60 * i))] for i in range(6)
        ])
        max_projections = np.zeros(6)
        for k in range(11):
            verts = get_hex_vertices(centers[k], angles[k])
            proj = verts @ outer_normals.T
            max_projections = np.maximum(max_projections, np.max(proj, axis=0))
        
        apothem = np.max(max_projections)
        side = 2.0 * apothem / np.sqrt(3.0)
        
        return side

    # Initial configuration: centered hexagon with 6 around it
    d = np.sqrt(3)  # distance between centers of adjacent hexagons
    base_centers = np.zeros((11, 2))
    base_angles = np.zeros(11)
    for i in range(6):
        angle = np.deg2rad(60 * i)
        base_centers[i + 1, 0] = d * np.cos(angle)
        base_centers[i + 1, 1] = d * np.sin(angle)
    
    base_theta = 0.0
    init_params = np.concatenate([base_centers.flatten(), base_angles, [base_theta]])
    
    # Check initial validity
    if is_valid(base_centers, base_angles, base_theta):
        side0 = compute_side(base_centers, base_angles, base_theta)
    else:
        side0 = 1e10

    best_side = side0
    best_params = init_params.copy()
    
    # Try optimization with SLSQP
    try:
        # Define constraints: separation for all pairs
        def separation_constraints(params):
            centers = params[:22].reshape(11, 2)
            angles = params[22:33]
            constraints = []
            for i in range(11):
                for j in range(i + 1, 11):
                    gap = pair_gap(centers[i], centers[j], angles[i], angles[j])
                    constraints.append(gap - 1e-7)
            return np.array(constraints)
        
        res = minimize(objective, init_params, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': separation_constraints}],
                       options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.fun < best_side:
            centers = res.x[:22].reshape(11, 2)
            angles = res.x[22:33]
            theta = res.x[33]
            if is_valid(centers, angles, theta):
                side = compute_side(centers, angles, theta)
                if side < best_side:
                    best_side = side
                    best_params = res.x.copy()
    except Exception:
        pass
    
    # Try differential evolution for global search
    if time.time() < deadline - 120:
        try:
            # Bounds: centers in [-5, 5], angles in [0, 60), theta in [0, 60)
            bounds = [(-5, 5)] * 22 + [(0, 60)] * 11 + [(0, 60)]
            
            def de_objective(params):
                if time.time() > deadline - 10:
                    return 1e10
                centers = params[:22].reshape(11, 2)
                angles = params[22:33]
                theta = params[33]
                
                for i in range(11):
                    for j in range(i + 1, 11):
                        gap = pair_gap(centers[i], centers[j], angles[i], angles[j])
                        if gap <= 1e-6:
                            return 1e8 + (1e-6 - gap)
                
                return compute_side(centers, angles, theta)
            
            de_res = differential_evolution(de_objective, bounds, maxiter=100,
                                            popsize=15, seed=42, tol=1e-8,
                                            polish=False)
            
            if de_res.fun < best_side:
                centers = de_res.x[:22].reshape(11, 2)
                angles = de_res.x[22:33]
                theta = de_res.x[33]
                if is_valid(centers, angles, theta):
                    side = compute_side(centers, angles, theta)
                    if side < best_side:
                        best_side = side
                        best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead from best
    if time.time() < deadline - 30:
        try:
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                perturbed = best_params + np.random.randn(34) * 0.01
                # Keep angles in [0, 60)
                perturbed[22:33] = perturbed[22:33] % 60
                perturbed[33] = perturbed[33] % 60
                
                res = minimize(objective, perturbed, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                
                if res.fun < best_side:
                    centers = res.x[:22].reshape(11, 2)
                    angles = res.x[22:33]
                    theta = res.x[33]
                    if is_valid(centers, angles, theta):
                        side = compute_side(centers, angles, theta)
                        if side < best_side:
                            best_side = side
                            best_params = res.x.copy()
        except Exception:
            pass
    
    # Extract final result
    centers = best_params[:22].reshape(11, 2)
    angles = best_params[22:33]
    theta = best_params[33]
    
    # Ensure angles are in [0, 60)
    angles = angles % 60
    theta = theta % 60
    
    # Final validation
    if not is_valid(centers, angles, theta):
        # Fallback to initial configuration
        centers = base_centers
        angles = base_angles
        theta = base_theta
        side = compute_side(centers, angles, theta)
    else:
        side = compute_side(centers, angles, theta)
    
    inner_hex_data = np.column_stack([centers, angles])
    outer_hex_data = np.array([0, 0, theta])
    
    return inner_hex_data, outer_hex_data, side
