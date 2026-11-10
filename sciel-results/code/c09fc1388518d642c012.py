# EVOLVE-BLOCK-START
import numpy as np
import time

def hexagon_packing_11():
    start_time = time.time()
    time_limit = 600.0
    n = 11

    # Precompute geometry for unit hexagons
    # Vertices of a unit regular hexagon (side 1) with angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    hex_angles = np.deg2rad(np.arange(0, 360, 60))
    hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # Normals of the edges of a regular hexagon
    # Edge 0 is between vertex 0 and 1. Midpoint angle is 30 deg.
    # Normals point outward.
    normal_angles = np.deg2rad(np.arange(30, 360, 60))
    hex_normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    def get_vertices(center):
        return center + hex_verts

    def hex_separated(c1, c2):
        """
        Check if two hexagons at centers c1, c2 are separated.
        Returns True if separated (valid), False if intersecting.
        Uses the 12 edge normals of the Minkowski sum (which are the same as the hexagon normals).
        Actually, for two convex polygons A and B, they are separated if there exists a separating axis.
        The separating axes are the normals of the edges of A and B.
        For regular hexagons, these are the 6 normals of the hexagon.
        Wait, the Minkowski sum of two hexagons has 6 normals?
        Let's verify.
        If they are touching, the gap is 0.
        If they are separated, the gap is > 0.
        The condition "projected intervals are apart by more than 1e-6" means strict separation.
        However, the task says "touching or a smaller gap is an intersection".
        So we need gap > 1e-6? Or just >=?
        "apart by more than 1e-6" -> gap > 1e-6.
        
        For two convex polygons, the maximum gap in direction n is:
        max_v (v dot n) - min_v (v dot n) for the combined set?
        No.
        Let A and B be the two hexagons.
        Projection of A onto n: [min_A, max_A]
        Projection of B onto n: [min_B, max_B]
        They are separated in direction n if max_A < min_B or max_B < min_A.
        The gap is min_B - max_A or min_A - max_B.
        
        We need to check if there exists ANY normal n (from the 12 edge normals of the two hexagons)
        such that they are separated.
        Actually, if they are convex and disjoint, there is a separating axis.
        For polygons, the separating axis is always parallel to an edge of one of the polygons.
        So we check the 12 normals (6 from A, 6 from B). Since they are identical regular hexagons
        (assuming same orientation for simplicity in the physics sim, or we can handle rotation),
        the normals are the same 6 directions.
        
        Wait, if the hexagons can rotate, the normals change.
        In my physics simulation, I will keep them at angle 0 for simplicity initially?
        No, the task allows angle. But for 11 unit hexagons in a small hexagon,
        optimal packing often has aligned hexagons or specific rotations.
        However, allowing rotation increases dimensionality (11*3 = 33 vars).
        Physics sim with rotation is complex.
        
        Let's try to simulate with fixed angle 0 first.
        If that yields a good result, great.
        If not, I might need to add rotation.
        Given the "different strategy" requirement, a physics sim with fixed orientation is still a different strategy 
        from the gradient-based optimization.
        
        Let's assume fixed angle 0 for the inner hexagons.
        """
        v1 = get_vertices(c1)
        v2 = get_vertices(c2)
        
        # Check 6 normals
        for i in range(6):
            n = hex_normals[i]
            p1 = v1 @ n
            p2 = v2 @ n
            
            min1, max1 = np.min(p1), np.max(p1)
            min2, max2 = np.min(p2), np.max(p2)
            
            # Separated if max1 < min2 or max2 < min1
            if max1 < min2 - 1e-6:
                return True
            if max2 < min1 - 1e-6:
                return True
        return False

    def check_all_separated(centers):
        for i in range(n):
            for j in range(i+1, n):
                if not hex_separated(centers[i], centers[j]):
                    return False
        return True

    def compute_outer_side(centers, outer_center):
        """
        Calculate the side length of the outer hexagon required to contain all inner hexagons.
        Outer hexagon is regular, centered at outer_center, angle 0.
        The condition for a point p to be inside the outer hexagon (side S, center C, angle 0) is:
        For each of the 6 normals n_k of the outer hexagon:
        dot(p - C, n_k) <= S * sqrt(3) / 2
        (Apothem of outer hexagon is S * sqrt(3) / 2)
        
        So S >= max_k ( dot(p - C, n_k) ) / (sqrt(3)/2)
        We take the max over all vertices of all inner hexagons.
        """
        apothem_factor = np.sqrt(3) / 2
        max_proj = 0.0
        for c in centers:
            verts = get_vertices(c)
            shifted = verts - outer_center
            projs = shifted @ hex_normals.T
            max_proj = max(max_proj, np.max(projs))
        
        return max_proj / apothem_factor

    def check_containment(centers, outer_center, side):
        """
        Check if all inner hexagons are inside the outer hexagon.
        """
        apothem = side * np.sqrt(3) / 2
        for c in centers:
            verts = get_vertices(c)
            shifted = verts - outer_center
            projs = shifted @ hex_normals.T
            if np.any(projs > apothem + 1e-6):
                return False
        return True

    def is_valid(centers, outer_center, side):
        if not check_all_separated(centers):
            return False
        if not check_containment(centers, outer_center, side):
            return False
        # Check side >= sqrt(11)
        if side < np.sqrt(11):
            return False
        return True

    # Physics Simulation
    # State: positions (n, 2)
    # Initial guess: Random or Grid
    np.random.seed(42)
    
    # Try a few initial configurations
    best_result = None
    best_side = float('inf')
    
    configs = []
    
    # Config 1: Hexagonal grid cluster (compact)
    # 1 center, 6 neighbors, 4 more?
    # Let's just use random points in a circle for the physics sim to find its own way.
    
    # But to help it, we can provide a structured start.
    # Start 1: Random
    c0 = np.random.rand(n, 2) * 4 - 2
    configs.append(c0.copy())
    
    # Start 2: Hex grid
    # Center (0,0)
    # Neighbors at distance sqrt(3)
    d = np.sqrt(3)
    angles = np.deg2rad(np.arange(0, 360, 60))
    grid = []
    grid.append(np.array([0.0, 0.0]))
    for a in angles:
        grid.append(np.array([d * np.cos(a), d * np.sin(a)]))
    # That's 7. Need 4 more.
    # Next ring? Distance 2*sqrt(3)?
    # Or fill gaps?
    # Let's just append 4 random points near the cluster for the start.
    for _ in range(4):
        a = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, 2)
        grid.append(np.array([r * np.cos(a), r * np.sin(a)]))
    configs.append(np.array(grid))
    
    # Start 3: Linear chain
    chain = []
    for i in range(n):
        chain.append(np.array([i * d, 0.0]))
    configs.append(np.array(chain))

    for init_centers in configs:
        centers = init_centers.copy()
        outer_center = np.mean(centers, axis=0)
        
        # Simulation parameters
        dt = 0.05
        damping = 0.9
        repulsion_strength = 10.0
        boundary_strength = 100.0
        iterations = 5000
        
        for step in range(iterations):
            if time.time() - start_time > time_limit - 30:
                break
                
            # Compute forces
            forces = np.zeros_like(centers)
            
            # 1. Repulsion between hexagons
            for i in range(n):
                for j in range(i+1, n):
                    c1 = centers[i]
                    c2 = centers[j]
                    vec = c2 - c1
                    dist = np.linalg.norm(vec)
                    
                    # Approximate collision if dist < 2.0 (diameter of hex is 2)
                    # Actually, min distance between centers for touching is sqrt(3) ~ 1.732
                    # Max distance is 2.0
                    # If dist < sqrt(3), they definitely overlap.
                    # If dist < 2.0, they might overlap depending on angle.
                    # For simplicity in force calc, use a soft potential.
                    # If dist < 2.0, push apart.
                    if dist < 2.0:
                        if dist < 1e-6:
                            vec = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)])
                            dist = np.linalg.norm(vec)
                        force_mag = repulsion_strength * (2.0 - dist)
                        force_vec = force_mag * (vec / dist)
                        forces[i] -= force_vec
                        forces[j] += force_vec
            
            # 2. Boundary force (confinement)
            # We want to minimize the outer side length.
            # This is equivalent to pushing hexagons towards the center.
            # Or, we can simulate a shrinking box.
            # Let's calculate the current required side length and apply a force to reduce it.
            # A simpler approach: Attract all centers to the outer_center (0,0) with a spring force.
            # This compresses the cluster.
            for i in range(n):
                vec_to_center = -centers[i]
                dist_to_center = np.linalg.norm(vec_to_center)
                if dist_to_center > 1e-6:
                    force_vec = boundary_strength * 0.1 * vec_to_center # Weak attraction
                    forces[i] += force_vec
            
            # Update velocities (implicit in Verlet or simple Euler)
            # Let's use simple Euler with damping
            # We need to store velocities.
            if not hasattr(hexagon_packing_11, 'velocities'):
                hexagon_packing_11.velocities = np.zeros_like(centers)
            
            # Apply forces
            hexagon_packing_11.velocities += forces * dt
            hexagon_packing_11.velocities *= damping
            centers += hexagon_packing_11.velocities * dt
            
            # Update outer center to be the centroid (keeps it centered)
            outer_center = np.mean(centers, axis=0)
            
            # Check validity and update best
            side = compute_outer_side(centers, outer_center)
            if is_valid(centers, outer_center, side):
                if side < best_side:
                    best_side = side
                    best_result = (centers.copy(), outer_center.copy(), side)
        
        # If we ran out of time, break
        if time.time() - start_time > time_limit - 30:
            break

    if best_result is None:
        # Fallback: Return a valid but potentially suboptimal configuration
        # Use the last state of the first config
        centers = configs[0].copy()
        outer_center = np.mean(centers, axis=0)
        side = compute_outer_side(centers, outer_center)
        # Force validity by expanding if necessary?
        # The task requires side >= sqrt(11).
        if side < np.sqrt(11):
            side = np.sqrt(11) + 0.1
        # Ensure separation by scaling? No, that might break containment.
        # Just return what we have. The evaluator will mark it invalid if so.
        # But we should try to make it valid.
        # Simple fallback: Grid layout with large enough outer hex.
        d = np.sqrt(3)
        grid = []
        grid.append(np.array([0.0, 0.0]))
        angles = np.deg2rad(np.arange(0, 360, 60))
        for a in angles:
            grid.append(np.array([d * np.cos(a), d * np.sin(a)]))
        # Add 4 more in a second ring
        for i in range(4):
            a = np.deg2rad(30 + i * 90)
            r = 2 * d
            grid.append(np.array([r * np.cos(a), r * np.sin(a)]))
        centers = np.array(grid)
        outer_center = np.mean(centers, axis=0)
        side = compute_outer_side(centers, outer_center)
        best_result = (centers, outer_center, side)

    centers, outer_center, side = best_result
    
    # Final refinement: Try to reduce side slightly by moving outer center?
    # The compute_outer_side already optimizes for the given centers.
    # Moving outer center might help if the centroid is not the optimal center.
    # But for a regular hexagon containing a set of points, the center that minimizes the max projection
    # is not necessarily the centroid.
    # However, the physics sim kept it at the centroid.
    # Let's do a quick local search on outer center.
    
    best_side_local = side
    best_outer_center = outer_center
    
    # Simple grid search for outer center around the current one
    for dx in np.linspace(-0.5, 0.5, 5):
        for dy in np.linspace(-0.5, 0.5, 5):
            oc = outer_center + np.array([dx, dy])
            s = compute_outer_side(centers, oc)
            if s < best_side_local:
                if is_valid(centers, oc, s):
                    best_side_local = s
                    best_outer_center = oc

    side = best_side_local
    outer_center = best_outer_center

    # Format result
    inner_hex_data = np.zeros((n, 3))
    inner_hex_data[:, 0] = centers[:, 0]
    inner_hex_data[:, 1] = centers[:, 1]
    inner_hex_data[:, 2] = 0.0  # Angle in degrees
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], 0.0])
    
    return inner_hex_data, outer_hex_data, float(side)

# EVOLVE-BLOCK-END
