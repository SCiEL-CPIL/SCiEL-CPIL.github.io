import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 7 central hexagons
    # Lattice vectors
    u = np.array([math.sqrt(3), 0])
    v = np.array([math.sqrt(3)/2, 1.5])
    
    centers = [np.array([0.0, 0.0])]
    # 6 neighbors
    neighbors = [
        u, v, v-u, -u, -v, u-v
    ]
    for n in neighbors:
        centers.append(n)
        
    # 4 more from the next shell
    # The next shell centers are at 2u, 2v, 2(v-u), -2u, -2v, 2(u-v)
    # Let's pick 4 that form a compact shape
    # Picking u*2, v*2, (v-u)*2, (u-v)*2 ?
    # Let's try all combinations of 4 from the 6 available spots
    # and all rotations of the container
    
    shell2_spots = [
        2*u, 2*v, 2*(v-u), -2*u, -2*v, 2*(u-v)
    ]
    
    best_S = float('inf')
    best_config = None
    best_angle = 0.0
    
    # Try different sets of 4 outer hexagons
    # There are 15 combinations
    from itertools import combinations
    for combo in combinations(range(6), 4):
        outer_centers = [shell2_spots[i] for i in combo]
        all_centers = centers + outer_centers
        
        # Try different rotations of the container
        # The cluster has some symmetry, but the selection of 4 breaks it.
        # We need to find the rotation of the container that minimizes S.
        # This is equivalent to finding the minimal enclosing hexagon.
        # We can sample container angles.
        
        for angle_deg in np.linspace(0, 60, 61):
            angle_rad = np.radians(angle_deg)
            # The container is a regular hexagon.
            # We need to find the minimal S such that all inner hexagons are inside.
            # This is a linear programming problem or can be solved by checking constraints.
            # For a hexagon with side S and rotation phi, the inequality for a point p to be inside is:
            # max_{k=0..5} dot(p, n_k) <= S * sqrt(3) / 2
            # where n_k are the unit normals to the sides.
            
            # Normals for a hexagon rotated by angle_rad
            # Sides are at angles angle_rad, angle_rad+60, ...
            # Normals are perpendicular to sides.
            # If side normal is at angle theta, the constraint is p . n(theta) <= apothem.
            # Apothem = S * sqrt(3) / 2.
            
            # So S >= max_{p in all_vertices} max_{k} dot(p, n_k) * 2 / sqrt(3)
            
            # We need to check all vertices of all inner hexagons.
            # Each inner hexagon has 6 vertices.
            # Inner hexagons are all rotated by 0 (aligned with lattice).
            # Vertices of a unit hexagon centered at c, aligned with x-axis (vertex at 0):
            # c + [1, 0], c + [0.5, sqrt(3)/2], ...
            
            max_req_S = 0.0
            for c in all_centers:
                # Generate vertices of the inner hexagon
                # Standard orientation: vertex at (1,0)
                for k in range(6):
                    theta = np.radians(60 * k)
                    vx = c[0] + math.cos(theta)
                    vy = c[1] + math.sin(theta)
                    p = np.array([vx, vy])
                    
                    # Check against container sides
                    # Container normals
                    for j in range(6):
                        n_theta = angle_rad + np.radians(60 * j)
                        n = np.array([math.cos(n_theta), math.sin(n_theta)])
                        dist = np.dot(p, n)
                        # dist <= S * sqrt(3) / 2
                        # S >= dist * 2 / sqrt(3)
                        req_S = dist * 2.0 / math.sqrt(3)
                        if req_S > max_req_S:
                            max_req_S = req_S
            
            if max_req_S < best_S:
                best_S = max_req_S
                best_config = all_centers
                best_angle = angle_deg

    # Format the output
    # inner: list of [x, y, angle]
    # angle is 0.0 for all as they are aligned
    inner = [[c[0], c[1], 0.0] for c in best_config]
    center = [0.0, 0.0]
    side = best_S
    angle = best_angle
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
