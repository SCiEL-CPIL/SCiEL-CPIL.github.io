"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # 150s limit, keep 5s margin
    
    # Helper functions for geometry
    def hex_vertices(cx, cy, side, angle_rad):
        """Generate vertices of a regular hexagon."""
        angles = angle_rad + np.arange(6) * np.pi / 3
        return np.column_stack((cx + side * np.cos(angles), 
                                cy + side * np.sin(angles)))

    def min_outer_hex_side(inner_positions, inner_angles, inner_side=1.0):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, rotatable)
        that contains all given inner hexagons.
        We optimize over the outer hexagon's rotation angle.
        """
        # Collect all vertices of all inner hexagons
        all_verts = []
        for (x, y, ang_deg) in inner_positions:
            ang_rad = math.radians(ang_deg)
            verts = hex_vertices(x, y, inner_side, ang_rad)
            all_verts.append(verts)
        
        if not all_verts:
            return 0.0, 0.0
            
        all_verts = np.vstack(all_verts)
        
        # The width of a regular hexagon with side 's' in direction theta is:
        # W(theta) = s * (2 * cos(theta - alpha)) where alpha is orientation?
        # Actually, for a regular hexagon centered at origin with side s and orientation phi:
        # The support function h(theta) = s * (sqrt(3)/2 * |cos(theta - phi)| + 1/2 * |sin(theta - phi)|) ?
        # Let's use the property: A regular hexagon with side s is the intersection of 6 half-planes.
        # Distance from center to side (apothem) is a = s * sqrt(3)/2.
        # If we rotate the container hexagon by angle beta, its sides are normal to angles beta + k*pi/3.
        # For a point (x,y), the distance to the side normal to theta is x*sin(theta) - y*cos(theta) (assuming appropriate sign).
        # Actually, let's define the container by its apothem 'a'.
        # A point (x,y) is inside if for all k=0..5:
        # x * sin(beta + k*pi/3) - y * cos(beta + k*pi/3) <= a
        # AND
        # -x * sin(beta + k*pi/3) + y * cos(beta + k*pi/3) <= a
        # Wait, the standard form for hexagon sides normal to angles theta_k:
        # n_k . p <= a
        # where n_k = (sin(theta_k), -cos(theta_k)) is a normal vector?
        # Let's use: n_k = (cos(theta_k), sin(theta_k)).
        # Then |n_k . p| <= a is for a square. For hexagon, the normals are 60 deg apart.
        # If normals are at angles phi + k*pi/3, the condition is:
        # max_k ( | (x cos(phi_k) + y sin(phi_k)) | ) <= a ? No.
        
        # Let's derive:
        # Regular hexagon with apothem 'a' and rotation 'phi'.
        # Vertices are at radius R = a / cos(pi/6) = 2a/sqrt(3).
        # The boundary is defined by 6 lines.
        # Line k is at distance 'a' from center, normal angle phi + k*pi/3.
        # Equation: x cos(theta_k) + y sin(theta_k) = a
        # For a point to be inside, we must have:
        # x cos(theta_k) + y sin(theta_k) <= a for all k=0..5
        # AND
        # x cos(theta_k + pi) + y sin(theta_k + pi) <= a  => -x cos(theta_k) - y sin(theta_k) <= a
        # So: |x cos(theta_k) + y sin(theta_k)| <= a for all k=0..5.
        
        # We want to find min 'a' and 'phi' such that for all points p in all_verts:
        # max_k ( | p . n_k(phi) | ) <= a
        # where n_k(phi) = (cos(phi + k*pi/3), sin(phi + k*pi/3))
        
        # Since a = s * sqrt(3)/2, minimizing a is equivalent to minimizing s.
        # s = 2a / sqrt(3)
        
        def required_apothem(phi):
            # For a given rotation phi of the outer hexagon,
            # calculate the max projected distance of any point onto the 6 normal directions.
            # theta_k = phi + k * pi/3
            thetas = phi + np.arange(6) * np.pi / 3
            # Project all points onto these directions
            # projections[i, k] = x_i * cos(theta_k) + y_i * sin(theta_k)
            cos_thetas = np.cos(thetas)
            sin_thetas = np.sin(thetas)
            
            # all_verts is (M, 2)
            # proj is (M, 6)
            proj = all_verts[:, 0] * cos_thetas + all_verts[:, 1] * sin_thetas
            # We need max over points and max over k of |proj|
            # Actually, for a fixed phi, the required apothem is:
            # a_required = max_i max_k | proj[i, k] |
            max_abs_proj = np.max(np.abs(proj))
            return max_abs_proj

        # Optimize phi. The function required_apothem(phi) is periodic with period pi/3?
        # Rotating the hexagon by pi/3 maps it to itself (but normals rotate).
        # Actually, the set of normals {phi + k*pi/3} is the same for phi and phi+pi/3.
        # So the function is periodic with period pi/3.
        # We can search phi in [0, pi/3).
        
        # Coarse search
        best_a = float('inf')
        best_phi = 0.0
        num_steps = 100
        for i in range(num_steps):
            phi = i * (np.pi / 3) / num_steps
            a = required_apothem(phi)
            if a < best_a:
                best_a = a
                best_phi = phi
        
        # Fine search around best_phi
        step = (np.pi / 3) / num_steps
        for i in range(-10, 11):
            phi = best_phi + i * step / 10
            a = required_apothem(phi)
            if a < best_a:
                best_a = a
                best_phi = phi
                
        side = 2 * best_a / np.sqrt(3)
        return side, best_phi

    # Strategy:
    # 1. Generate a hexagonal lattice arrangement of 12 hexagons.
    #    Center + 6 surrounding + 5 more?
    #    A cluster of 7 (center+6) fits in a hexagon of side 3 (if touching).
    #    We need 12.
    #    Let's try a "flower" shape with more layers or a rectangular grid in hex coordinates.
    
    # Hex coordinate system (axial):
    # x = 1.5 * s * q
    # y = sqrt(3) / 2 * s * (q + 2*r)
    # For unit hexagons (side 1), distance between centers is sqrt(3).
    # Wait, distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    
    # Let's generate a set of 12 positions in hex grid coordinates.
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1) -> 7 total
    # Need 5 more.
    # (2,0), (2,-1), (2,-2), (-2,0), (-2,1)
    
    coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (2, -1), (2, -2), (-2, 0), (-2, 1)
    ]
    
    # Convert to cartesian
    # q, r
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # Distance check:
    # (0,0) to (1,0): x=sqrt(3), y=0. dist=sqrt(3). Correct.
    # (0,0) to (0,1): x=sqrt(3)/2, y=1.5. dist=sqrt(3/4 + 2.25) = sqrt(3). Correct.
    
    positions = []
    for q, r in coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        positions.append([x, y, 0.0]) # angle 0 initially
    
    # Optimize the arrangement?
    # For now, just calculate the bounding hexagon for this fixed lattice.
    # The lattice is symmetric, so the optimal outer hexagon should align with it.
    
    side, angle_rad = min_outer_hex_side(positions, [0.0]*N)
    angle_deg = math.degrees(angle_rad)
    
    # Check validity: Are inner hexagons disjoint?
    # In a hex lattice with side 1, touching hexagons share edges.
    # The problem says "disjoint". Usually this means non-overlapping interiors.
    # Sharing edges is allowed in packing problems unless "strictly disjoint" (positive distance) is specified.
    # Given "place 12 disjoint... hexagons", standard interpretation is interiors do not overlap.
    # My lattice has them touching. This is valid for packing.
    
    # However, if the evaluator requires a gap, we might need to scale up slightly or shift.
    # But minimizing side length implies we want them as close as possible.
    # Let's assume touching is allowed.
    
    # Refinement: Try rotating the inner hexagons?
    # If we rotate inner hexagons by 30 degrees, they fit into the "holes" of the lattice?
    # No, the lattice is defined by centers. The orientation of inner hexagons affects the bounding box.
    # Let's try optimizing the orientation of the inner hexagons too.
    # Due to symmetry, all inner hexagons likely have the same optimal orientation.
    
    best_side = side
    best_inner = positions
    best_outer_angle = angle_deg
    
    # Try a few rotations for the inner hexagons
    for inner_rot in range(0, 30, 5): # 0, 5, 10, 15, 20, 25
        if time.time() > deadline:
            break
        rotated_positions = [[x, y, float(inner_rot)] for x, y, _ in positions]
        s, a = min_outer_hex_side(rotated_positions, [float(inner_rot)]*N)
        if s < best_side:
            best_side = s
            best_inner = rotated_positions
            best_outer_angle = math.degrees(a)
            
    # Try scaling the lattice slightly? No, we want to minimize the container.
    # The lattice is fixed by the "unit" constraint.
    
    # One more check: Is the "flower" of 7 + 5 the best shape?
    # A 3x4 rectangle in hex coords might be different?
    # Let's try a rectangular block in hex coords:
    # q from -1 to 1, r from -1 to 1? That's 9.
    # q from -1 to 2, r from -1 to 1?
    # Let's stick with the cluster for now, it's compact.
    
    return best_inner, (0.0, 0.0), best_side, best_outer_angle
# EVOLVE-BLOCK-END
