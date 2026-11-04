import numpy as np
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 3x4 honeycomb grid of unit hexagon centers
    # u vector: distance between centers in x direction is sqrt(3)
    # v vector: distance between centers in y direction is 1.5, x offset sqrt(3)/2
    # Let's generate centers for i in 0..2, j in 0..3
    centers = []
    for i in range(3):
        for j in range(4):
            x = i * math.sqrt(3) + (j % 2) * (math.sqrt(3) / 2) # Wait, standard honeycomb
            # Let's use a simpler coordinate system.
            # Row 0: x = 0, sqrt(3), 2sqrt(3)
            # Row 1: x = sqrt(3)/2, 3sqrt(3)/2, 5sqrt(3)/2, 7sqrt(3)/2
            # This creates a parallelogram.
            # Let's use the standard basis vectors for hex grid centers.
            # d1 = (sqrt(3), 0)
            # d2 = (sqrt(3)/2, 1.5)
            # Center(i,j) = i*d1 + j*d2
            x = i * math.sqrt(3) + j * (math.sqrt(3) / 2)
            y = j * 1.5
            centers.append([x, y])
    
    centers = np.array(centers)
    centroid = np.mean(centers, axis=0)
    centers -= centroid # Center the cluster at origin
    
    # Unit hexagon vertices (pointy top, side 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    hex_verts = np.array([[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)])
    
    def support_hex(theta):
        # Max projection of unit hexagon on direction theta
        u = np.array([math.cos(theta), math.sin(theta)])
        return np.max(hex_verts @ u)
        
    def calculate_S(alpha):
        # Rotate centers by alpha
        c_alpha = np.cos(alpha)
        s_alpha = math.sin(alpha)
        R = np.array([[c_alpha, -s_alpha], [s_alpha, c_alpha]])
        rot_centers = centers @ R.T
        
        # The hexagons are also rotated by alpha.
        # The support function of a hexagon rotated by alpha in direction phi is support_hex(phi - alpha)
        
        # Outer hexagon normals are at 0, 60, 120, 180, 240, 300
        # We want to minimize the max projection along these normals.
        # S = (2/sqrt(3)) * max_k R_global(n_k)
        
        max_proj = 0.0
        for k in range(6):
            phi = k * math.pi / 3
            u = np.array([math.cos(phi), math.sin(phi)])
            
            # Projection of centers
            center_projs = rot_centers @ u
            max_center_proj = np.max(center_projs)
            
            # Support of the hexagon shape in direction phi
            # Since hexagons are rotated by alpha, we need support in local frame (phi - alpha)
            hex_sup = support_hex(phi - alpha)
            
            proj = max_center_proj + hex_sup
            if proj > max_proj:
                max_proj = proj
                
        return (2.0 / math.sqrt(3.0)) * max_proj

    # Search for optimal alpha
    # The symmetry of the hexagon and the 3x4 grid suggests checking 0 to pi/3
    # Fine grid search
    best_alpha = 0
    best_S = float('inf')
    
    # Initial coarse search
    alphas = np.linspace(0, math.pi/3, 100)
    for a in alphas:
        S = calculate_S(a)
        if S < best_S:
            best_S = S
            best_alpha = a
            
    # Refine around best
    step = (math.pi/3) / 100
    for _ in range(5):
        alphas = np.linspace(best_alpha - step, best_alpha + step, 100)
        for a in alphas:
            S = calculate_S(a)
            if S < best_S:
                best_S = S
                best_alpha = a
        step /= 10
        
    # Final result
    # Rotate centers by best_alpha
    c_alpha = math.cos(best_alpha)
    s_alpha = math.sin(best_alpha)
    R = np.array([[c_alpha, -s_alpha], [s_alpha, c_alpha]])
    final_centers = centers @ R.T
    
    # Construct inner list: (x, y, angle_degrees)
    # The angle of the inner hexagons is best_alpha (in degrees)
    angle_deg = best_alpha * 180.0 / math.pi
    
    inner = []
    for c in final_centers:
        inner.append([float(c[0]), float(c[1]), angle_deg])
        
    # Outer hexagon: center (0,0), side best_S, rotation 0
    # Note: The evaluator might expect the outer hexagon rotation to match the "flat" or "pointy" orientation
    # Our calculation assumed outer hexagon normals at 0, 60... which corresponds to a specific orientation.
    # A regular hexagon with side S and rotation 0 usually has vertices at 0, 60... or sides flat?
    # Standard math: vertices at (S,0), (S/2, S*sqrt(3)/2)...
    # This has a vertex at angle 0.
    # The normals to the sides are at 30, 90, 150...
    # My calculation used normals at 0, 60...
    # If the outer hexagon has vertices at 0, 60..., its sides are "flat" relative to the axes?
    # Let's check the distance from center to side.
    # If vertices are at 0, 60..., the side connecting 0 and 60 is perpendicular to 30.
    # So the normal is at 30.
    # My code used normals at 0, 60...
    # This implies the outer hexagon is rotated by 30 degrees relative to the standard vertex-at-0 hexagon.
    # So if I return rotation 0, I should ensure the geometry matches.
    # If I return rotation 0, and the evaluator uses vertices at 0, 60..., then the normals are at 30, 90...
    # My calculation minimized for normals at 0, 60...
    # So I should return an outer rotation of 30 degrees (pi/6) to align the normals with my calculation?
    # Or I can just say the outer hexagon is rotated by 30 degrees.
    
    # Let's re-verify.
    # If outer hexagon rotation is $\beta$, its normals are at $\beta + 30^\circ, \beta + 90^\circ...$
    # I calculated min S for normals at $0, 60...$
    # So I need $\beta + 30^\circ = 0^\circ \Rightarrow \beta = -30^\circ$ or $330^\circ$.
    # Alternatively, I can just shift my cluster rotation.
    # Let's keep it simple:
    # I found optimal cluster rotation $\alpha$ relative to a hexagon with normals at $0, 60...$
    # If I return outer rotation 0 (vertices at 0, 60...), the normals are at 30, 90...
    # This is a 30 degree shift.
    # So I should rotate my cluster by $\alpha + 30^\circ$?
    # No, the relative angle is what matters.
    # If I set outer rotation to 0, the "effective" normal angles are 30, 90...
    # My `calculate_S` checked normals 0, 60...
    # So the configuration that is optimal for normals 0, 60... is NOT optimal for normals 30, 90...
    # Unless the system has 30 degree symmetry?
    # The 3x4 grid does NOT have 30 degree symmetry.
    # So I must align the outer hexagon to the normals I used in calculation.
    # Normals at 0, 60... correspond to a hexagon with SIDES horizontal?
    # Normal at 0 is horizontal. Side is vertical.
    # Vertices would be at 30, 90, 150...
    # So if I return outer rotation such that vertices are at 30, 90..., that's a rotation of 30 degrees from standard.
    # Let's just return outer rotation = 30.0 (degrees) to match the normals 0, 60... used in calc.
    # Wait, if rotation is 30 deg, vertices are at 30, 90... Normals are at 0, 60...
    # Yes.
    
    outer_rotation = 30.0
    
    return inner, (0.0, 0.0), best_S, outer_rotation
# EVOLVE-BLOCK-END
