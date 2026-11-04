import numpy as np

N = 11

def construct():
    # Generate 11 hexagon centers in a compact cluster
    # Axial coordinates (q, r)
    # 7 center cluster
    coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1)
    ]
    # Add 4 more to make 11, keeping it compact
    # Extend in one direction
    coords.extend([
        (2, 0),
        (2, -1),
        (1, -2),
        (0, -2)
    ])
    
    # Convert axial to Cartesian (flat-top)
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in coords:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append([x, y])
    
    centers = np.array(centers)
    
    # Center the cluster at the origin
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Inner hexagons: (x, y, angle)
    # Angle 0 for all
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    # Outer hexagon parameters
    # We need to find the minimal side length S and angle alpha
    # such that all unit hexagons (aligned) fit inside.
    # Inradius of unit hexagon r_in = sqrt(3)/2
    r_in = sqrt3 / 2.0
    
    best_S = float('inf')
    best_alpha = 0.0
    best_center = [0.0, 0.0]
    
    # Try angles from 0 to 60 degrees
    # The lattice is symmetric, so 0 to 60 is enough.
    # Step 1 degree for speed, or just check 0 and 30.
    # Let's check a few angles to be safe.
    for alpha_deg in np.linspace(0, 60, 61):
        alpha = np.deg2rad(alpha_deg)
        
        # Normals to the sides of the outer hexagon
        # A regular hexagon has sides with normals at alpha, alpha+60, ...
        # The apothem direction is alpha.
        # Normals: alpha + k * 60 degrees for k=0..5
        normals = []
        for k in range(6):
            theta = alpha + k * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        # For each normal, find the max projection of centers + r_in
        # The distance from center C to the side in direction n is A (apothem)
        # We need (c_i - C) . n <= A - r_in ?
        # Wait, the unit hexagon is aligned. Its support in direction n is r_in if n is normal to a side?
        # If the unit hexagon is aligned with the outer one, and n is normal to the outer side,
        # then the unit hexagon's face is parallel to the outer side.
        # The distance from the unit hexagon center to its face in direction n is r_in.
        # So the condition is (c_i - C) . n + r_in <= A
        # So A >= max_i ((c_i - C) . n + r_in)
        
        # We want to minimize S = 2 * A / sqrt(3)
        # We can choose C to minimize the max A.
        # For a fixed set of normals, the optimal C is the center of the minimal enclosing circle?
        # No, it's the Chebyshev center in the hexagonal metric.
        # However, for a compact cluster, the centroid is a good approximation.
        # Let's assume C = [0,0] (centroid) for now.
        
        C = [0.0, 0.0]
        
        max_A = 0.0
        for n in normals:
            projections = centers @ n
            max_proj = np.max(projections)
            required_A = max_proj + r_in
            if required_A > max_A:
                max_A = required_A
        
        S = 2 * max_A / sqrt3
        
        if S < best_S:
            best_S = S
            best_alpha = alpha_deg
            best_center = C
            
    return inner, best_center, best_S, best_alpha

# EVOLVE-BLOCK-START
# The code above is the construct function.
# We need to wrap it in the expected format.
# The prompt asks to keep the function names and signatures.
# The current program has:
# N = 11
# def construct(): ...
# EVOLVE-BLOCK-START
# def construct(): ...
# EVOLVE-BLOCK-END

# I will place the code inside the block.
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    N = 11
    
    # Generate 11 hexagon centers in a compact cluster
    # Axial coordinates (q, r)
    # 7 center cluster
    coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1)
    ]
    # Add 4 more to make 11, keeping it compact
    # Extend in one direction
    coords.extend([
        (2, 0),
        (2, -1),
        (1, -2),
        (0, -2)
    ])
    
    # Convert axial to Cartesian (flat-top)
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in coords:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append([x, y])
    
    centers = np.array(centers)
    
    # Center the cluster at the origin
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Inner hexagons: (x, y, angle)
    # Angle 0 for all
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    # Outer hexagon parameters
    # We need to find the minimal side length S and angle alpha
    # such that all unit hexagons (aligned) fit inside.
    # Inradius of unit hexagon r_in = sqrt(3)/2
    r_in = sqrt3 / 2.0
    
    best_S = float('inf')
    best_alpha = 0.0
    best_center = [0.0, 0.0]
    
    # Try angles from 0 to 60 degrees
    # The lattice is symmetric, so 0 to 60 is enough.
    # Step 1 degree for speed.
    for alpha_deg in np.linspace(0, 60, 61):
        alpha = np.deg2rad(alpha_deg)
        
        # Normals to the sides of the outer hexagon
        # A regular hexagon has sides with normals at alpha, alpha+60, ...
        # The apothem direction is alpha.
        # Normals: alpha + k * 60 degrees for k=0..5
        normals = []
        for k in range(6):
            theta = alpha + k * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        # For each normal, find the max projection of centers + r_in
        # We assume C = [0,0] (centroid)
        
        C = [0.0, 0.0]
        
        max_A = 0.0
        for n in normals:
            projections = centers @ n
            max_proj = np.max(projections)
            required_A = max_proj + r_in
            if required_A > max_A:
                max_A = required_A
        
        S = 2 * max_A / sqrt3
        
        if S < best_S:
            best_S = S
            best_alpha = alpha_deg
            best_center = C
            
    return inner, best_center, best_S, best_alpha
# EVOLVE-BLOCK-END
