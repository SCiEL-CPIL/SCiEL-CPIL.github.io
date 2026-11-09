"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog

N = 12

def construct():
    # We place N unit regular hexagons inside a regular hexagon of side length L.
    # Each inner hexagon has side length 1, can be translated and rotated.
    # The outer hexagon has center (cx, cy), side length L, and rotation angle theta.
    #
    # Key insight: A regular hexagon of side length s can be described as the intersection
    # of 6 half-planes. For a hexagon centered at (cx, cy) with rotation angle theta,
    # the 6 supporting lines are at distances s from the center in 6 directions
    # theta + k*pi/3 for k=0..5.
    #
    # For containment: each vertex of each inner hexagon must lie inside the outer hexagon.
    # Equivalently, for each inner hexagon, all its vertices must satisfy the 6 half-plane
    # constraints of the outer hexagon.
    #
    # However, this is nonlinear if we allow rotation of inner hexagons. 
    # 
    # Alternative approach: Use the fact that the minimum enclosing hexagon problem for
    # convex bodies can be formulated via support functions. But let's use a simpler
    # lattice-based approach with LP.
    #
    # Strategy: Fix the orientation of all inner hexagons (angle=0) and the outer hexagon
    # (angle=0), and place them on a hexagonal lattice. Then optimize positions via LP.
    #
    # Actually, let's think more carefully. The optimal packing of regular hexagons in a
    # regular hexagon is known to be achieved by a hexagonal lattice arrangement.
    #
    # For n=12, we can arrange them in a pattern. Let's try a 3-4-3 or similar arrangement.
    #
    # Let's use a direct geometric approach:
    # Place inner hexagons on a hexagonal lattice with spacing d (center-to-center distance
    # for non-overlapping unit hexagons). Two unit hexagons with the same orientation don't
    # overlap if their centers are at least 2*sqrt(3) apart (the distance between opposite
    # sides is 2*sqrt(3) for a unit hexagon... wait, let me recalculate).
    #
    # A regular hexagon with side length 1:
    # - Distance between opposite sides (width) = 2 * (sqrt(3)/2) = sqrt(3)
    # - Distance between opposite vertices (diameter) = 2
    #
    # For two identical, non-rotated hexagons to be disjoint, the minimum center-to-center
    # distance depends on relative position. If aligned on a hexagonal lattice, the minimum
    # distance is 2*sqrt(3) for adjacent hexagons in the "flat-top" orientation.
    #
    # Actually, for flat-top hexagons (vertices at angles 0, 60, 120, 180, 240, 300),
    # the hexagonal lattice has spacing such that neighbors are at distance 2*sqrt(3) apart.
    #
    # Let's just use a numerical optimization approach with scipy.
    #
    # Given the complexity, let me use a simpler but effective approach:
    # 1. Assume all inner hexagons have angle 0 and outer hexagon has angle 0.
    # 2. Use a hexagonal lattice arrangement.
    # 3. Compute the required outer hexagon size.
    
    # For a regular hexagon of side L centered at origin, the vertices are at:
    # (L*cos(theta + k*pi/3), L*sin(theta + k*pi/3)) for k=0..5
    # With theta=0: vertices at (L,0), (L/2, L*sqrt(3)/2), (-L/2, L*sqrt(3)/2), etc.
    
    # The half-planes of the outer hexagon (side L, angle 0, center (cx,cy)):
    # The 6 normals point in directions 0, 60, 120, 180, 240, 300 degrees.
    # The distance from center to each side is L*sqrt(3)/2.
    # So constraint: n_i . (p - c) <= L*sqrt(3)/2 for each of the 6 normals.
    
    # For an inner hexagon of side 1, angle 0, centered at (x,y):
    # Its vertices are at (x + cos(k*pi/3), y + sin(k*pi/3)) for k=0..5.
    # For containment, each vertex must satisfy all 6 half-plane constraints.
    # The worst case is the vertex farthest in each normal direction.
    # For a unit hexagon with angle 0, the support function in direction phi is:
    # h(phi) = cos(phi - k*pi/3) for the k that minimizes |phi - k*pi/3|... 
    # Actually, for a regular hexagon with vertices at angles 0,60,...,300, the support
    # function in direction phi is max over vertices of (vertex . (cos phi, sin phi)).
    
    # Let's just do a direct optimization using scipy.optimize.minimize with a penalty method.
    
    from scipy.optimize import minimize
    
    def outer_hex_contains(inner_center, inner_angle, outer_center, outer_side, outer_angle):
        """Check if a unit hexagon at inner_center with inner_angle is inside outer hex."""
        # Generate vertices of inner hexagon
        inner_verts = []
        for k in range(6):
            ang = np.radians(inner_angle) + k * np.pi / 3
            vx = inner_center[0] + np.cos(ang)
            vy = inner_center[1] + np.sin(ang)
            inner_verts.append((vx, vy))
        
        # Check each vertex against outer hexagon half-planes
        for (vx, vy) in inner_verts:
            for k in range(6):
                ang = np.radians(outer_angle) + k * np.pi / 3
                nx = np.cos(ang)
                ny = np.sin(ang)
                # Distance from outer center to this vertex along normal
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > outer_side * np.sqrt(3) / 2 + 1e-9:
                    return False
        return True
    
    def no_overlap(i, j, centers, angles):
        """Check if two unit hexagons don't overlap."""
        # Simple check: minimum distance between centers must be sufficient
        # For same orientation, need distance >= 2*sqrt(3) approximately
        # For different orientations, it's more complex. Use a conservative bound.
        dx = centers[i][0] - centers[j][0]
        dy = centers[i][1] - centers[j][1]
        dist = np.sqrt(dx*dx + dy*dy)
        # Conservative: require dist >= 1.9 (should be safe for unit hexagons)
        return dist >= 1.9 - 1e-9
    
    def objective_and_constraints(params):
        """params: [x0,y0, ..., x11,y11, L]"""
        centers = [(params[2*i], params[2*i+1]) for i in range(N)]
        L = params[-1]
        angles = [0.0] * N
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
        # Objective: minimize L
        # Constraints: all inner hexagons inside outer, no overlaps
        violations = 0.0
        for i in range(N):
            if not outer_hex_contains(centers[i], angles[i], outer_center, L, outer_angle):
                # Compute how much it violates
                for k in range(6):
                    ang = np.radians(outer_angle) + k * np.pi / 3
                    nx = np.cos(ang)
                    ny = np.sin(ang)
                    # Max support of inner hex in this direction
                    max_proj = 0
                    for j in range(6):
                        v_ang = np.radians(angles[i]) + j * np.pi / 3
                        vx = centers[i][0] + np.cos(v_ang)
                        vy = centers[i][1] + np.sin(v_ang)
                        proj = vx * nx + vy * ny
                        max_proj = max(max_proj, proj)
                    # Required L for this constraint
                    required = (max_proj - outer_center[0]*nx - outer_center[1]*ny) / (np.sqrt(3)/2)
                    if required > L:
                        violations += (required - L) ** 2
        
        for i in range(N):
            for j in range(i+1, N):
                if not no_overlap(i, j, centers, angles):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    violations += (1.9 - dist) ** 2
        
        return L + 10 * violations
    
    # Initial guess: hexagonal lattice
    # For 12 hexagons, try a 3-4-3 arrangement or similar
    # Hexagonal lattice spacing: 2*sqrt(3) ≈ 3.464
    d = 2 * np.sqrt(3)
    
    # Try a compact arrangement
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons? No, that's 12.
    # Actually 3+4+3 = 10, need 2 more.
    # Let's try 2+3+4+3 = 12
    rows = [2, 3, 4, 3]
    row_y = []
    row_x_starts = []
    y = 0
    for r, count in enumerate(rows):
        row_y.append(y)
        # Offset every other row
        offset = (d / 2) if r % 2 == 1 else 0
        total_width = (count - 1) * d
        x_start = -total_width / 2 + offset
        row_x_starts.append(x_start)
        y += d * np.sqrt(3) / 2
    
    initial_params = []
    idx = 0
    for r, count in enumerate(rows):
        for c in range(count):
            x = row_x_starts[r] + c * d
            y = row_y[r]
            initial_params.append(x)
            initial_params.append(y)
    
    # Estimate L
    max_x = max(abs(p[2*i]) for i in range(N))
    max_y = max(abs(p[2*i+1]) for i in range(N))
    L_guess = max(max_x, max_y) * 2 + 5
    initial_params.append(L_guess)
    
    result = minimize(objective_and_constraints, initial_params, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_params = result.x
    centers = [(best_params[2*i], best_params[2*i+1]) for i in range(N)]
    L = best_params[-1]
    
    # Verify and compute actual required L
    def compute_required_L(centers, angles, outer_center, outer_angle):
        """Compute the minimum L needed."""
        max_L = 0
        for i in range(N):
            for k in range(6):
                ang = np.radians(outer_angle) + k * np.pi / 3
                nx = np.cos(ang)
                ny = np.sin(ang)
                # Max projection of inner hex i in direction (nx, ny)
                max_proj = -np.inf
                for j in range(6):
                    v_ang = np.radians(angles[i]) + j * np.pi / 3
                    vx = centers[i][0] + np.cos(v_ang)
                    vy = centers[i][1] + np.sin(v_ang)
                    proj = vx * nx + vy * ny
                    max_proj = max(max_proj, proj)
                # The constraint is: max_proj - (outer_center . n) <= L * sqrt(3)/2
                oc_proj = outer_center[0] * nx + outer_center[1] * ny
                required_L = (max_proj - oc_proj) / (np.sqrt(3) / 2)
                max_L = max(max_L, required_L)
        return max_L
    
    angles = [0.0] * N
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    L_final = compute_required_L(centers, angles, outer_center, outer_angle)
    # Add small margin
    L_final *= 1.001
    
    inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
    return inner, outer_center, L_final, outer_angle

# EVOLVE-BLOCK-END
