"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Unit regular hexagon vertices (pointy-top orientation)
    # For a unit side length hexagon, vertices at angles 30, 90, 150, 210, 270, 330 degrees
    # But let's use flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees
    # Actually, let's define a standard unit hexagon with vertices at:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # This is a "pointy-right" or flat-top hexagon with circumradius = 1
    
    # For containment in a regular hexagon of side length S, center at origin, rotation 0:
    # The outer hexagon has vertices at S*(cos(k*60°), S*sin(k*60°)) for k=0..5
    # The support function in direction theta is S * max_k cos(theta - k*60°)
    # For a convex body to be inside, its support function in all directions must be <= outer's
    
    # For a regular hexagon of side s centered at c with rotation phi:
    # Support function: h(theta) = c·n + s * max_k cos((theta - phi) - k*60°)
    # where n = (cos theta, sin theta)
    
    # Actually, for a regular hexagon with side length s, the distance from center to each side (apothem) is s*sqrt(3)/2
    # The support function in direction n is: apothem + (projection of center onto n)
    # Wait, no. For a regular hexagon, the support function depends on the direction.
    
    # Let me think more carefully. A regular hexagon with side length s can be described as the intersection of 6 half-planes.
    # For a hexagon centered at origin with one vertex pointing right (flat-top), the sides are at angles 30, 90, 150, 210, 270, 330 degrees from the x-axis.
    # The distance from center to each side is the apothem = s * sqrt(3) / 2.
    
    # For a hexagon with center c, side length s, rotation phi:
    # The normal directions to the sides are at angles phi + 30° + k*60° for k=0..5
    # The constraint for a point p to be inside is: (p - c) · n_k <= apothem for all k
    # where n_k = (cos(phi + 30° + k*60°), sin(phi + 30° + k*60°))
    # and apothem = s * sqrt(3) / 2
    
    # So for all inner hexagon vertices to be inside the outer hexagon:
    # For each inner hexagon i with center c_i and rotation phi_i:
    #   For each vertex v_j of hexagon i:
    #     (c_i + R(phi_i) * v_j - c_outer) · n_k <= s_outer * sqrt(3) / 2 for all k=0..5
    
    # This is equivalent to: for each inner hexagon i and each outer normal direction n_k:
    #   (c_i - c_outer) · n_k + max_j (R(phi_i) * v_j) · n_k <= s_outer * sqrt(3) / 2
    
    # The max_j (R(phi_i) * v_j) · n_k is the support function of the unit hexagon in direction n_k, evaluated after rotation.
    # For a unit hexagon (side 1), the support function in direction theta is:
    #   h(theta) = max over vertices of v · (cos theta, sin theta)
    # For the unit hexagon with vertices at angles 0, 60, ..., 300 degrees:
    #   h(theta) = cos(theta mod 60° - 30°) * ... let me just compute it numerically or use the formula.
    
    # Actually, for a regular hexagon with circumradius R (side = R), the support function is:
    # h(theta) = R * cos(theta - 30°) for theta in [0°, 60°], etc.
    # More precisely, h(theta) = R * cos((theta mod 60°) - 30°) where we take the appropriate branch.
    # For R=1 (unit side length hexagon has circumradius 1):
    
    def hex_support(theta):
        """Support function of unit regular hexagon (side 1, circumradius 1) in direction theta."""
        # Reduce theta to [0, 60)
        theta_mod = theta % (np.pi / 3)
        # The support is cos(theta_mod - pi/6)
        return np.cos(theta_mod - np.pi / 6)
    
    def hex_support_vec(thetas):
        """Vectorized support function."""
        thetas_mod = thetas % (np.pi / 3)
        return np.cos(thetas_mod - np.pi / 6)
    
    # Vertices of unit hexagon
    hex_verts = np.array([[np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)])
    
    def objective(params):
        """
        params: [x1, y1, phi1, x2, y2, phi2, ..., xN, yN, phiN, cx, cy, S, phi_outer]
        Returns: S (outer side length) + penalty for violations
        """
        n = N
        # Extract positions and rotations of inner hexagons
        centers = params[:3*n].reshape(n, 3)
        outer_center = params[3*n:3*n+2]
        S = params[3*n+2]
        phi_outer = params[3*n+3]
        
        # Outer hexagon normal directions
        outer_normals = np.array([phi_outer + np.pi/6 + k*np.pi/3 for k in range(6)])
        
        apothem_outer = S * np.sqrt(3) / 2
        
        # Check all inner hexagon vertices against outer hexagon constraints
        # For each inner hexagon, for each of its 6 vertices, check against 6 outer half-planes
        # Total: n * 6 * 6 = 396 constraints for n=11
        
        total_violation = 0.0
        
        for i in range(n):
            cx_i, cy_i, phi_i = centers[i]
            # Rotate hexagon vertices
            cos_phi = np.cos(phi_i)
            sin_phi = np.sin(phi_i)
            rot_matrix = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
            verts_i = hex_verts @ rot_matrix.T + np.array([cx_i, cy_i])
            
            for k in range(6):
                n_dir = outer_normals[k]
                n_vec = np.array([np.cos(n_dir), np.sin(n_dir)])
                # Project all vertices onto this normal
                projections = verts_i @ n_vec
                max_proj = np.max(projections)
                # The constraint: max_proj - outer_center · n_vec <= apothem_outer
                oc_proj = outer_center @ n_vec
                violation = max_proj - oc_proj - apothem_outer
                if violation > 0:
                    total_violation += violation * violation
        
        return S + 100.0 * total_violation
    
    # Initial guess: place hexagons in a honeycomb pattern
    # Honeycomb spacing: horizontal distance = 1.5, vertical distance = sqrt(3)/2
    # For N=11, let's try a compact arrangement
    
    # Let's use a hexagonal packing arrangement
    # For 11 hexagons, we can do rows of 3, 4, 3 or similar
    
    # Let me try a simple linear arrangement first, then optimize
    # Actually, let's use a honeycomb cluster
    
    def initial_guess():
        """Generate initial positions for 11 unit hexagons in a honeycomb pattern."""
        positions = []
        # Center hexagon at origin
        positions.append([0.0, 0.0, 0.0])
        
        # Ring 1: 6 hexagons around center
        # In honeycomb, neighbors are at distance 2 (for unit hexagons with circumradius 1,
        # the center-to-center distance for touching hexagons is 2*apothem = sqrt(3))
        # Wait, for two unit hexagons to touch, center distance = 2 * apothem = sqrt(3)
        # No, that's not right either. Let me think...
        
        # Two unit regular hexagons (side 1) touching side-to-side have center distance = 2 * apothem = sqrt(3)
        # But if they're in a honeycomb, the neighbors are at angles 0, 60, 120, 180, 240, 300 degrees
        # at distance sqrt(3)
        
        for k in range(6):
            angle = k * np.pi / 3
            x = np.sqrt(3) * np.cos(angle)
            y = np.sqrt(3) * np.sin(angle)
            positions.append([x, y, 0.0])
        
        # We have 7, need 4 more. Add them in a second ring or extend.
        # Let's add 4 more in a line or compact arrangement.
        # Actually, for 11, let's do a different arrangement.
        
        # Let me restart with a better arrangement for 11 hexagons.
        # A hexagonal number pattern: 1, 7, 19... so 11 is between 7 and 19.
        # Let's do a 3-4-3 arrangement or similar.
        
        return positions
    
    # Let me create a better initial arrangement
    def make_initial_positions():
        """Create initial positions for N hexagons in a compact honeycomb-like arrangement."""
        positions = []
        
        # For N=11, let's use a arrangement that's roughly hexagonal
        # Row-based: 
        # Row 0: 3 hexagons
        # Row 1: 4 hexagons  
        # Row 2: 3 hexagons
        # Total: 10... need 11
        
        # Let's try:
        # Row 0: 4
        # Row 1: 4
        # Row 2: 3
        # Total: 11
        
        # Honeycomb vertical spacing = sqrt(3)/2 * 2 = sqrt(3) (for side-to-side touching)
        # Horizontal spacing = 1.5 for staggered, but for aligned rows it's 2
        
        # Actually, for regular hexagons in a grid:
        # If aligned (same orientation), horizontal spacing = 2 (circumradius*2), 
        # vertical spacing = sqrt(3) (apothem*2)
        # For staggered honeycomb, it's more complex.
        
        # Let's just use a simple grid with spacing sqrt(3) in both directions for now
        # and let the optimizer figure out the best arrangement.
        
        # Start with a compact cluster
        # Use a hexagonal lattice with basis vectors
        a1 = np.array([np.sqrt(3), 0])
        a2 = np.array([np.sqrt(3)/2, 3/2])  # Wait, this isn't right for hexagons
        
        # For unit hexagons (side 1, circumradius 1), the lattice vectors for honeycomb are:
        # v1 = (sqrt(3), 0) - horizontal neighbor
        # v2 = (sqrt(3)/2, 3/2) - diagonal neighbor? No...
        
        # Let me just use distance sqrt(3) between centers in a triangular lattice
        # Triangular lattice with spacing d = sqrt(3):
        # Basis: (d, 0) and (d/2, d*sqrt(3)/2) = (sqrt(3)/2, 3/2)
        
        d = np.sqrt(3)
        basis1 = np.array([d, 0.0])
        basis2 = np.array([d/2, d * np.sqrt(3)/2])
        
        # Generate points in a hexagonal cluster
        # For 11 points, use a 3x3 grid minus corners, or similar
        coords = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                pos = i * basis1 + j * basis2
                coords.append(pos)
        
        # Sort by distance from origin and take the closest 11
        coords = np.array(coords)
        dists = np.linalg.norm(coords, axis=1)
        order = np.argsort(dists)
        selected = coords[order[:N]]
        
        for pos in selected:
            positions.append([pos[0], pos[1], 0.0])
        
        return positions
    
    positions = make_initial_positions()
    
    # Build initial parameter vector
    init_params = []
    for p in positions:
        init_params.extend([p[0], p[1], p[2]])
    init_params.extend([0.0, 0.0, 10.0, 0.0])  # outer center, side length, outer rotation
    
    # Optimize using Nelder-Mead or L-BFGS-B
    # Let's try multiple starting points and local optimizations
    
    best_params = None
    best_score = np.inf
    
    # Try a few different initial configurations
    initial_configs = [
        make_initial_positions(),
    ]
    
    # Add a linear arrangement as another starting point
    linear_pos = []
    for k in range(N):
        linear_pos.append([2.0 * k - N, 0.0, 0.0])
    initial_configs.append(linear_pos)
    
    # Add a random perturbation of the first config
    random_pos = []
    for p in make_initial_positions():
        random_pos.append([p[0] + np.random.uniform(-0.5, 0.5), 
                          p[1] + np.random.uniform(-0.5, 0.5), 
                          np.random.uniform(-0.1, 0.1)])
    initial_configs.append(random_pos)
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        params = []
        for p in config:
            params.extend([p[0], p[1], p[2]])
        params.extend([0.0, 0.0, 10.0, 0.0])
        
        # Use Nelder-Mead for local optimization
        try:
            result = minimize(objective, params, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # If we have a good result, extract it
    if best_params is not None:
        n = N
        centers = best_params[:3*n].reshape(n, 3)
        outer_center = best_params[3*n:3*n+2]
        S = best_params[3*n+2]
        phi_outer = best_params[3*n+3]
        
        # Convert to output format
        inner = []
        for i in range(n):
            inner.append([float(centers[i][0]), float(centers[i][1]), float(np.degrees(centers[i][2]))])
        
        return inner, (float(outer_center[0]), float(outer_center[1])), float(S), float(np.degrees(phi_outer))
    else:
        # Fallback: return the initial linear arrangement with a large outer hexagon
        inner = [[2.0 * k - N, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 50.0, 0.0

# EVOLVE-BLOCK-END
