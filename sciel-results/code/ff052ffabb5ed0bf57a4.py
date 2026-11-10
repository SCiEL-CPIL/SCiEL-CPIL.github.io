# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

def hexagon_packing_12():
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 15.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angles = np.deg2rad(angle_deg + np.arange(6) * 60)
        return cx + side * np.cos(angles), cy + side * np.sin(angles)

    def compute_s(pos, angle):
        """Compute the minimum side length s of a regular hexagon centered at origin
        with angle 0 that contains all vertices of the 12 inner hexagons."""
        # For a regular hexagon centered at origin with side s and angle 0:
        # Vertices at s*(cos(k*60), sin(k*60)) for k=0..5
        # The constraint for a point (px, py) to be inside is:
        # For each edge, the point must be on the correct side.
        # Edge i goes from vertex i to vertex i+1.
        # The inward normal for edge i (from vertex i to i+1) points toward the center.
        
        # For a hexagon with vertices at angles 0, 60, 120, 180, 240, 300:
        # Edge 0: from (s,0) to (s/2, s*sqrt(3)/2). Direction: (-s/2, s*sqrt(3)/2).
        # Normal (inward): rotate direction by -90 deg: (s*sqrt(3)/2, s/2) -> normalized: (sqrt(3)/2, 1/2)
        # Actually, let's use the standard support function approach.
        
        # A regular hexagon with side s and angle 0 has apothem a = s * sqrt(3)/2.
        # The distance from center to each edge is the apothem.
        # The 6 edge normals point in directions: 30, 90, 150, 210, 270, 330 degrees.
        # For a point (px, py) to be inside, we need:
        # px * cos(theta_i) + py * sin(theta_i) <= a for all i, where theta_i are the outward normals.
        
        # Outward normals for hexagon with angle 0 (vertices at 0, 60, ...):
        # Edge between vertex 5 (300 deg) and vertex 0 (0 deg): midpoint direction 330 deg. Outward normal: 330 deg.
        # Edge between vertex 0 (0 deg) and vertex 1 (60 deg): midpoint direction 30 deg. Outward normal: 30 deg.
        # Edge between vertex 1 (60 deg) and vertex 2 (120 deg): midpoint direction 90 deg. Outward normal: 90 deg.
        # Edge between vertex 2 (120 deg) and vertex 3 (180 deg): midpoint direction 150 deg. Outward normal: 150 deg.
        # Edge between vertex 3 (180 deg) and vertex 4 (240 deg): midpoint direction 210 deg. Outward normal: 210 deg.
        # Edge between vertex 4 (240 deg) and vertex 5 (300 deg): midpoint direction 270 deg. Outward normal: 270 deg.
        
        normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # For each inner hexagon vertex (vx, vy), the constraint is:
        # normals @ [vx, vy] <= s * sqrt(3)/2
        # So s >= 2/sqrt(3) * max(normals @ [vx, vy])
        
        sq3 = np.sqrt(3)
        max_proj = 0.0
        
        for i in range(12):
            vx, vy = get_hex_vertices(pos[i, 0], pos[i, 1], 1.0, angle)
            # Project all 6 vertices onto the 6 normals
            projections = normals @ np.column_stack([vx, vy]).T
            max_proj = max(max_proj, np.max(projections))
        
        s = 2.0 / sq3 * max_proj
        return s

    def check_separation(pos, angle, tol=1e-8):
        """Check if all inner hexagons are separated. Returns True if valid."""
        verts = []
        for i in range(12):
            vx, vy = get_hex_vertices(pos[i, 0], pos[i, 1], 1.0, angle)
            verts.append((vx, vy))
        
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_separated(verts[i], verts[j], tol):
                    return False
        return True

    def are_separated(a_x, a_y, b_x, b_y, tol=1e-8):
        """SAT check: returns True if polygons are separated (no overlap)."""
        # Collect all edge normals from both polygons
        axes = []
        for i in range(6):
            ex = a_x[(i+1)%6] - a_x[i]
            ey = a_y[(i+1)%6] - a_y[i]
            axes.append((-ey, ex))
        for i in range(6):
            ex = b_x[(i+1)%6] - b_x[i]
            ey = b_y[(i+1)%6] - b_y[i]
            axes.append((-ey, ex))
        
        for nx, ny in axes:
            norm = np.sqrt(nx*nx + ny*ny)
            if norm < 1e-12:
                continue
            nx, ny = nx/norm, ny/norm
            
            pa = a_x * nx + a_y * ny
            pb = b_x * nx + b_y * ny
            
            pa_min, pa_max = np.min(pa), np.max(pa)
            pb_min, pb_max = np.min(pb), np.max(pb)
            
            # Check for separation
            if pa_max < pb_min - tol or pb_max < pa_min - tol:
                return True  # Separated on this axis
        
        return False  # Not separated on any axis -> overlapping

    def objective(params):
        """Objective: minimize s. params = [x1,y1,...,x12,y12, angle]"""
        elapsed = time.time() - start_time
        if elapsed > time_limit - safety_margin:
            return 1e10  # Timeout
        
        pos = params[:24].reshape(12, 2)
        angle = params[24]
        
        # Quick overlap check
        if not check_separation(pos, angle):
            return 1e10
        
        s = compute_s(pos, angle)
        
        # Check if all vertices are inside the outer hexagon
        sq3 = np.sqrt(3)
        normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        apothem = s * sq3 / 2
        
        for i in range(12):
            vx, vy = get_hex_vertices(pos[i, 0], pos[i, 1], 1.0, angle)
            projections = normals @ np.column_stack([vx, vy]).T
            if np.any(projections > apothem + 1e-6):
                return 1e10
        
        return s

    def objective_3p(params):
        """Objective for 3-parameter affine transformation of honeycomb.
        params = [scale_x, scale_y, rotation_deg]
        We apply a linear transformation to the base honeycomb positions,
        then optimize the angle of the hexagons separately."""
        elapsed = time.time() - start_time
        if elapsed > time_limit - safety_margin:
            return 1e10
        
        sx, sy, rot_deg = params
        rot = np.deg2rad(rot_deg)
        
        # Base honeycomb positions (12 hexagons)
        # Center: (0, 0)
        # Ring 1: 6 hexagons at distance sqrt(3)
        # Ring 2: 5 hexagons (to make 12 total: 1+6+5=12)
        
        # Let's use a specific arrangement:
        # Position 0: center (0, 0)
        # Positions 1-6: ring at distance d1 = sqrt(3), angles 0, 60, 120, 180, 240, 300
        # Positions 7-11: ring at distance d2, specific angles
        
        # Actually, for 12 hexagons, a common packing is:
        # 1 center + 6 in first ring + 5 in second ring (one missing)
        # Or a 2-3-4-3 arrangement in rows.
        
        # Let's use axial coordinates for a honeycomb:
        # Axial (q, r) -> Cartesian: x = sqrt(3)*(q + r/2), y = 3*r/2
        # For unit hexagons, adjacent centers are at distance sqrt(3).
        
        # 12 hexagons in a hexagonal shape:
        # Center (0,0)
        # 6 neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # That's 7. Need 5 more for 12.
        # Next ring: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
        # That's 11 more. Total would be 18.
        
        # For 12, let's take center + 6 + 5 from the next ring.
        # Choose 5 from the 11 in the next ring for symmetry.
        
        # Let's use a different approach: just define 12 positions directly.
        # Honeycomb with 12 cells:
        # Row 0: 3 hexes at y = 0, x = -sqrt(3), 0, sqrt(3)
        # Row 1: 3 hexes at y = 3/2, x = -3*sqrt(3)/2, -sqrt(3)/2, sqrt(3)/2
        # Row 2: 3 hexes at y = 3, x = -sqrt(3), 0, sqrt(3)
        # Row 3: 3 hexes at y = 9/2, x = -3*sqrt(3)/2, -sqrt(3)/2, sqrt(3)/2
        # Total: 12. This is a parallelogram-shaped cluster.
        
        # Actually, let's use the 1-6-5 arrangement.
        # Center: (0, 0)
        # Ring 1 (6): at distance sqrt(3), angles 0, 60, 120, 180, 240, 300
        # Ring 2 (5): at distance 2*sqrt(3), angles 30, 90, 150, 210, 270 (every other one)
        
        d1 = np.sqrt(3)
        d2 = 2 * np.sqrt(3)
        
        base_positions = np.zeros((12, 2))
        base_positions[0] = [0, 0]
        for k in range(6):
            ang = np.deg2rad(k * 60)
            base_positions[k+1] = [d1 * np.cos(ang), d1 * np.sin(ang)]
        for k in range(5):
            ang = np.deg2rad(30 + k * 72)  # 5 points evenly spaced? No, let's use specific angles.
            # Actually, for a hexagonal lattice, the next ring has 12 positions.
            # Let's pick 5 that are symmetric.
            # Angles: 0, 72, 144, 216, 288? That's not hexagonal.
            # Let's just use a simple grid for now.
            pass
        
        # Let's use a simpler base configuration: 12 hexagons in a 3x4 grid pattern.
        # Or better, use the known optimal-ish configuration.
        
        # For now, let's just use a fixed set of 12 positions and optimize the affine transform.
        # Positions for 12 hexagons in a compact arrangement:
        # Using axial coordinates (q, r):
        # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -- 7
        # (2,0), (1,1), (0,2), (-1,2), (-2,1) -- 5 more, total 12
        
        axial = np.array([
            [0, 0],
            [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
            [2, 0], [1, 1], [0, 2], [-1, 2], [-2, 1]
        ], dtype=float)
        
        # Convert axial to Cartesian
        # x = sqrt(3) * (q + r/2)
        # y = 3/2 * r
        q = axial[:, 0]
        r = axial[:, 1]
        base_x = np.sqrt(3) * (q + r / 2)
        base_y = 1.5 * r
        
        # Apply affine transformation: scale and rotate
        # First scale, then rotate
        x_scaled = base_x * sx
        y_scaled = base_y * sy
        
        # Rotate
        x_rot = x_scaled * np.cos(rot) - y_scaled * np.sin(rot)
        y_rot = x_scaled * np.sin(rot) + y_scaled * np.cos(rot)
        
        pos = np.column_stack([x_rot, y_rot])
        angle = 0.0  # We'll optimize angle separately or fix it
        
        # For now, fix angle to 0 and just optimize the affine transform
        if not check_separation(pos, angle):
            return 1e10
        
        s = compute_s(pos, angle)
        return s

    # Initial guess: use the base honeycomb configuration
    axial = np.array([
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [1, 1], [0, 2], [-1, 2], [-2, 1]
    ], dtype=float)
    q = axial[:, 0]
    r = axial[:, 1]
    base_x = np.sqrt(3) * (q + r / 2)
    base_y = 1.5 * r
    
    pos0 = np.column_stack([base_x, base_y])
    angle0 = 0.0
    
    x0 = np.zeros(25)
    x0[:24] = pos0.flatten()
    x0[24] = angle0
    
    # First, try Nelder-Mead on the full 25D problem
    best_s = compute_s(pos0, angle0)
    best_params = x0.copy()
    
    # Check if initial is valid
    if check_separation(pos0, angle0):
        best_s = compute_s(pos0, angle0)
    
    # Run Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.fun < best_s:
            best_s = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Also try differential evolution for a global search
    try:
        elapsed = time.time() - start_time
        if elapsed < time_limit - safety_margin - 60:
            bounds = [(-5, 5)] * 24 + [(-90, 90)]
            de_result = differential_evolution(objective, bounds, maxiter=200, popsize=15,
                                               tol=1e-4, seed=42, polish=False)
            if de_result.fun < best_s:
                best_s = de_result.fun
                best_params = de_result.x
    except Exception:
        pass
    
    # Refine with Nelder-Mead from the DE result
    try:
        elapsed = time.time() - start_time
        if elapsed < time_limit - safety_margin - 30:
            result2 = minimize(objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
            if result2.fun < best_s:
                best_s = result2.fun
                best_params = result2.x
    except Exception:
        pass
    
    # Final refinement
    try:
        elapsed = time.time() - start_time
        if elapsed < time_limit - safety_margin - 10:
            result3 = minimize(objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if result3.fun < best_s:
                best_s = result3.fun
                best_params = result3.x
    except Exception:
        pass
    
    # Extract final parameters
    pos = best_params[:24].reshape(12, 2)
    angle = best_params[24]
    
    # Verify validity
    if not check_separation(pos, angle):
        # Fall back to initial
        pos = pos0
        angle = angle0
        best_s = compute_s(pos, angle)
    
    s = compute_s(pos, angle)
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i] = [pos[i, 0], pos[i, 1], angle]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
