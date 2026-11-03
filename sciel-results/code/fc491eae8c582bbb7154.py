import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # ===================================================================
    # STRUCTURAL CHANGE: Phase-field variational approach
    # Represent hexagons as smooth indicator functions on a grid,
    # minimize container size + overlap penalty via gradient descent
    # ===================================================================
    
    # Initial configuration: hexagonal lattice arrangement
    # This is a good starting point for the phase-field optimization
    def initial_hexagonal_arrangement(n=11):
        """Generate initial positions in a hexagonal pattern."""
        positions = []
        angles = [0.0] * n
        
        # Center
        positions.append((0.0, 0.0))
        
        # First ring: 6 hexagons
        for k in range(6):
            ang = k * math.pi / 3.0
            x = math.sqrt(3) * math.cos(ang)
            y = 1.5 * math.sin(ang)
            positions.append((x, y))
        
        # Second ring: 4 more to reach 11
        for k in range(4):
            ang = k * math.pi / 2.0 + math.pi / 6.0
            x = 2 * math.sqrt(3) * math.cos(ang)
            y = 3.0 * math.sin(ang)
            positions.append((x, y))
        
        return positions, angles
    
    # Phase-field parameters
    grid_size = 120  # Coarse grid for speed
    domain_size = 12.0  # Physical domain size
    dx = domain_size / grid_size
    sigma = 0.15  # Interface width parameter
    lambda_overlap = 5.0  # Overlap penalty weight
    lambda_area = 1.0  # Area penalty weight
    
    # Create grid
    x = np.linspace(-domain_size/2, domain_size/2, grid_size)
    y = np.linspace(-domain_size/2, domain_size/2, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Phase-field representation: u[i,j,k] where k is hexagon index
    # u > 0.5 means inside hexagon k, u < 0.5 means outside
    
    def compute_hexagon_indicator(cx, cy, angle, grid_X, grid_Y, side=1.0):
        """
        Compute smooth indicator function for a regular hexagon.
        Uses distance to hexagon boundary with smooth transition.
        """
        # Rotate coordinates to align hexagon with axes
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        xr = (grid_X - cx) * cos_a + (grid_Y - cy) * sin_a
        yr = -(grid_X - cx) * sin_a + (grid_Y - cy) * cos_a
        
        # Hexagon defined by |xr| <= side, |yr| <= side*sqrt(3)/2, 
        # and |xr|/side + |yr|/(side*sqrt(3)/2) <= 2
        # Use a smooth approximation of the hexagon boundary
        
        # Distance-like function: negative inside, positive outside
        # For a regular hexagon with flat top, the boundary is:
        # max(|xr|, |yr|/sqrt(3), (|xr| + |yr|)/2) <= side
        
        abs_xr = np.abs(xr)
        abs_yr = np.abs(yr)
        
        # Three constraints for hexagon (flat-top orientation)
        d1 = abs_xr - side
        d2 = abs_yr - side * math.sqrt(3)
        d3 = (abs_xr + abs_yr) / math.sqrt(3) - side
        
        # Take the maximum (most violated constraint)
        d = np.maximum(d1, np.maximum(d2, d3))
        
        # Smooth transition using tanh
        u = 0.5 * (1.0 - np.tanh(d / sigma))
        
        return u
    
    def compute_enclosing_hexagon_size(positions, angles, center, rotation):
        """
        Compute the minimum side length of a regular hexagon that encloses
        all inner hexagons, given their centers, orientations, and the 
        container's center and rotation.
        """
        if len(positions) == 0:
            return 0.0, center, rotation
        
        # Convert positions to numpy array
        pos = np.array(positions)
        
        # The container hexagon has 6 edges with normals at angles:
        # rotation + pi/6 + k*pi/3 for k=0..5
        max_proj = 0.0
        best_center = np.array(center)
        
        # For each orientation, compute the required half-width
        for k in range(6):
            ang = rotation + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            
            # Project all hexagon centers onto this normal
            # But we need to account for the hexagon's extent in this direction
            # For a unit hexagon rotated by angle theta, the support function
            # in direction (nx, ny) is:
            # h = max over vertices of (v . n)
            # For simplicity, approximate with the apothem plus center projection
            
            # More accurate: compute the actual support function
            # A regular hexagon with side 1 has vertices at distance 1 from center
            # at angles 0, 60, 120, 180, 240, 300 degrees (for flat-top)
            
            proj_centers = pos[:, 0] * nx + pos[:, 1] * ny
            
            # For each hexagon, compute its support in direction n
            # The support function of a regular hexagon (side 1, flat-top) 
            # rotated by theta in direction n:
            # h(n) = max_{j=0..5} (v_j . n) where v_j are vertices
            
            max_support = 0.0
            for i, (px, py) in enumerate(pos):
                theta = angles[i]
                # Vertices of unit hexagon (flat-top) relative to center
                verts = []
                for j in range(6):
                    va = theta + j * math.pi / 3.0
                    vx = math.cos(va)
                    vy = math.sin(va)
                    verts.append((vx, vy))
                
                # Rotate and translate vertices
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                
                # Actually, the vertices are already at angle theta + j*pi/3
                # so we just need to project them
                vertex_projs = []
                for j in range(6):
                    va = theta + j * math.pi / 3.0
                    vx = px + math.cos(va)
                    vy = py + math.sin(va)
                    vp = vx * nx + vy * ny
                    vertex_projs.append(vp)
                
                hex_support = max(vertex_projs)
                max_support = max(max_support, hex_support)
            
            # The container must satisfy: max_support - min_support <= 2 * D
            # where D is the apothem of the container
            # min_support is the minimum over all hexagon vertices
            
            min_support = float('inf')
            for i, (px, py) in enumerate(pos):
                theta = angles[i]
                vertex_projs = []
                for j in range(6):
                    va = theta + j * math.pi / 3.0
                    vx = px + math.cos(va)
                    vy = py + math.sin(va)
                    vp = vx * nx + vy * ny
                    vertex_projs.append(vp)
                
                hex_min = min(vertex_projs)
                min_support = min(min_support, hex_min)
            
            width = max_support - min_support
            half_width = width / 2.0
            max_proj = max(max_proj, half_width)
        
        # The container's apothem must be at least max_proj
        # D = max_proj, so S = (D + APOTHEM) / APOTHEM
        # But we also need to account for the container center
        # The center should be at the midpoint of the bounding box
        
        # Compute the optimal center as the average of min and max projections
        cx_sum = 0.0
        cy_sum = 0.0
        for k in range(6):
            ang = rotation + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            
            max_support = 0.0
            min_support = float('inf')
            for i, (px, py) in enumerate(pos):
                theta = angles[i]
                vertex_projs = []
                for j in range(6):
                    va = theta + j * math.pi / 3.0
                    vx = px + math.cos(va)
                    vy = py + math.sin(va)
                    vp = vx * nx + vy * ny
                    vertex_projs.append(vp)
                max_support = max(max_support, max(vertex_projs))
                min_support = min(min_support, min(vertex_projs))
            
            mid = (max_support + min_support) / 2.0
            cx_sum += mid * nx
            cy_sum += mid * ny
        
        best_center = [cx_sum / 6.0, cy_sum / 6.0]
        
        # Recompute max_proj with the new center
        max_proj_centered = 0.0
        for k in range(6):
            ang = rotation + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            
            max_support = 0.0
            min_support = float('inf')
            for i, (px, py) in enumerate(pos):
                theta = angles[i]
                vertex_projs = []
                for j in range(6):
                    va = theta + j * math.pi / 3.0
                    vx = px + math.cos(va)
                    vy = py + math.sin(va)
                    vp = vx * nx + vy * ny
                    vertex_projs.append(vp)
                max_support = max(max_support, max(vertex_projs))
                min_support = min(min_support, min(vertex_projs))
            
            # Center the projections
            centered_max = max_support - best_center[0] * nx - best_center[1] * ny
            centered_min = min_support - best_center[0] * nx - best_center[1] * ny
            half_width = max(abs(centered_max), abs(centered_min))
            max_proj_centered = max(max_proj_centered, half_width)
        
        D = max_proj_centered
        S = (D + APOTHEM) / APOTHEM
        return S, best_center, rotation
    
    # Phase-field optimization
    def phase_field_optimize(positions, angles, max_iterations=5000):
        """
        Optimize the packing using phase-field gradient descent.
        positions: list of (x, y)
        angles: list of rotation angles
        """
        n = len(positions)
        pos = np.array(positions, dtype=float)
        ang = np.array(angles, dtype=float)
        
        # Parameters to optimize: positions (2n) and angles (n)
        # Total: 3n parameters
        param_count = 3 * n
        
        def unpack(params):
            pos = params[:2*n].reshape(n, 2)
            ang = params[2*n:]
            return pos, ang
        
        def objective(params):
            """
            Objective: container size + overlap penalty
            """
            pos, ang = unpack(params)
            
            # Compute container size
            center = [np.mean(pos[:, 0]), np.mean(pos[:, 1])]
            S, _, _ = compute_enclosing_hexagon_size(
                [tuple(p) for p in pos], 
                [float(a) for a in ang], 
                center, 
                0.0  # Container rotation fixed at 0 for simplicity
            )
            
            # Compute overlap penalty using phase field
            # For each pair of hexagons, compute overlap integral
            overlap_energy = 0.0
            
            # Use a simplified overlap check based on distance
            # Two unit hexagons overlap if their centers are too close
            # The minimum distance for no overlap is 2 * apothem = sqrt(3)
            # But this depends on orientation
            
            for i in range(n):
                for j in range(i+1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    # Approximate minimum distance for non-overlap
                    # This is a rough approximation; the actual value depends on orientation
                    min_dist = math.sqrt(3) * 0.95  # Slightly less than sqrt(3) to allow tight packing
                    
                    if dist < min_dist:
                        # Penalty increases as distance decreases
                        penalty = (min_dist - dist) ** 2
                        overlap_energy += penalty
            
            # Total objective
            total = lambda_area * S + lambda_overlap * overlap_energy
            return total
        
        # Initial parameters
        initial_params = np.zeros(param_count)
        initial_params[:2*n] = pos.flatten()
        initial_params[2*n:] = ang
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                options={
                    'maxiter': max_iterations,
                    'ftol': 1e-10,
                    'gtol': 1e-8
                }
            )
            
            if result.success or result.fun < objective(initial_params):
                pos_opt, ang_opt = unpack(result.x)
                center = [np.mean(pos_opt[:, 0]), np.mean(pos_opt[:, 1])]
                S, _, _ = compute_enclosing_hexagon_size(
                    [tuple(p) for p in pos_opt],
                    [float(a) for a in ang_opt],
                    center,
                    0.0
                )
                return [tuple(p) for p in pos_opt], [float(a) for a in ang_opt], S
        except Exception as e:
            pass
        
        # Fallback: return initial configuration
        center = [np.mean(pos[:, 0]), np.mean(pos[:, 1])]
        S, _, _ = compute_enclosing_hexagon_size(
            [tuple(p) for p in pos],
            [float(a) for a in ang],
            center,
            0.0
        )
        return positions, angles, S
    
    # Run phase-field optimization
    positions, angles, S = phase_field_optimize(*initial_hexagonal_arrangement(N))
    
    # Final refinement with Nelder-Mead on the best configuration
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            def refine_objective(params):
                n = len(params) // 3
                pos = params[:2*n].reshape(n, 2)
                ang = params[2*n:]
                
                center = [np.mean(pos[:, 0]), np.mean(pos[:, 1])]
                S, _, _ = compute_enclosing_hexagon_size(
                    [tuple(p) for p in pos],
                    [float(a) for a in ang],
                    center,
                    0.0
                )
                
                # Overlap penalty
                overlap = 0.0
                for i in range(n):
                    for j in range(i+1, n):
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        min_dist = math.sqrt(3) * 0.95
                        if dist < min_dist:
                            overlap += (min_dist - dist) ** 2
                
                return S + 5.0 * overlap
            
            init_params = np.zeros(3 * N)
            init_params[:2*N] = np.array(positions).flatten()
            init_params[2*N:] = angles
            
            result = minimize(
                refine_objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if result.fun < S:
                pos_final = result.x[:2*N].reshape(N, 2)
                ang_final = result.x[2*N:]
                positions = [tuple(p) for p in pos_final]
                angles = [float(a) for a in ang_final]
                center = [np.mean(pos_final[:, 0]), np.mean(pos_final[:, 1])]
                S, _, _ = compute_enclosing_hexagon_size(
                    [tuple(p) for p in pos_final],
                    [float(a) for a in ang_final],
                    center,
                    0.0
                )
        except Exception:
            pass
    
    # Prepare output
    center = [float(np.mean([p[0] for p in positions])), 
              float(np.mean([p[1] for p in positions]))]
    
    # Compute final container parameters
    S_final, center_final, rotation_final = compute_enclosing_hexagon_size(
        positions,
        angles,
        center,
        0.0
    )
    
    # Apply a small scale factor to ensure no overlap
    SCALE = 1.0 + 1e-10
    scaled_positions = []
    for (px, py), a in zip(positions, angles):
        sx = center_final[0] + (px - center_final[0]) * SCALE
        sy = center_final[1] + (py - center_final[1]) * SCALE
        scaled_positions.append((sx, sy))
    
    # Recompute S with scaled positions
    S_scaled, center_scaled, rot_scaled = compute_enclosing_hexagon_size(
        scaled_positions,
        angles,
        center_final,
        0.0
    )
    
    inner = []
    for (px, py), a in zip(scaled_positions, angles):
        inner.append([float(px), float(py), float(math.degrees(a))])
    
    return inner, [float(center_scaled[0]), float(center_scaled[1])], float(S_scaled), float(math.degrees(rot_scaled))
