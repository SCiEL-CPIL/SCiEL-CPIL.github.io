"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Unit hexagon vertices (flat-top, side length 1)
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.pi/6 + np.arange(6) * np.pi/3
        return np.column_stack([cx + side * np.cos(angles), 
                                cy + side * np.sin(angles)])
    
    # Check if a point is inside a regular hexagon (center, side, rotation)
    def point_in_hex(px, py, cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        # Transform to hexagon's local frame
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        # Hexagon with flat-top orientation in local frame
        # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
        # For a regular hexagon with side s, the apothem is s*sqrt(3)/2
        # The hexagon can be defined by 6 half-planes
        apothem = side * np.sqrt(3) / 2
        for i in range(6):
            theta = np.pi/3 * i + np.pi/6
            # Normal direction
            nx = np.cos(theta)
            ny = np.sin(theta)
            # Distance from center to edge
            if lx * nx + ly * ny > apothem:
                return False
        return True
    
    # Compute the minimal side length of a hexagon (center, rotation) that contains all inner hexagons
    def compute_outer_side(inner_positions, outer_center, outer_rotation):
        cx, cy = outer_center
        angle_rad = np.radians(outer_rotation)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        
        max_dist = 0
        for (px, py, pa) in inner_positions:
            # For each inner hexagon, find the max distance from outer center
            # to any vertex of the inner hexagon, in the outer hexagon's frame
            verts = hex_vertices(px, py, pa)
            for vx, vy in verts:
                # Transform to outer hexagon's local frame
                dx = vx - cx
                dy = vy - cy
                lx = dx * cos_a - dy * sin_a
                ly = dx * sin_a + dy * cos_a
                # For a regular hexagon, the required side length is determined by
                # the maximum of |lx|/cos(30*k*pi) and |ly|/sin(30*k*pi) over the 6 sectors
                # Actually, for a regular hexagon with side s and apothem a = s*sqrt(3)/2,
                # a point (lx, ly) is inside if for all 6 edges, the projection is <= apothem
                # The required apothem is the max over the 6 edge normals of the projection
                for i in range(6):
                    theta = np.pi/3 * i + np.pi/6
                    nx = np.cos(theta)
                    ny = np.sin(theta)
                    proj = abs(lx * nx + ly * ny)
                    # Required apothem >= proj
                    # side = 2 * apothem / sqrt(3)
                    req_side = 2 * proj / np.sqrt(3)
                    if req_side > max_dist:
                        max_dist = req_side
        return max_dist
    
    # Initial arrangement: hexagonal lattice
    # Place hexagons on a triangular lattice with spacing 2
    # Arrange in a compact hexagonal cluster
    def initial_arrangement():
        positions = []
        # Use axial coordinates for hex grid
        # Center at (0,0)
        # Ring 0: 1 hex
        # Ring 1: 6 hexes
        # Ring 2: 12 hexes, but we only need 5 more
        # Total: 1 + 6 + 5 = 12
        
        # Axial to pixel (for pointy-top hex, but we'll use flat-top)
        # For flat-top hex with side 1, spacing between centers = 2
        # Axial coords (q, r) to pixel:
        # x = 2 * q + 1 * r (for flat-top, but let me verify)
        # Actually for flat-top: x = 2*q + r, y = sqrt(3)*r (approximately)
        # Let me use a simpler approach: just place them manually
        
        # Center
        positions.append((0.0, 0.0, 0.0))
        
        # Ring 1: 6 hexagons at distance 2 from center
        for i in range(6):
            angle = np.pi/3 * i
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            positions.append((x, y, 0.0))
        
        # Ring 2: need 5 more, place them at distance ~4 from center
        # The 12 positions in ring 2 are at angles 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
        # with distances 4 for the 6 axis-aligned and 2*sqrt(3) for the 6 in-between
        # Let me place 5 of the 12 ring-2 positions
        # To be symmetric, pick every other one or a contiguous arc
        ring2_angles = [0, 60, 120, 180, 240, 300]  # axis-aligned, distance 4
        ring2_angles_mid = [30, 90, 150, 210, 270, 330]  # in-between, distance 2*sqrt(3)
        
        # Add 5 from ring 2 - pick a contiguous arc for compactness
        # Let's add the 5 that are closest to make a compact shape
        # Add ring2 at angles 0, 60, 120, 180, 240 (5 of the 6 axis-aligned)
        for i in range(5):
            angle = np.pi/3 * i
            x = 4.0 * np.cos(angle)
            y = 4.0 * np.sin(angle)
            positions.append((x, y, 0.0))
        
        return positions
    
    positions = initial_arrangement()
    
    # Compute initial outer side
    outer_center = (0.0, 0.0)
    outer_rotation = 0.0
    current_side = compute_outer_side(positions, outer_center, outer_rotation)
    
    best_positions = [list(p) for p in positions]
    best_center = list(outer_center)
    best_rotation = outer_rotation
    best_side = current_side
    
    # Optimization: adjust positions to minimize outer side
    # Variables: positions (3*N) + center (2) + rotation (1) = 3*12 + 2 + 1 = 39
    def objective(params):
        if time.time() > deadline:
            return best_side
        pos = params[:3*N].reshape(N, 3)
        ocx, ocy = params[3*N], params[3*N+1]
        orot = params[3*N+2]
        
        # Check no overlaps (simple check using center distances)
        # For unit hexagons, minimum center distance is 2 if same orientation
        # With rotations, it's more complex, but let's use a soft penalty
        penalty = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1.99:  # Allow slight tolerance
                    penalty += (1.99 - dist) ** 2
        
        side = compute_outer_side([tuple(p) for p in pos], (ocx, ocy), orot)
        return side + 10 * penalty
    
    # Initial parameters
    init_params = []
    for p in positions:
        init_params.extend([p[0], p[1], p[2]])
    init_params.extend([outer_center[0], outer_center[1], outer_rotation])
    
    # Optimize
    result = None
    try:
        result = minimize(objective, init_params, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    except Exception:
        pass
    
    if result is not None and result.fun < best_side:
        pos = result.x[:3*N].reshape(N, 3)
        ocx, ocy = result.x[3*N], result.x[3*N+1]
        orot = result.x[3*N+2]
        best_positions = [list(p) for p in pos]
        best_center = [ocx, ocy]
        best_rotation = orot
        best_side = compute_outer_side([tuple(p) for p in pos], (ocx, ocy), orot)
    
    # Refine with tighter optimization
    if time.time() < deadline - 30:
        try:
            result2 = minimize(objective, np.array([p[0] for p in best_positions] + [p[1] for p in best_positions] + [p[2] for p in best_positions] + best_center + [best_rotation]), 
                              method='Nelder-Mead', 
                              options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
            if result2.fun < best_side:
                pos = result2.x[:3*N].reshape(N, 3)
                ocx, ocy = result2.x[3*N], result2.x[3*N+1]
                orot = result2.x[3*N+2]
                best_positions = [list(p) for p in pos]
                best_center = [ocx, ocy]
                best_rotation = orot
                best_side = compute_outer_side([tuple(p) for p in pos], (ocx, ocy), orot)
        except Exception:
            pass
    
    # Final result
    inner = [[p[0], p[1], p[2]] for p in best_positions]
    return inner, (best_center[0], best_center[1]), best_side, best_rotation
# EVOLVE-BLOCK-END

