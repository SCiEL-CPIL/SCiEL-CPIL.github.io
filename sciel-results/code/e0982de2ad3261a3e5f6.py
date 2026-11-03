import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130.0  # safety margin
    
    # Unit hexagon vertices (flat-top orientation, side 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees, radius 1
    hex_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    unit_hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    def get_hex_verts(center, angle_deg, side=1.0):
        """Get vertices of a regular hexagon with given center, rotation, side length."""
        theta = np.radians(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return center + R @ (side * unit_hex_verts).T
    
    def hexagon_contains_point(px, py, center, side, angle_deg):
        """Check if point (px,py) is inside regular hexagon with given center, side, rotation."""
        theta = np.radians(angle_deg)
        # Transform point to hexagon's local frame
        dx, dy = px - center[0], py - center[1]
        R_inv = np.array([[np.cos(-theta), -np.sin(-theta)],
                          [np.sin(-theta), np.cos(-theta)]])
        lx, ly = R_inv @ np.array([dx, dy])
        # In local frame, hexagon with side s has vertices at (s,0), (s/2, s*sqrt(3)/2), etc.
        # The hexagon is defined by |x| <= s, |y| <= s*sqrt(3)/2, and |x| + |y|/sqrt(3) <= s
        # More precisely, for a flat-top hexagon: 
        # The 6 edges are at angles 30, 90, 150, 210, 270, 330 degrees from center
        # Each edge is at distance s*sqrt(3)/2 from center
        s = side
        apothem = s * np.sqrt(3) / 2
        # The 6 normals are at angles 30, 90, 150, 210, 270, 330
        normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
        for na in normal_angles:
            nx, ny = np.cos(na), np.sin(na)
            proj = lx * nx + ly * ny
            if proj > apothem + 1e-9:
                return False
        return True
    
    def max_violation(points, center, side, angle_deg):
        """Return max violation (0 if all inside, positive if outside)."""
        theta = np.radians(angle_deg)
        R_inv = np.array([[np.cos(-theta), -np.sin(-theta)],
                          [np.sin(-theta), np.cos(-theta)]])
        local_pts = (points - center) @ R_inv
        s = side
        apothem = s * np.sqrt(3) / 2
        normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        # For each point, for each normal, compute projection
        # Shape: (n_points, 6)
        projections = local_pts @ normals.T
        violations = projections - apothem
        return np.max(violations)
    
    def find_min_enclosing_hexagon(all_points, init_center=None, init_angle=0.0, init_side=None):
        """Find minimum side length regular hexagon enclosing all points."""
        if init_center is None:
            init_center = np.mean(all_points, axis=0)
        if init_side is None:
            # Estimate: max distance from center to any point, times some factor
            d = np.max(np.linalg.norm(all_points - init_center, axis=1))
            init_side = d * 1.2
        
        def objective(params):
            cx, cy, side, angle_deg = params
            if side <= 0:
                return 1e10
            violation = max_violation(all_points, np.array([cx, cy]), side, angle_deg)
            # We want to minimize side, but with constraint violation <= 0
            # Use penalty: side + large * max(0, violation)
            penalty = 1000.0 * max(0.0, violation)
            return side + penalty
        
        x0 = np.array([init_center[0], init_center[1], init_side, init_angle])
        bounds = [(-50, 50), (-50, 50), (0.1, 100), (-180, 180)]
        
        best_result = None
        best_side = 1e10
        best_params = x0.copy()
        
        for method in ['L-BFGS-B', 'Nelder-Mead']:
            try:
                res = minimize(objective, x0, method=method, bounds=bounds, 
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.success or res.fun < best_side:
                    cx, cy, side, angle_deg = res.x
                    if side > 0:
                        # Verify
                        v = max_violation(all_points, np.array([cx, cy]), side, angle_deg)
                        if v < 1e-6:
                            if side < best_side:
                                best_side = side
                                best_params = res.x.copy()
            except:
                pass
        
        if best_params[2] > 0:
            return best_params[2], (best_params[0], best_params[1]), best_params[3]
        return None
    
    def generate_configuration(config_type, angle_offset=0.0):
        """Generate inner hexagon positions for a given configuration type."""
        inner = []
        
        if config_type == 'flower7_plus4':
            # Central hexagon
            inner.append([0.0, 0.0, angle_offset])
            # 6 neighbors in ring
            d = np.sqrt(3)  # distance between centers of adjacent hexagons
            for k in range(6):
                ang = np.radians(60 * k + angle_offset)
                inner.append([d * np.cos(ang), d * np.sin(ang), angle_offset])
            # 4 more hexagons - place them in gaps or extend
            # Place at distance 2*d in directions 30, 150, 270, 30 (offset)
            # Actually, let's try placing them further out
            # Positions at 120-degree spacing, at distance sqrt(3) + sqrt(3) = 2*sqrt(3)
            # But that might be too far. Let's try a different approach.
            # Add 4 hexagons at the "corners" of the hexagonal arrangement
            # At angles 0, 90, 180, 270 + offset, at distance 2*sqrt(3)
            d2 = 2 * np.sqrt(3)
            for k in range(4):
                ang = np.radians(90 * k + 30 + angle_offset)
                inner.append([d2 * np.cos(ang), d2 * np.sin(ang), angle_offset])
                
        elif config_type == 'linear_chain':
            # All in a line
            d = np.sqrt(3)
            for k in range(-(N//2), N - N//2):
                inner.append([k * d, 0.0, angle_offset])
                
        elif config_type == 'hex_grid':
            # Hexagonal grid arrangement
            # For 11, try: row of 3, row of 4, row of 3, row of 1? 
            # Or 2-3-3-3-0? Let's do a compact hex-like shape
            # Hex grid coordinates
            d = np.sqrt(3)
            # Center at 0
            # Ring 0: 1 hex (center)
            # Ring 1: 6 hexes
            # That's 7. Need 4 more.
            # Add them at ring 2 positions (but only 4 of the 12)
            positions = [(0, 0)]
            for k in range(6):
                ang = np.radians(60 * k)
                positions.append((d * np.cos(ang), d * np.sin(ang)))
            # Ring 2: 12 positions, pick 4
            d2 = 2 * d
            for k in range(4):
                ang = np.radians(60 * k + 30)
                positions.append((d2 * np.cos(ang), d2 * np.sin(ang)))
            
            for (x, y) in positions:
                inner.append([x, y, angle_offset])
                
        elif config_type == 'compact_cluster':
            # Try a more compact arrangement
            d = np.sqrt(3)
            # 3 in a row, 3 above, 3 below, 2 extra
            positions = [
                (0, 0),
                (d, 0),
                (-d, 0),
                (d/2, d*np.sqrt(3)/2),
                (-d/2, d*np.sqrt(3)/2),
                (0, d*np.sqrt(3)),
                (d/2, -d*np.sqrt(3)/2),
                (-d/2, -d*np.sqrt(3)/2),
                (0, -d*np.sqrt(3)),
                (d*1.5, d*np.sqrt(3)/2),
                (-d*1.5, -d*np.sqrt(3)/2),
            ]
            for (x, y) in positions:
                inner.append([x, y, angle_offset])
                
        return inner
    
    # Try multiple configuration types and angles
    best_side = 1e10
    best_result = None
    
    config_types = ['flower7_plus4', 'hex_grid', 'compact_cluster', 'linear_chain']
    angle_offsets = [0.0, 30.0, 60.0, 15.0, 45.0]
    
    for ct in config_types:
        if time.time() > deadline:
            break
        for ao in angle_offsets:
            if time.time() > deadline:
                break
            try:
                inner = generate_configuration(ct, ao)
                if len(inner) != N:
                    continue
                # Get all vertices
                all_pts = []
                for (cx, cy, ang) in inner:
                    verts = get_hex_verts(np.array([cx, cy]), ang)
                    all_pts.append(verts)
                all_pts = np.vstack(all_pts)
                
                # Find minimum enclosing hexagon
                result = find_min_enclosing_hexagon(all_pts)
                if result is not None:
                    side, center, angle = result
                    if side < best_side:
                        best_side = side
                        best_result = (inner, center, side, angle)
            except:
                continue
    
    if best_result is not None:
        inner, center, side, angle = best_result
        return inner, center, side, angle
    else:
        # Fallback: simple configuration
        inner = [[0.0, 0.0, 0.0]]
        d = np.sqrt(3)
        for k in range(6):
            ang = np.radians(60 * k)
            inner.append([d * np.cos(ang), d * np.sin(ang), 0.0])
        d2 = 2 * d
        for k in range(4):
            ang = np.radians(90 * k + 30)
            inner.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

# EVOLVE-BLOCK-END
