"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    """Place N unit regular hexagons inside a regular hexagon, minimizing outer side length."""
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    # Hexagon geometry: unit regular hexagon centered at origin, flat-top orientation
    # Vertices of unit hexagon (side=1, flat-top) at angles 0, 60, 120, 180, 240, 300
    # For a hexagon with center (cx, cy) and rotation angle theta:
    # vertices = center + R(theta) @ base_vertices
    
    def hexagon_vertices(cx, cy, theta_deg):
        """Get vertices of unit regular hexagon with center (cx,cy) and rotation theta_deg."""
        theta = np.deg2rad(theta_deg)
        # Base vertices for unit hexagon, flat-top (vertex at angle 0)
        angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
        base_verts = np.column_stack([np.cos(angles), np.sin(angles)])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return np.array([[cx, cy]]) + base_verts @ R.T
    
    def outer_hexagon_vertices(cx, cy, side, theta_deg):
        """Get vertices of regular hexagon with center (cx,cy), side length side, rotation theta_deg."""
        theta = np.deg2rad(theta_deg)
        angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
        base_verts = np.column_stack([np.cos(angles), np.sin(angles)]) * side
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return np.array([[cx, cy]]) + base_verts @ R.T
    
    def point_in_hexagon(px, py, cx, cy, side, theta_deg):
        """Check if point (px,py) is inside regular hexagon with center (cx,cy), side, rotation."""
        # Transform to hexagon's local frame
        theta = np.deg2rad(theta_deg)
        dx, dy = px - cx, py - cy
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        lx = cos_t * dx + sin_t * dy
        ly = -sin_t * dx + cos_t * dy
        # Unit hexagon (side=1) has constraints: |lx| <= 1, |ly| <= sqrt(3)/2, |lx/2 + ly*sqrt(3)/2| <= 1
        # More precisely, for regular hexagon with circumradius = side:
        # The hexagon is intersection of 6 half-planes
        # Directions to edges: 30, 90, 150, 210, 270, 330 degrees (for flat-top)
        # Actually for pointy-top with vertex at 0 deg, edges are at 30, 90, etc.
        # Let's use: for unit hexagon (circumradius=1), the inradius = sqrt(3)/2
        # Support function: max over 6 directions
        edge_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
        edge_normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        # The hexagon with circumradius r has support r in vertex directions
        # and inradius = r * cos(30) = r * sqrt(3)/2 in edge normal directions
        # For side length s, circumradius = s, inradius = s * sqrt(3)/2
        inradius = side * np.sqrt(3) / 2.0
        # In local frame, check if |dot(p, n)| <= inradius for all 6 edge normals
        dots = np.abs(np.array([lx, ly]) @ edge_normals.T)
        return np.all(dots <= inradius)
    
    def is_point_in_hexagon_fast(px, py, cx, cy, side, theta_deg):
        """Fast check using support function."""
        theta = np.deg2rad(theta_deg)
        dx, dy = px - cx, py - cy
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        lx = cos_t * dx + sin_t * dy
        ly = -sin_t * dx + cos_t * dy
        edge_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
        edge_normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        inradius = side * np.sqrt(3) / 2.0
        dots = np.abs(np.array([lx, ly]) @ edge_normals.T)
        return np.all(dots <= inradius)
    
    def get_all_inner_vertices(inner):
        """Get all vertices of all inner hexagons."""
        verts = []
        for (cx, cy, theta_deg) in inner:
            hv = hexagon_vertices(cx, cy, theta_deg)
            verts.extend(hv.tolist())
        return np.array(verts)
    
    def all_vertices_in_outer(inner, outer_center, outer_side, outer_theta):
        """Check if all vertices of inner hexagons are inside outer hexagon."""
        vertices = get_all_inner_vertices(inner)
        ocx, ocy = outer_center
        for px, py in vertices:
            if not is_point_in_hexagon_fast(px, py, ocx, ocy, outer_side, outer_theta):
                return False
        return True
    
    def violation_amount(inner, outer_center, outer_side, outer_theta):
        """Return max violation (positive if any vertex outside)."""
        vertices = get_all_inner_vertices(inner)
        ocx, ocy = outer_center
        max_viol = 0.0
        for px, py in vertices:
            theta = np.deg2rad(outer_theta)
            dx, dy = px - ocx, py - ocy
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            lx = cos_t * dx + sin_t * dy
            ly = -sin_t * dx + cos_t * dy
            edge_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
            edge_normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
            inradius = outer_side * np.sqrt(3) / 2.0
            dots = np.abs(np.array([lx, ly]) @ edge_normals.T)
            viol = np.max(dots) - inradius
            if viol > max_viol:
                max_viol = viol
        return max_viol
    
    def objective(params):
        """Objective: minimize outer side length with constraint that all inner hexagons fit."""
        n = N
        # params: [x1,y1,th1, x2,y2,th2, ..., xn,yn,thn, ocx, ocy, oside, otheta]
        # Total: 3*n + 4
        inner = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i + 1]
            th = params[3*i + 2]
            inner.append([x, y, th])
        ocx = params[3*n]
        ocy = params[3*n + 1]
        oside = params[3*n + 2]
        otheta = params[3*n + 3]
        
        # Compute violation
        viol = violation_amount(inner, (ocx, ocy), oside, otheta)
        # Penalize violations heavily
        if viol > 0:
            return oside + 100.0 * viol
        return oside
    
    # Initial guess: hexagonal packing
    # For 11 hexagons, try a compact arrangement
    # Let's try: center + ring of 6 + ring of 4 more
    # Or simple: 3-4-3-1 arrangement
    
    # Start with a reasonable initial configuration
    # Hexagonal close packing: centers at distance sqrt(3) apart (flat-to-flat)
    d = np.sqrt(3)  # distance between adjacent hexagon centers (flat-to-flat)
    
    # Try a few initial configurations
    best_result = None
    best_score = float('inf')
    
    configs = []
    
    # Config 1: linear arrangement
    lin_config = []
    for i in range(N):
        lin_config.append([i * d, 0.0, 0.0])
    configs.append(lin_config)
    
    # Config 2: hexagonal packing (center + 6 around + 4 more)
    hex_config = [[0.0, 0.0, 0.0]]
    # Ring 1: 6 hexagons at distance d
    for k in range(6):
        angle = np.pi / 3 * k
        hex_config.append([d * np.cos(angle), d * np.sin(angle), 0.0])
    # Ring 2 partial: add 4 more
    for k in [0, 2, 4, 6]:
        angle = np.pi / 3 * k + np.pi / 6
        hex_config.append([2 * d * np.cos(angle), 2 * d * np.sin(angle), 0.0])
    configs.append(hex_config)
    
    # Config 3: 3-4-3-1
    cfg3 = []
    # Row 0: 3 hexagons
    for i in range(3):
        cfg3.append([(i - 1) * d, 0.0, 0.0])
    # Row 1: 4 hexagons (offset)
    for i in range(4):
        cfg3.append([(i - 1.5) * d, d * np.sqrt(3) / 2, 0.0])
    # Row 2: 3 hexagons
    for i in range(3):
        cfg3.append([(i - 1) * d, d * np.sqrt(3), 0.0])
    # Row 3: 1 hexagon
    cfg3.append([0.0, d * np.sqrt(3) * 1.5, 0.0])
    configs.append(cfg3)
    
    # Config 4: 4-3-4
    cfg4 = []
    for i in range(4):
        cfg4.append([(i - 1.5) * d, 0.0, 0.0])
    for i in range(3):
        cfg4.append([(i - 1.0) * d, d * np.sqrt(3) / 2, 0.0])
    for i in range(4):
        cfg4.append([(i - 1.5) * d, d * np.sqrt(3), 0.0])
    configs.append(cfg4)
    
    for cfg in configs:
        if time.time() > deadline:
            break
        
        # Estimate bounding box
        pts = np.array([[c[0], c[1]] for c in cfg])
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        
        # Estimate required side length
        # Max extent in various directions
        max_extent = 0
        for angle_deg in range(0, 360, 30):
            angle = np.deg2rad(angle_deg)
            projections = pts @ np.array([np.cos(angle), np.sin(angle)])
            extent = np.max(projections) - np.min(projections)
            max_extent = max(max_extent, extent)
        
        # Outer hexagon with side s has width across flats = s*sqrt(3), across corners = 2*s
        # We need s*sqrt(3) >= max_extent in flat directions AND 2*s >= max_extent in corner directions
        # Conservative estimate
        est_side = max_extent / np.sqrt(3) + 1.0  # Add margin
        
        # Build initial params
        params = []
        for (x, y, th) in cfg:
            params.extend([x, y, th])
        params.extend([cx, cy, est_side, 0.0])
        
        # Optimize
        try:
            result = minimize(objective, np.array(params, dtype=float),
                             method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.success or result.fun < best_score:
                opt_params = result.x
                n_inner = N
                inner_opt = []
                for i in range(n_inner):
                    inner_opt.append([opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]])
                ocx = opt_params[3*n_inner]
                ocy = opt_params[3*n_inner + 1]
                oside = opt_params[3*n_inner + 2]
                otheta = opt_params[3*n_inner + 3]
                
                # Verify
                if all_vertices_in_outer(inner_opt, (ocx, ocy), oside, otheta):
                    if oside < best_score:
                        best_score = oside
                        best_result = (inner_opt, (ocx, ocy), oside, otheta)
        except Exception:
            pass
    
    # If we found a valid result, return it
    if best_result is not None:
        inner, center, side, angle = best_result
        return inner, center, side, angle
    
    # Fallback: linear arrangement
    inner = []
    d = np.sqrt(3)
    for i in range(N):
        inner.append([(i - N//2) * d, 0.0, 0.0])
    
    # Compute bounding
    pts = np.array([[c[0], c[1]] for c in inner])
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    max_extent = 0
    for angle_deg in range(0, 360, 30):
        angle = np.deg2rad(angle_deg)
        projections = pts @ np.array([np.cos(angle), np.sin(angle)])
        extent = np.max(projections) - np.min(projections)
        max_extent = max(max_extent, extent)
    
    # Add margin for hexagon vertices
    side = max_extent / np.sqrt(3) + 2.0
    
    return inner, (cx, cy), side, 0.0
# EVOLVE-BLOCK-END
