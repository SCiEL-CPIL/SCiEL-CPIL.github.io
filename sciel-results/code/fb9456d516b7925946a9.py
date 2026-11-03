import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    deadline = time.time() + 140.0
    
    def hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        return [(cx + side * math.cos(angle + i * math.pi / 3),
                 cy + side * math.sin(angle + i * math.pi / 3)) for i in range(6)]
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
        # Transform to hexagon's local coordinate system
        dx = px - cx
        dy = py - cy
        angle = math.radians(-angle_deg)
        lx = dx * math.cos(angle) - dy * math.sin(angle)
        ly = dx * math.sin(angle) + dy * math.cos(angle)
        
        # A regular hexagon with side s has vertices at distance s from center
        # In local coords, the hexagon is defined by |lx| <= s*sqrt(3)/2 for the flat sides
        # Actually, let's use the support function approach
        # The hexagon with circumradius r=side has 6 faces
        # Face normals at angles 0, 60, 120, 180, 240, 300 degrees
        r = side
        for i in range(6):
            theta = i * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            # Distance from center to face
            d = lx * nx + ly * ny
            # The inradius is r * cos(pi/6) = r * sqrt(3)/2
            if d > r * math.sqrt(3) / 2 + 1e-9:
                return False
        return True
    
    def enclosing_hexagon_side(inner_hexes, outer_center, outer_angle_deg):
        """Find minimum side length of outer hexagon (centered at outer_center, rotated by outer_angle_deg) 
        that contains all vertices of all inner hexagons."""
        ox, oy = outer_center
        outer_angle = math.radians(outer_angle_deg)
        
        max_side = 0.0
        for (ix, iy, iangle) in inner_hexes:
            verts = hex_vertices(ix, iy, 1.0, iangle)
            for (vx, vy) in verts:
                # Transform point to outer hexagon's local frame
                dx = vx - ox
                dy = vy - oy
                lx = dx * math.cos(-outer_angle) - dy * math.sin(-outer_angle)
                ly = dx * math.sin(-outer_angle) + dy * math.cos(-outer_angle)
                
                # For a hexagon with circumradius R, a point (lx,ly) is inside iff
                # for all 6 face normals, lx*nx + ly*ny <= R * sqrt(3)/2
                # The minimum R needed is max over faces of (lx*nx + ly*ny) / (sqrt(3)/2)
                for i in range(6):
                    theta = i * math.pi / 3
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    d = lx * nx + ly * ny
                    needed_R = d / (math.sqrt(3) / 2) if d > 0 else 0.0
                    if needed_R > max_side:
                        max_side = needed_R
        return max_side
    
    def total_side(inner_hexes, outer_center, outer_angle_deg):
        return enclosing_hexagon_side(inner_hexes, outer_center, outer_angle_deg)
    
    # Generate candidate lattice arrangements
    # Hexagonal lattice: centers at distance 2 apart
    # Row offsets for hex packing
    candidates = []
    
    # Pattern 1: 3-4-3-2 rows (or similar)
    # Let's try various row configurations that sum to 12
    row_patterns = [
        [3, 4, 3, 2],
        [2, 4, 3, 3],
        [3, 3, 4, 2],
        [4, 3, 3, 2],
        [3, 2, 4, 3],
        [2, 3, 4, 3],
        [3, 3, 3, 3],
        [4, 4, 2, 2],
        [2, 2, 4, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [5, 3, 2, 2],
        [2, 2, 3, 5],
        [3, 5, 2, 2],
        [2, 3, 5, 2],
        [2, 2, 5, 3],
        [5, 2, 2, 3],
        [3, 2, 2, 5],
        [4, 3, 2, 3],
        [3, 2, 3, 4],
        [2, 3, 3, 4],
        [3, 4, 2, 3],
        [4, 2, 3, 3],
        [3, 3, 2, 4],
    ]
    
    for pattern in row_patterns:
        if sum(pattern) != N:
            continue
        # Generate hex lattice positions
        # Row i has pattern[i] hexagons, centered horizontally
        # Vertical spacing: sqrt(3) * 2 = 2*sqrt(3) between rows (for touching hexagons)
        # Actually for hex lattice with side 1, center-to-center distance is 2
        # Vertical distance between rows: 2 * sin(60°) = sqrt(3)
        # Wait: if hexagons have side 1, their inradius is sqrt(3)/2
        # For two hexagons to touch, center distance = 2 * inradius = sqrt(3)
        # No wait: two unit hexagons touching: the distance between centers = 2 * (sqrt(3)/2) * 2? 
        # Let me think again. A unit hexagon has circumradius 1, inradius sqrt(3)/2.
        # Two hexagons touching side-to-side: center distance = 2 * inradius = sqrt(3)
        # But that's for flat-to-flat. For the hexagonal lattice, the nearest neighbor distance is 2.
        # Actually, for regular hexagons of side s, the distance between centers of adjacent hexagons 
        # in a honeycomb is 2*s. So for s=1, it's 2.
        
        # Hex lattice: 
        # Row 0: x = 0, 2, 4, ...
        # Row 1: x = 1, 3, 5, ... (offset by 1), y = sqrt(3)
        # Row 2: x = 0, 2, 4, ..., y = 2*sqrt(3)
        
        hexes = []
        n_rows = len(pattern)
        for r in range(n_rows):
            n_in_row = pattern[r]
            # Center this row horizontally
            # x positions: start from -(n_in_row-1) to (n_in_row-1) step 2
            x_start = -(n_in_row - 1)
            y_pos = r * math.sqrt(3)
            for j in range(n_in_row):
                x_pos = x_start + 2 * j
                # Offset odd rows by 1
                if r % 2 == 1:
                    x_pos += 1
                hexes.append([float(x_pos), float(y_pos), 0.0])
        
        # Center the arrangement
        xs = [h[0] for h in hexes]
        ys = [h[1] for h in hexes]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        hexes = [[h[0] - cx, h[1] - cy, h[2]] for h in hexes]
        
        candidates.append(hexes)
    
    # Also try a simple 2D grid-like arrangement
    # 4x3 grid with hex offset
    for cols in [3, 4]:
        rows = N // cols
        remainder = N % cols
        hexes = []
        for r in range(rows):
            n_in_row = cols + (1 if r < remainder else 0)
            x_start = -(n_in_row - 1)
            y_pos = r * math.sqrt(3)
            for j in range(n_in_row):
                x_pos = x_start + 2 * j
                if r % 2 == 1:
                    x_pos += 1
                hexes.append([float(x_pos), float(y_pos), 0.0])
        xs = [h[0] for h in hexes]
        ys = [h[1] for h in hexes]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        hexes = [[h[0] - cx, h[1] - cy, h[2]] for h in hexes]
        candidates.append(hexes)
    
    # Evaluate each candidate with optimal outer hexagon rotation
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    for hexes in candidates:
        # Try different outer hexagon rotations
        for outer_angle_deg in [0, 30, 60, 90, 15, 45, 75]:
            side = enclosing_hexagon_side(hexes, (0.0, 0.0), outer_angle_deg)
            if side < best_side:
                best_side = side
                best_inner = [list(h) for h in hexes]
                best_outer_center = (0.0, 0.0)
                best_outer_angle = outer_angle_deg
    
    # Fine-tune with optimization on the best candidate
    if best_inner is not None and time.time() < deadline - 30:
        # Optimize the outer hexagon center and angle
        def objective(params):
            ox, oy, oangle = params
            return enclosing_hexagon_side(best_inner, (ox, oy), oangle)
        
        x0 = np.array([best_outer_center[0], best_outer_center[1], math.radians(best_outer_angle)])
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            if result.fun < best_side:
                best_side = result.fun
                best_outer_center = (result.x[0], result.x[1])
                best_outer_angle = math.degrees(result.x[2])
        except:
            pass
        
        # Also try optimizing inner hexagon positions slightly
        if time.time() < deadline - 20:
            def obj_inner(params):
                # params: flattened positions of inner hexagons (x, y for each)
                # Keep angles fixed at 0
                new_hexes = []
                for i in range(len(best_inner)):
                    new_hexes.append([params[2*i], params[2*i+1], 0.0])
                return enclosing_hexagon_side(new_hexes, best_outer_center, best_outer_angle)
            
            x0_inner = np.array([v for h in best_inner for v in h[:2]])
            try:
                result2 = minimize(obj_inner, x0_inner, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
                if result2.fun < best_side:
                    best_side = result2.fun
                    new_hexes = []
                    for i in range(len(best_inner)):
                        new_hexes.append([result2.x[2*i], result2.x[2*i+1], 0.0])
                    best_inner = new_hexes
            except:
                pass
    
    # Safety: if optimization failed or gave bad result, use the best lattice result
    if best_inner is None or best_side >= 40.0:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_outer_center, best_side, best_outer_angle

