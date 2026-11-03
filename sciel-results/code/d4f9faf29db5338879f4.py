"""Combined program: LP-based outer hexagon with continuous angle search + inner polish."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """Use LP to find optimal center for given outer hexagon angle.
    Returns (cx, cy, side_length) or None on failure."""
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    
    # Variables: cx, cy, t (inradius)
    # Minimize t
    c = [0.0, 0.0, 1.0]
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        
        # For each vertex: (vx - cx)*nx + (vy - cy)*ny <= t
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
        
        # For each vertex: -(vx - cx)*nx - (vy - cy)*ny <= t
        for vx, vy in all_vertices:
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)
    
    # t >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                        bounds=[(None, None), (None, None), (0, None)], 
                        method='highs')
        if result.success:
            cx, cy, t = result.x
            side = t / (math.sqrt(3) / 2)
            return cx, cy, side
    except:
        pass
    
    return None

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Uses continuous angle search. Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse search: 0 to 30 degrees (due to 60-degree symmetry)
    for angle_deg in np.arange(0, 30.01, 0.5):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = angle_deg
    
    # Fine search around best coarse angle
    if best_angle > 0 or best_side < float('inf'):
        search_range = 0.5
        step = 0.01
        for angle_deg in np.arange(max(0, best_angle - search_range), 
                                   min(30, best_angle + search_range) + step, step):
            if time.time() > _DEADLINE - 10.0:
                break
            res = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = angle_deg
    
    return best_center[0], best_center[1], best_side, best_angle

def _check_disjoint(inner_hexes):
    """Check if all inner hexagons have disjoint interiors."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner_hexes[i]
            x2, y2, a2 = inner_hexes[j]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # Conservative check for overlapping
            if dist < math.sqrt(3) - 0.001:
                return False
    return True

def _nelder_mead_polish(inner_hexes, fixed_angle, max_iter=200, tol=1e-6):
    """Polish inner hexagon positions using Nelder-Mead to minimize enclosing hexagon side at fixed angle."""
    sqrt3 = math.sqrt(3)
    
    def objective(params):
        # params: flattened [x1, y1, a1, x2, y2, a2, ...]
        new_inner = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            new_inner.append([x, y, a])
        
        # Check disjointness - if violated, return large penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = new_inner[i][0] - new_inner[j][0]
                dy = new_inner[i][1] - new_inner[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < sqrt3 - 0.005:
                    return 1000.0 + (sqrt3 - dist) * 100
        
        # Use LP at fixed angle for speed
        res = _min_enclosing_hex_side_lp(new_inner, fixed_angle)
        if res is None:
            return 1000.0
        return res[2]
    
    # Initial params
    x0 = []
    for h in inner_hexes:
        x0.extend([h[0], h[1], h[2]])
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
        
        if result.fun < 1000.0:
            polished = []
            for i in range(N):
                polished.append([result.x[i*3], result.x[i*3+1], result.x[i*3+2]])
            return polished
    except:
        pass
    
    return inner_hexes

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate 11 points in a hexagonal cluster arrangement
    # Use triangular lattice with spacing sqrt(3) for edge-to-edge contact
    sqrt3 = math.sqrt(3)
    # Scale factor to ensure strict disjointness (small gap between hexagons)
    scale = 1.000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Hexagonal cluster: center + 6 neighbors + 4 of the 12 next-nearest
    # This gives a compact hexagonal shape
    # Layer 0: (0,0)
    # Layer 1: 6 points at distance sqrt(3)*scale
    # Layer 2: 12 points at distance 2*sqrt(3)*scale (take 5 to make 11 total)
    
    positions = [(0.0, 0.0)]
    
    # Layer 1: 6 neighbors
    for k in range(6):
        a = math.pi * k / 3
        x = sqrt3 * scale * math.cos(a)
        y = sqrt3 * scale * math.sin(a)
        positions.append((x, y))
    
    # Layer 2: take 4 more from the 12 next-nearest neighbors
    # These are at positions like (2*sqrt3*scale, 0), (sqrt3*scale, 1.5*scale), etc.
    layer2_candidates = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            # Only include points at the right distance (next-nearest)
            if 3.5 < dist_sq < 4.5:  # approximately (2*sqrt3*scale)^2 = 12*scale^2 ≈ 12
                # Actually let me compute properly
                # Nearest neighbors are at distance sqrt(3)*scale, so squared = 3*scale^2
                # Next nearest at distance 2*sqrt(3)*scale or sqrt(3)*sqrt(3)*scale = 3*scale
                # Let's just collect all and filter by being in layer 2
                layer2_candidates.append((dist_sq, x, y))
    
    # Sort and take closest 4
    layer2_candidates.sort(key=lambda t: t[0])
    for _, x, y in layer2_candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    # If we don't have enough, add more from the grid
    if len(positions) < N:
        all_candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                dist_sq = x * x + y * y
                all_candidates.append((dist_sq, x, y))
        all_candidates.sort(key=lambda t: t[0])
        existing = set((round(x, 6), round(y, 6)) for x, y in positions)
        for _, x, y in all_candidates:
            key = (round(x, 6), round(y, 6))
            if key not in existing:
                positions.append((x, y))
                existing.add(key)
                if len(positions) == N:
                    break
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try inner angles 0 and 30
    for inner_angle in [0.0, 30.0]:
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner = [[pos[0], pos[1], inner_angle] for pos in positions]
        
        # Compute initial minimum enclosing hexagon
        cx, cy, side, angle = _min_enclosing_hex_side(inner)
        
        if side < best_side:
            best_side = side
            best_inner = [h[:] for h in inner]
            best_center = (cx, cy)
            best_outer_angle = angle
    
    if best_inner is None:
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Try Nelder-Mead polish if time permits - use fixed angle for speed
    if time.time() < _DEADLINE - 40.0:
        polished = _nelder_mead_polish(best_inner, fixed_angle=best_outer_angle, max_iter=100, tol=1e-5)
        if polished is not best_inner:
            # Re-find best angle for polished configuration
            pcx, pcy, pside, pangle = _min_enclosing_hex_side(polished)
            if pside < best_side - 1e-6:
                best_side = pside
                best_inner = polished
                best_center = (pcx, pcy)
                best_outer_angle = pangle
    
    # Second polish round if time permits
    if time.time() < _DEADLINE - 25.0 and best_inner is not None:
        polished2 = _nelder_mead_polish(best_inner, fixed_angle=best_outer_angle, max_iter=80, tol=1e-6)
        if polished2 is not best_inner:
            pcx2, pcy2, pside2, pangle2 = _min_enclosing_hex_side(polished2)
            if pside2 < best_side - 1e-7:
                best_side = pside2
                best_inner = polished2
                best_center = (pcx2, pcy2)
                best_outer_angle = pangle2
    
    # Final high-precision LP polish for outer hexagon
    if time.time() < _DEADLINE - 10.0:
        # Try a finer range of angles around the best angle
        for da in np.arange(-1.5, 1.51, 0.05):
            if time.time() > _DEADLINE - 5.0:
                break
            test_angle = (best_outer_angle + da) % 60
            if test_angle < 0 or test_angle > 30:
                continue
            res = _min_enclosing_hex_side_lp(best_inner, test_angle)
            if res:
                tcx, tcy, tside = res
                if tside < best_side - 1e-9:
                    best_side = tside
                    best_center = (tcx, tcy)
                    best_outer_angle = test_angle
    
    return best_inner, (best_center[0], best_center[1]), best_side, best_outer_angle
