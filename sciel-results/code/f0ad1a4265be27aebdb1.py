import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Leave 5s margin

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _compute_d_min():
    """
    Precompute minimum distance between two unit hexagons
    as a function of relative rotation angle alpha in [0, 30].
    """
    alphas = np.linspace(0, 30, 61)
    d_mins = []
    for a_deg in alphas:
        # Hex 1 at origin, rot 0
        # Hex 2 at (d, 0), rot a_deg
        # Find min d such that they touch.
        
        def intersects(d, a_rad):
            v1 = _get_hex_vertices(0, 0, 0)
            v2 = _get_hex_vertices(d, 0, math.degrees(a_rad))
            
            normals = []
            for k in range(6):
                ang = math.radians(k * 60)
                normals.append((math.cos(ang), math.sin(ang)))
            for k in range(6):
                ang = math.radians(a_rad + k * 60)
                normals.append((math.cos(ang), math.sin(ang)))
            
            for nx, ny in normals:
                p1 = [vx * nx + vy * ny for vx, vy in v1]
                p2 = [vx * nx + vy * ny for vx, vy in v2]
                if max(p1) < min(p2) or max(p2) < min(p1):
                    return False
            return True

        lo, hi = 1.0, 3.0
        for _ in range(20):
            mid = (lo + hi) / 2
            if intersects(mid, math.radians(a_deg)):
                lo = mid
            else:
                hi = mid
        d_mins.append(lo)
    return alphas, d_mins

_ALPHAS, _D_MINS = _compute_d_min()

def _get_d_min(alpha_deg):
    """Interpolate d_min for a given relative angle."""
    alpha = abs(alpha_deg) % 60
    if alpha > 30:
        alpha = 60 - alpha
    idx = int(alpha / 0.5)
    if idx >= len(_D_MINS) - 1:
        return _D_MINS[-1]
    frac = (alpha - idx * 0.5) / 0.5
    return _D_MINS[idx] * (1 - frac) + _D_MINS[idx + 1] * frac

def _check_disjoint(x_i, y_i, th_i, eps=1e-4):
    for i in range(N):
        for j in range(i + 1, N):
            dx = x_i[i] - x_i[j]
            dy = y_i[i] - y_i[j]
            d = math.sqrt(dx*dx + dy*dy)
            alpha = th_i[i] - th_i[j]
            d_min = _get_d_min(alpha)
            if d < d_min - eps:
                return False
    return True

def _calc_outer_side(x_i, y_i, th_i):
    """
    Calculate the minimum side length of a regular hexagon
    that can contain all inner hexagons.
    We optimize the outer hexagon center (X,Y) and rotation (Theta).
    """
    all_vertices = []
    for i in range(N):
        verts = _get_hex_vertices(x_i[i], y_i[i], th_i[i])
        all_vertices.extend(verts)
    
    # Coarse grid search for outer hex rotation and center
    # The optimal center is likely the centroid of the vertices
    cx = np.mean([v[0] for v in all_vertices])
    cy = np.mean([v[1] for v in all_vertices])
    
    best_side = float('inf')
    
    # Search over rotations
    for theta_deg in np.arange(0, 60, 5):
        # For a fixed rotation, find optimal center?
        # Actually, for a fixed rotation, the problem is to find the smallest hexagon of that rotation containing points.
        # This is equivalent to finding the min t such that points are in Hex(center, t, theta).
        # Hex(center, t, theta) is intersection of 6 half-planes.
        # This is a LP. But we can just use the smooth approximation or a local search on X,Y.
        
        # Local search on X, Y for this theta
        # Initial guess: centroid
        def obj_xy(params):
            X, Y = params
            normals = []
            for k in range(6):
                a = math.radians(theta_deg + 30 + 60 * k)
                normals.append((math.cos(a), math.sin(a)))
            
            max_proj = 0
            for nx, ny in normals:
                proj = [ (vx - X)*nx + (vy - Y)*ny for vx, vy in all_vertices ]
                # The width in this direction is max(proj) - min(proj)
                # The distance from center to the furthest point in direction n is max(proj)
                # Wait, if the hexagon is centered at (X,Y), the constraint is:
                # |(x-X, y-Y) dot n| <= t * apothem? No.
                # A regular hexagon of side s has apothem s * sqrt(3)/2.
                # The condition for a point (x,y) to be inside Hex(C, s, theta) is:
                # For all 6 normals n_k: (x-C) dot n_k <= s * sqrt(3)/2
                # So we need s * sqrt(3)/2 >= max_k ( (x-C) dot n_k ) for all x.
                # So s = (2/sqrt(3)) * max_k max_x ( (x-C) dot n_k )
                
                # Let's compute R = max_x ( (x-C) dot n_k ) for each k.
                # We want to minimize max(R_1, ..., R_6).
                # This is a convex function of C.
                
                # For this fixed theta, let's just do a few steps of gradient descent on X,Y
                # Or just use the centroid and refine.
                pass
            
            # Simplified: Just evaluate the side length for center (X,Y)
            widths = []
            for nx, ny in normals:
                proj = [ (vx - X)*nx + (vy - Y)*ny for vx, vy in all_vertices ]
                # The distance from center to the supporting line in direction n
                # If the hexagon is centered at C, the boundary is at distance apothem from C.
                # So we need apothem >= max( (x-C) dot n )
                # Note: (x-C) dot n can be negative. The max will be positive.
                max_proj = max(proj)
                widths.append(max_proj)
            
            max_width = max(widths)
            # Side length s = 2 / sqrt(3) * apothem
            return 2.0 / math.sqrt(3) * max_width

        # Optimize X, Y
        res = minimize(obj_xy, [cx, cy], method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        side = res.fun
        if side < best_side:
            best_side = side
            
    return best_side

def _smooth_outer_side(params, all_vertices):
    """
    params: [X, Y, Theta_deg]
    all_vertices: list of (x, y)
    """
    X, Y, Theta_deg = params
    normals = []
    for k in range(6):
        a = math.radians(Theta_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    widths = []
    for nx, ny in normals:
        projections = [(vx - X) * nx + (vy - Y) * ny for vx, vy in all_vertices]
        width = max(projections) - min(projections)
        # Wait, if we center at (X,Y), the constraint is max(projections) <= apothem.
        # The "width" relative to center is just max(projections) if we assume the min is <= 0?
        # No, the hexagon is symmetric. The condition is |(x-C) dot n| <= apothem.
        # So we need apothem >= max( (x-C) dot n, -(x-C) dot n ) = max( |(x-C) dot n| ).
        # So we should use abs.
        max_abs_proj = max(abs(p) for p in projections)
        widths.append(max_abs_proj)
    
    max_width = max(widths)
    if max_width <= 0:
        return 0.0
    beta = 100.0
    exp_sum = sum(math.exp(w - max_width) for w in widths)
    smoothed_max = max_width + math.log(exp_sum) / beta
    return (2.0 / math.sqrt(3)) * smoothed_max

def _objective(params):
    if time.time() > _DEADLINE:
        return float('inf')
        
    x_i = params[0:11]
    y_i = params[11:22]
    th_i = params[22:33]
    X = params[33]
    Y = params[34]
    Th_outer = params[35]
    
    all_vertices = []
    for i in range(N):
        verts = _get_hex_vertices(x_i[i], y_i[i], th_i[i])
        all_vertices.extend(verts)
    
    S = _smooth_outer_side([X, Y, Th_outer], all_vertices)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x_i[i] - x_i[j]
            dy = y_i[i] - y_i[j]
            d = math.sqrt(dx*dx + dy*dy)
            alpha = th_i[i] - th_i[j]
            d_min = _get_d_min(alpha)
            if d < d_min:
                penalty += (d_min - d) ** 2
    
    return S + 10000.0 * penalty

def _repair_overlaps(x_i, y_i, th_i, iterations=20):
    """
    Simple iterative repair: if two hexes overlap, push them apart.
    """
    for _ in range(iterations):
        if _check_disjoint(x_i, y_i, th_i, eps=1e-5):
            break
        for i in range(N):
            for j in range(i + 1, N):
                dx = x_i[i] - x_i[j]
                dy = y_i[i] - y_i[j]
                d = math.sqrt(dx*dx + dy*dy)
                alpha = th_i[i] - th_i[j]
                d_min = _get_d_min(alpha)
                
                if d < d_min:
                    # Push apart
                    # Direction from j to i
                    if d == 0:
                        dx, dy = 1.0, 0.0
                        d = 1.0
                    else:
                        dx /= d
                        dy /= d
                    
                    overlap = d_min - d + 1e-4
                    # Move each by half the overlap
                    x_i[i] += dx * overlap * 0.5
                    y_i[i] += dy * overlap * 0.5
                    x_i[j] -= dx * overlap * 0.5
                    y_i[j] -= dy * overlap * 0.5

def construct():
    global _START_TIME, _DEADLINE
    
    # Initial guess: Symmetric layout
    x0 = []
    y0 = []
    th0 = []
    
    # Center
    x0.append(0.0); y0.append(0.0); th0.append(0.0)
    
    # Ring 1
    for k in range(6):
        ang = math.radians(k * 60)
        x0.append(math.sqrt(3) * math.cos(ang))
        y0.append(math.sqrt(3) * math.sin(ang))
        th0.append(0.0)
        
    # Ring 2 (4 hexes)
    # Place at 0, 90, 180, 270 to avoid clustering
    angles_r2 = [0, 90, 180, 270]
    for ang_deg in angles_r2:
        ang = math.radians(ang_deg)
        # Distance 2*sqrt(3) is touching. Use slightly more to start disjoint?
        # Actually, if they touch, penalty is 0. But numerical issues might cause overlap.
        # Let's use 2.02 * sqrt(3) to be safe in initial guess.
        dist = 2.02 * math.sqrt(3)
        x0.append(dist * math.cos(ang))
        y0.append(dist * math.sin(ang))
        th0.append(0.0)
        
    # Outer hex
    X0 = 0.0
    Y0 = 0.0
    Th0_outer = 0.0
    
    initial_params = x0 + y0 + th0 + [X0, Y0, Th0_outer]
    
    # Bounds
    bounds = []
    for _ in range(11):
        bounds.append((-5, 5)) # x
    for _ in range(11):
        bounds.append((-5, 5)) # y
    for _ in range(11):
        bounds.append((-180, 180)) # theta
    bounds.append((-5, 5)) # X
    bounds.append((-5, 5)) # Y
    bounds.append((-180, 180)) # Th_outer
    
    best_result = None
    best_score = float('inf')
    
    # Differential Evolution
    if time.time() < _DEADLINE - 20:
        try:
            de_result = differential_evolution(
                _objective, bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15,
                tol=1e-6,
                polish=False
            )
            if de_result.success:
                best_result = de_result.x
                best_score = de_result.fun
        except Exception as e:
            pass
    
    # Polish with L-BFGS-B
    if best_result is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_result, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < best_score:
                best_result = res.x
                best_score = res.fun
        except Exception:
            pass
    
    if best_result is None:
        best_result = initial_params
        best_score = _objective(best_result)
        
    x_i = list(best_result[0:11])
    y_i = list(best_result[11:22])
    th_i = list(best_result[22:33])
    
    # Repair overlaps
    _repair_overlaps(x_i, y_i, th_i)
    
    # Check if disjoint
    if not _check_disjoint(x_i, y_i, th_i):
        # Fallback to a very safe configuration
        # Large outer hex, hexes in a line
        x_f = [i * 2.5 for i in range(N)]
        y_f = [0.0] * N
        th_f = [0.0] * N
        x_i, y_i, th_i = x_f, y_f, th_f
    
    # Calculate true outer side
    side = _calc_outer_side(x_i, y_i, th_i)
    
    # Center of outer hex (approximate)
    cx = np.mean(x_i)
    cy = np.mean(y_i)
    
    inner = []
    for i in range(N):
        inner.append([x_i[i], y_i[i], th_i[i]])
        
    return inner, (cx, cy), side, 0.0

def construct_fallback():
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
