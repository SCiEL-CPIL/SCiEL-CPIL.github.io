"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 12
SQRT3 = math.sqrt(3.0)

# EVOLVE-BLOCK-START

def _hex_verts(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _outer_verts(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _point_in_convex(px, py, poly, tol=1e-9):
    """Check if point is inside convex polygon (CCW)."""
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        ex = poly[j, 0] - poly[i, 0]
        ey = poly[j, 1] - poly[i, 1]
        cross = ex * (py - poly[i, 1]) - ey * (px - poly[i, 0])
        if cross < -tol:
            return False
    return True

def _sat_overlap(v1, v2, tol=1e-9):
    """True if two convex polygons have interior overlap."""
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            d1 = v1 @ normal
            d2 = v2 @ normal
            if d1.max() < d2.min() + tol or d2.max() < d1.min() + tol:
                return False
    return True

def _min_distance(v1, v2):
    """Approximate minimum distance between two convex polygons using vertex-edge distances."""
    min_dist = float('inf')
    for v in v1:
        for i in range(len(v2)):
            a = v2[i]
            b = v2[(i + 1) % len(v2)]
            # Distance from point v to segment ab
            ab = b - a
            t = np.dot(v - a, ab) / (np.dot(ab, ab) + 1e-15)
            t = max(0.0, min(1.0, t))
            closest = a + t * ab
            d = np.linalg.norm(v - closest)
            if d < min_dist:
                min_dist = d
    for v in v2:
        for i in range(len(v1)):
            a = v1[i]
            b = v1[(i + 1) % len(v1)]
            ab = b - a
            t = np.dot(v - a, ab) / (np.dot(ab, ab) + 1e-15)
            t = max(0.0, min(1.0, t))
            closest = a + t * ab
            d = np.linalg.norm(v - closest)
            if d < min_dist:
                min_dist = d
    return min_dist

def _objective(params, outer_side, outer_center, outer_angle):
    """Objective: maximize minimum violation (we minimize negative of this).
    Violations: 
    1. Inner hex vertices outside outer hex
    2. Overlapping inner hexes (penalty based on overlap depth)
    """
    ox, oy = outer_center
    outer_v = _outer_verts(ox, oy, outer_side, outer_angle)
    
    # Decode params: for each hex, (x, y, angle_deg)
    hexes = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        ang = params[3*i+2]
        hexes.append(_hex_verts(x, y, ang, 1.0))
    
    max_violation = 0.0
    
    # Check each inner hex is inside outer
    for hv in hexes:
        for v in hv:
            if not _point_in_convex(v[0], v[1], outer_v, tol=1e-9):
                # Compute how far outside (approximate)
                # Find minimum distance to outer boundary
                min_d = float('inf')
                for i in range(6):
                    a = outer_v[i]
                    b = outer_v[(i+1) % 6]
                    ab = b - a
                    t = np.dot(v - a, ab) / (np.dot(ab, ab) + 1e-15)
                    t = max(0.0, min(1.0, t))
                    closest = a + t * ab
                    d = np.linalg.norm(v - closest)
                    if d < min_d:
                        min_d = d
                # If outside, distance is positive; we want to penalize
                max_violation = max(max_violation, min_d + 0.1)
    
    # Check pairwise overlaps
    for i in range(N):
        for j in range(i+1, N):
            if _sat_overlap(hexes[i], hexes[j], tol=1e-9):
                # Compute overlap depth (approximate)
                d = _min_distance(hexes[i], hexes[j])
                # If overlapping, distance is 0 or negative; use a penalty
                max_violation = max(max_violation, 0.5)
    
    return max_violation

def _fit(outer_side, outer_center, outer_angle, max_time=10.0):
    """Try to find valid placement of 12 unit hexagons."""
    start = time.time()
    
    # Initial guess: honeycomb-like arrangement
    # For outer side S, inradius = S * sqrt(3)/2
    inradius = outer_side * SQRT3 / 2.0
    
    # Try to arrange hexagons in a pattern
    # A regular hexagonal grid of side 1 hexagons:
    # Distance between centers in honeycomb = 2 * inradius = sqrt(3) for touching
    
    # Let's try a 2x3 or 3x2 arrangement, or a hexagonal cluster
    
    # Strategy: start with a symmetric arrangement and optimize
    
    # Honeycomb positions for 12 hexagons:
    # Center at origin, then 6 at distance sqrt(3), then 6 more at distance 2*sqrt(3) but offset
    # Actually for flat-top hexagons touching, center-to-center = sqrt(3)
    
    # Let's use a simple grid: 3 columns x 4 rows, but hexagonal offset
    # Or better: use differential evolution
    
    # Bounds for positions
    # The hexagons must be inside the outer hex
    # Max extent: outer_side in x, outer_side*sqrt(3)/2 in y (for flat-top)
    # But centers must be at least 1.0 (circumradius) from boundary
    # So x in [-outer_side+1, outer_side-1], y in [-outer_side*sqrt(3)/2+1, outer_side*sqrt(3)/2-1]
    
    x_min = -outer_side + 1.2
    x_max = outer_side - 1.2
    y_min = -outer_side * SQRT3/2 + 1.2
    y_max = outer_side * SQRT3/2 - 1.2
    
    # Angle bounds: 0 to 180 (hexagon has 60-degree symmetry, so 0-60 is enough, but let's use 0-180)
    ang_min = 0.0
    ang_max = 180.0
    
    bounds = []
    for i in range(N):
        bounds.append((x_min, x_max))
        bounds.append((y_min, y_max))
        bounds.append((ang_min, ang_max))
    
    # Initial population: try a few symmetric arrangements
    
    # Arrangement 1: 2 rows of 6
    def make_arrangement_2x6():
        params = np.zeros(3*N)
        # 2 rows, 6 columns
        # Row spacing: sqrt(3) (for touching flat-top hexagons)
        # Column spacing: 1.5 (horizontal distance for touching)
        # Actually for flat-top hexagons:
        # Width = 2, height = sqrt(3)
        # Touching horizontally: center distance = 2 (side to side)
        # Wait, for flat-top, the width is 2*side = 2, height is sqrt(3)*side = sqrt(3)
        # So horizontal touch: dx = 2, vertical touch: dy = sqrt(3)
        # But for honeycomb, offset rows: dx = 1, dy = sqrt(3)/2
        
        # Let's just place them in a grid
        dx = 2.1  # slight gap
        dy = SQRT3 + 0.1
        for i in range(6):
            for j in range(2):
                idx = 3*(i*2+j)
                params[idx] = (i - 2.5) * dx
                params[idx+1] = (j - 0.5) * dy
                params[idx+2] = 0.0
        return params
    
    # Arrangement 2: 3x4 grid
    def make_arrangement_3x4():
        params = np.zeros(3*N)
        dx = 2.1
        dy = SQRT3 + 0.1
        for i in range(4):
            for j in range(3):
                idx = 3*(i*3+j)
                params[idx] = (i - 1.5) * dx
                params[idx+1] = (j - 1.0) * dy
                params[idx+2] = 0.0
        return params
    
    # Arrangement 3: hexagonal cluster (1 center + 6 around + 5 more)
    def make_arrangement_hex():
        params = np.zeros(3*N)
        # Center
        params[0] = 0.0
        params[1] = 0.0
        params[2] = 0.0
        # 6 around at distance 2.1
        for i in range(6):
            theta = i * math.pi / 3.0
            idx = 3*(i+1)
            params[idx] = 2.1 * math.cos(theta)
            params[idx+1] = 2.1 * math.sin(theta)
            params[idx+2] = 0.0
        # 5 more at distance 4.0, angles 30, 90, 150, 210, 270
        for i in range(5):
            theta = math.radians(30 + i*60)
            idx = 3*(i+7)
            params[idx] = 4.0 * math.cos(theta)
            params[idx+1] = 4.0 * math.sin(theta)
            params[idx+2] = 0.0
        return params
    
    # Try differential evolution
    best_params = None
    best_obj = float('inf')
    
    # Try a few initial arrangements to see if any are already good
    for init_func in [make_arrangement_2x6, make_arrangement_3x4, make_arrangement_hex]:
        p = init_func()
        obj = _objective(p, outer_side, outer_center, outer_angle)
        if obj < 1e-6:
            return p
        if obj < best_obj:
            best_obj = obj
            best_params = p
    
    # If none are good, use differential evolution
    if time.time() - start < max_time - 2.0:
        # Use differential evolution
        def obj_func(p):
            return _objective(p, outer_side, outer_center, outer_angle)
        
        # Use the best initial as x0
        x0 = best_params.copy()
        
        try:
            result = differential_evolution(
                obj_func, bounds, x0=x0,
                maxiter=50, popsize=20,
                tol=1e-4, mutation=(0.5, 1.0),
                recombination=0.7, seed=42,
                polish=False,
                updating='deferred',
                callback=None,
                workers=1
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Check if we found a valid solution
    if best_params is not None and best_obj < 1e-6:
        return best_params
    
    # Try local refinement with Nelder-Mead if we're close
    if best_params is not None and best_obj < 0.5 and time.time() - start < max_time - 1.0:
        from scipy.optimize import minimize
        try:
            result = minimize(
                lambda p: _objective(p, outer_side, outer_center, outer_angle),
                best_params, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is not None and best_obj < 1e-6:
        return best_params
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Bisection on outer side length
    # Lower bound from area: S >= sqrt(6) ≈ 2.45, but packing needs more
    # Let's start with lo=3.0, hi=10.0
    lo = 3.0
    hi = 10.0
    
    best_side = hi
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # First find a feasible upper bound
    for test_side in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
        if time.time() > deadline - 20.0:
            break
        params = _fit(test_side, (0.0, 0.0), 0.0, max_time=15.0)
        if params is not None:
            best_side = test_side
            best_params = params
            hi = test_side
            break
    
    if best_params is None:
        # Fallback: line arrangement
        inner = []
        for k in range(N):
            inner.append([3.0 * k, 0.0, 0.0])
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Bisection to find minimum
    for _ in range(15):
        if time.time() > deadline - 10.0:
            break
        mid = (lo + hi) / 2.0
        params = _fit(mid, (0.0, 0.0), 0.0, max_time=10.0)
        if params is not None:
            hi = mid
            best_side = mid
            best_params = params
        else:
            lo = mid
    
    # Decode best_params to inner list
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([x, y, ang])
    
    return inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
