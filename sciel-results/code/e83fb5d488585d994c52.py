import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side s and rotation angle."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append([cx + s * np.cos(a), cy + s * np.sin(a)])
    return np.array(verts)

def _project_polygon(vertices, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    dots = vertices @ axis
    return dots.min(), dots.max()

def _polygons_overlap_sat(v1, v2):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (including touching)."""
    polys = [v1, v2]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            min1, max1 = _project_polygon(v1, normal)
            min2, max2 = _project_polygon(v2, normal)
            if max1 < min2 or max2 < min1:
                return False  # Found separating axis
    return True  # No separating axis found -> overlap

def _check_overlap(inner_verts, outer_verts, margin=1e-6):
    """Check if inner hexagon overlaps with outer hexagon boundary (inner must be fully inside outer)."""
    # For inner to be inside outer, no edge of inner should cross any edge of outer,
    # and at least one vertex of inner should be inside outer.
    # Simpler: check if any vertex of inner is outside outer, or any edge intersects.
    # Use SAT-like: if they overlap as polygons, and inner is not fully inside, there's a problem.
    # Actually, we need inner hexagon to be entirely INSIDE outer hexagon.
    # Check: all vertices of inner must be inside or on boundary of outer.
    # And no edges should cross (but if all vertices inside, for convex polygons, it's inside).
    
    # Point-in-convex-polygon test for each vertex of inner
    def point_in_convex_polygon(pt, poly):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            cross = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
            if cross < -margin:
                return False
        return True
    
    for v in inner_verts:
        if not point_in_convex_polygon(v, outer_verts):
            return True  # Overlap (inner not fully inside)
    return False

def _inner_overlap(v1, v2):
    """Check if two inner hexagons overlap. Return True if they overlap."""
    # Use SAT
    return _polygons_overlap_sat(v1, v2)

def _evaluate(params):
    """Given parameters, compute outer side length or return inf if invalid."""
    # params: for each inner hex: x, y, angle_deg; then outer: cx, cy, side, angle_deg
    n_inner = N
    # Parse
    inner_params = params[:n_inner * 3]
    outer_params = params[n_inner * 3:]
    outer_cx, outer_cy, outer_side, outer_angle = outer_params
    
    if outer_side < 1.0:
        return np.inf
    
    outer_verts = _hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    inner_verts_list = []
    for i in range(n_inner):
        x = inner_params[i * 3]
        y = inner_params[i * 3 + 1]
        ang = inner_params[i * 3 + 2]
        inner_verts_list.append(_hex_vertices(x, y, 1.0, ang))
    
    # Check inner-inner overlaps
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            if _inner_overlap(inner_verts_list[i], inner_verts_list[j]):
                return np.inf
    
    # Check inner-outer containment
    for i in range(n_inner):
        if _check_overlap(inner_verts_list[i], outer_verts):
            return np.inf
    
    return outer_side

def _initial_guess():
    """Generate initial guess based on hexagonal grid."""
    # Hexagonal grid with spacing sqrt(3) for flat-to-flat
    d = np.sqrt(3)
    
    # Ring 0: 1 hex
    # Ring 1: 6 hexes
    # Ring 2: 12 hexes, use 4 of them
    
    positions = []
    angles = [0.0] * N
    
    # Ring 0
    positions.append((0.0, 0.0))
    
    # Ring 1: 6 positions at distance d
    for i in range(6):
        a = i * np.pi / 3
        positions.append((d * np.cos(a), d * np.sin(a)))
    
    # Ring 2: 12 positions at distance 2*d, use 4 (every other one)
    for i in range(0, 12, 3):  # 4 positions
        a = i * np.pi / 6
        positions.append((2 * d * np.cos(a), 2 * d * np.sin(a)))
    
    # That's 1 + 6 + 4 = 11
    assert len(positions) == N
    
    # Outer hexagon: needs to contain all
    # Max distance from center: 2*d = 2*sqrt(3) ≈ 3.464
    # For a hexagon with side s, the distance to a vertex is s, to a flat is s*sqrt(3)/2
    # We need s such that all inner hex vertices are inside
    # Rough estimate: s ≈ 2*d + 1 = 2*sqrt(3) + 1 ≈ 4.464, but need to account for hex shape
    # Let's use s = 5.0 as initial
    
    params = []
    for (x, y), ang in zip(positions, angles):
        params.extend([x, y, ang])
    
    # Outer: center (0,0), side 5.0, angle 0
    params.extend([0.0, 0.0, 5.0, 0.0])
    
    return np.array(params)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_params = _initial_guess()
    best_score = _evaluate(best_params)
    
    if np.isinf(best_score):
        # Try a larger outer
        best_params[3*N + 2] = 6.0
        best_score = _evaluate(best_params)
    
    # Use Nelder-Mead optimization
    from scipy.optimize import minimize
    
    def objective(params):
        s = _evaluate(params)
        if np.isinf(s):
            return 1e6
        return s
    
    # Try multiple random restarts
    best_score_final = best_score
    best_params_final = best_params.copy()
    
    n_restarts = 5
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = best_params.copy()
        else:
            # Perturb initial guess
            x0 = _initial_guess()
            # Add random perturbation
            perturbation = rng.normal(0, 0.5, size=x0.size)
            # Don't perturb outer side too much
            perturbation[3*N + 2] = rng.normal(0, 1.0)
            x0 = x0 + perturbation
        
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if time.time() > deadline:
            break
        
        score = objective(result.x)
        if score < best_score_final:
            best_score_final = score
            best_params_final = result.x.copy()
    
    # Extract result
    inner = []
    for i in range(N):
        x = best_params_final[i * 3]
        y = best_params_final[i * 3 + 1]
        ang = best_params_final[i * 3 + 2]
        inner.append([x, y, ang])
    
    outer_cx = best_params_final[3*N]
    outer_cy = best_params_final[3*N + 1]
    outer_side = best_params_final[3*N + 2]
    outer_angle = best_params_final[3*N + 3]
    
    # Ensure side is at least 1
    outer_side = max(outer_side, 1.0)
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle

