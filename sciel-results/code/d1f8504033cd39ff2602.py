"""Rewritten program: optimize 12 unit hexagons in a regular hexagon container."""
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize, bounds as scipy_bounds

N = 12
DEADLINE = 120  # seconds, safety margin before 150s kill

import time
_T0 = time.time()

def _deadline():
    return time.time() - _T0 > DEADLINE

def _hex_vertices(center, angle_deg=0.0, side=1.0):
    """Return 6 vertices of a regular hexagon (flat-top by default, i.e., vertex at 0 deg)."""
    cx, cy = center
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + math.radians(60 * k)
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def _poly_area(crosses):
    """Compute twice the signed area of a polygon given as list of vertices."""
    area = 0.0
    n = len(crosses)
    for i in range(n):
        x1, y1 = crosses[i]
        x2, y2 = crosses[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0

def _convex_hull(points):
    """Andrew's monotone chain convex hull. Returns list of points (CCW)."""
    points = sorted(set([(p[0], p[1]) for p in points]))
    if len(points) <= 2:
        return points
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _min_hex_containing(conv_hull_pts, angle_deg):
    """
    Given convex hull points and an outer hexagon rotation, compute the minimum
    side length of a regular hexagon (at that rotation) that contains the convex hull.
    We center the hexagon at the centroid of the convex hull for symmetry.
    """
    if len(conv_hull_pts) < 3:
        return 0.0
    centroid_x = sum(p[0] for p in conv_hull_pts) / len(conv_hull_pts)
    centroid_y = sum(p[1] for p in conv_hull_pts) / len(conv_hull_pts)
    centroid = (centroid_x, centroid_y)
    a = math.radians(angle_deg)
    # The six supporting directions for a flat-top hexagon rotated by a:
    # Directions to vertices are at a + 0, 60, 120, 180, 240, 300
    # Supporting lines (normals) are perpendicular to edges.
    # Edge normals for a flat-top hexagon (vertex at angle 0) are at 30, 90, 150, 210, 270, 330
    # After rotation by a: normals at a + 30 + 60*k
    max_proj = 0.0
    for k in range(6):
        theta = a + math.radians(30 + 60 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        # project all hull points onto normal, compute range
        projs = [p[0] * nx + p[1] * ny for p in conv_hull_pts]
        # The range of projections gives the width in that direction
        width = max(projs) - min(projs)
        # For a regular hexagon of side s, the width in the normal-to-edge direction is sqrt(3)*s
        # So s_candidate = width / sqrt(3)
        s_cand = width / math.sqrt(3)
        if s_cand > max_proj:
            max_proj = s_cand
    return max_proj

def _objective(params):
    """
    params: 3N floats: x_1, y_1, angle_1 (deg), x_2, y_2, angle_2 (deg), ...
    Returns the minimum outer hexagon side length.
    """
    if _deadline():
        return 1000.0
    centers = []
    angles = []
    for i in range(N):
        x = params[3 * i]
        y = params[3 * i + 1]
        ang = params[3 * i + 2]
        centers.append((x, y))
        angles.append(ang)
    # Check pairwise overlap: minimum distance between centers must be >= 2 (side=1 hexagons, flat-to-flat)
    # Actually, for two regular hexagons of side 1, the minimum distance between centers
    # to avoid overlap depends on relative rotation. For a safe lower bound, use 2*sqrt(3)/2 = sqrt(3)
    # But let's compute more carefully. Two hexagons of side 1: the support function in direction u
    # is 1.5 * cos(angle diff in 60-deg bins) or so. For a simpler check, use a slightly larger bound.
    # For two regular hexagons of side s, minimum center distance = 2*s if aligned (edge-to-edge)
    # Actually, two flat-top hexagons of side 1: distance between parallel edges = sqrt(3)
    # So center distance >= sqrt(3) for same orientation.
    # With different rotations, it's more complex. Use a slightly larger threshold.
    # Let's use 1.8 as a safe minimum center distance for any orientation.
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1.8:
                return 100.0 + (1.8 - dist) * 10
    # Compute convex hull of all vertices
    all_pts = []
    for i in range(N):
        verts = _hex_vertices(centers[i], angles[i], 1.0)
        all_pts.extend(verts)
    hull = _convex_hull(all_pts)
    # Try several outer hexagon orientations to find the minimum side
    # The outer hexagon can be at any angle; we'll check a few and take the min
    best_S = 1000.0
    for outer_ang in range(0, 60, 5):
        S = _min_hex_containing(hull, float(outer_ang))
        if S < best_S:
            best_S = S
        if _deadline():
            break
    return best_S

def construct():
    t_start = time.time()
    
    # Strategy: use differential evolution to find good positions, then refine with Nelder-Mead.
    # Initial guess: hexagonal honeycomb pattern.
    # Honeycomb: centers at positions that form a hexagonal lattice, but with 12 points.
    # A compact arrangement: central point + 5 around + 6 more? 
    # Actually, for 12 hexagons, a good starting pattern is a hexagonal cluster.
    # Layer 0: 1 at center
    # Layer 1: 6 around
    # Layer 2: we need 5 more
    # Alternative: a rectangular-ish arrangement.
    
    # Let's use a honeycomb pattern centered at origin.
    # Hexagonal lattice points with axial coordinates (q, r), distance metric max(|q|, |r|, |q+r|)
    # Generate all points within radius 2 (which gives 19 points), pick 12 most central.
    
    def hex_to_pix(q, r):
        x = 1.5 * q
        y = math.sqrt(3) / 2 * q + math.sqrt(3) * r
        return (x, y)
    
    # Generate honeycomb points
    points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Hexagonal distance
            d = max(abs(q), abs(r), abs(q + r))
            points.append((d, q, r, hex_to_pix(q, r)))
    points.sort(key=lambda t: (t[0], t[1] ** 2 + t[2] ** 2))
    # Take the 12 most central
    honeycomb_pts = [p[3] for p in points[:12]]
    
    # Initial positions: honeycomb pattern, all angle 0
    x0 = []
    for (hx, hy) in honeycomb_pts:
        x0.append(hx)
        x0.append(hy)
        x0.append(0.0)
    x0 = np.array(x0)
    
    # Bounds for DE
    bounds = [(-5, 5)] * (3 * N)
    
    best_S = 1000.0
    best_params = x0.copy()
    
    # Run differential evolution
    if not _deadline():
        result = differential_evolution(
            _objective, bounds, maxiter=80, popsize=15, seed=42,
            tol=0.01, mutation=(0.3, 1.0), recombination=0.9
        )
        if result.fun < best_S:
            best_S = result.fun
            best_params = result.x
    
    # Refine with Nelder-Mead
    if not _deadline() and best_S < 50:
        try:
            result2 = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
            )
            if result2.fun < best_S:
                best_S = result2.fun
                best_params = result2.x
        except Exception:
            pass
    
    # Final refinement: local search with small perturbations
    if not _deadline() and best_S < 10:
        # Try to improve by fine-tuning each parameter
        best_S_cur = best_S
        best_p = best_params.copy()
        for _ in range(50):
            if _deadline():
                break
            improved = False
            for i in range(3 * N):
                for delta in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                    p_try = best_p.copy()
                    p_try[i] += delta
                    # Check bounds
                    if -5 < p_try[i] < 5:
                        obj = _objective(p_try)
                        if obj < best_S_cur:
                            best_S_cur = obj
                            best_p = p_try
                            improved = True
                            break
            if not improved:
                break
        best_S = best_S_cur
        best_params = best_p
    
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[3 * i]
        y = best_params[3 * i + 1]
        ang = best_params[3 * i + 2]
        inner.append([float(x), float(y), float(ang)])
    
    # Determine outer hexagon: center at centroid, side = best_S, angle = 0
    cx = sum(p[0] for p in inner) / N
    cy = sum(p[1] for p in inner) / N
    outer_center = (float(cx), float(cy))
    outer_side = float(best_S)
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
