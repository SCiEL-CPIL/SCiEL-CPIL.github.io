"""Packing 11 unit regular hexagons inside a regular hexagon.

Strategy: start from a hexagonal lattice arrangement (radius 2 → side length 4),
then use Nelder-Mead local optimization to push the inner hexagons into a
tighter, non-overlapping cluster while minimizing the enclosing hexagon's
side length.  A SAT-based validity check (separating axis test for convex
polygons) ensures the configuration is geometrically feasible.
"""

import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 140.0  # seconds wall-clock safety margin before 150 s kill
EPS = 1e-9


def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon (CCW)."""
    a = math.radians(angle_deg)
    # A regular hexagon with side length s has circumradius s.
    # Vertices at angles a + k*pi/3, k=0..5
    vs = []
    for k in range(6):
        th = a + k * math.pi / 3.0
        vs.append((cx + side * math.cos(th), cy + side * math.sin(th)))
    return np.array(vs)


def edges_from_vertices(vs):
    """Return list of edge vectors for a polygon given as vertex array."""
    n = len(vs)
    return [vs[(k + 1) % n] - vs[k] for k in range(n)]


def proj_range_on_axis(vs, axis):
    """Project polygon vertices onto axis; return (min, max)."""
    dots = vs @ axis
    return dots.min(), dots.max()


def polygons_overlap(v1, v2, eps=1e-5):
    """SAT: check if two convex polygons overlap (including touching)."""
    edges1 = edges_from_vertices(v1)
    edges2 = edges_from_vertices(v2)
    for edge in edges1 + edges2:
        # Normal perpendicular to edge
        n = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(n)
        if norm < EPS:
            continue
        n = n / norm
        r1 = proj_range_on_axis(v1, n)
        r2 = proj_range_on_axis(v2, n)
        if r1[1] < r2[0] - eps or r2[1] < r1[0] - eps:
            return False
    return True


def point_in_convex_polygon(pt, vs):
    """Check if point pt is inside convex polygon vs (CCW)."""
    n = len(vs)
    for k in range(n):
        v1 = vs[k]
        v2 = vs[(k + 1) % n]
        cross = (v2[0] - v1[0]) * (pt[1] - v1[1]) - (v2[1] - v1[1]) * (pt[0] - v1[0])
        if cross < -EPS:
            return False
    return True


def min_enclosing_hex_side(positions, angles_deg):
    """
    Compute the minimum side length of a regular hexagon (centered at origin,
    axis-aligned) that contains all inner hexagons.
    For each inner hexagon vertex, find the maximum over the 6 half-plane
    constraints of the outer hexagon.
    Outer hexagon (axis-aligned, side S) has half-widths:
      -x: S*cos(30°) = S*sqrt(3)/2
      +x: S*sqrt(3)/2
      -y: S
      +y: S
      Actually for a regular hexagon with flat-top orientation (vertex at top),
      the 6 supporting lines are at angles 0,60,120,180,240,300 degrees.
    For a regular hexagon centered at origin with side length S and one vertex
    pointing up (angle 90°), the support function h(theta) = S * cos(theta - 90°)
    ... this gets complicated.  Simpler: the outer hexagon has 6 sides.  For each
    side (with outward normal n_k and offset d_k from center), the constraint is
    n_k · v <= d_k for all vertices v.
    For an axis-aligned regular hexagon (flat sides on left/right), the 6 normals
    point at 0°, 60°, 120°, 180°, 240°, 300° and the offset is S * cos(30°) = S*sqrt(3)/2
    for the left/right normals and S for the top/bottom normals... 
    
    Let me just use the standard: a regular hexagon with side S, centered at origin,
    with vertices at angles 0,60,...,300 degrees, has the property that the
    distance from center to each side (apothem) is S * sqrt(3) / 2.
    The 6 outward normals point at 30°, 90°, 150°, 210°, 270°, 330°.
    Constraint: (v · n_k) <= apothem = S * sqrt(3) / 2  for all vertices v, all k.
    So S >= max over all inner-hex vertices v, over all 6 normals n_k of (v · n_k) / (sqrt(3)/2).
    """
    sqrt3 = math.sqrt(3.0)
    apothem_factor = sqrt3 / 2.0  # S * apothem_factor = S * sqrt(3)/2
    # 6 outward normals for a regular hexagon with vertex at angle 0
    # The sides are between consecutive vertices.  Side between vertex at 0° and 60°
    # has outward normal at 30°.
    normals = np.array([[math.cos(math.radians(30 + 60 * k)),
                         math.sin(math.radians(30 + 60 * k))] for k in range(6)])
    max_val = 0.0
    for i in range(N):
        cx, cy = positions[i]
        vs = hex_vertices(cx, cy, 1.0, angles_deg[i])
        for v in vs:
            dots = vs @ normals.T
            m = dots.max()
            if m > max_val:
                max_val = m
    S = max_val / apothem_factor
    return S


def config_valid(positions, angles_deg, eps=1e-4):
    """Check that all inner hexagons are pairwise non-overlapping."""
    verts = [hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if polygons_overlap(verts[i], verts[j], eps):
                return False
    return True


def objective(x):
    """Objective: enclosing hexagon side length + penalty for overlap."""
    positions = x[:2 * N].reshape(N, 2)
    angles_deg = x[2 * N:]
    side = min_enclosing_hex_side(positions, angles_deg)
    # Penalty for overlaps: count overlapping pairs * large weight
    verts = [hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i]) for i in range(N)]
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if polygons_overlap(verts[i], verts[j], eps=1e-5):
                penalty += 50.0
    return side + penalty


def best_known_side():
    """Return the side length of the initial hexagonal lattice packing."""
    # Hexagonal lattice with 2 rings around center: 1 + 6 + 12 = 19, too many.
    # For 11 hexagons: center + ring of 6 + 4 more.
    # Let's just use a known good starting configuration.
    # 11 hexagons in a hexagonal pattern:
    # Row 0 (top): 2
    # Row 1: 3
    # Row 2 (middle): 3
    # Row 3: 2
    # Row 4 (bottom): 1
    # Actually, let's use the simple arrangement from the seed: 11 in a row.
    # That gives side length 40 (way too big).
    # Better: hexagonal cluster.
    # For 11 unit hexagons, a good packing is roughly side length ~4-5.
    # Let's return 6.0 as a safe upper bound for initialization.
    return 6.0


def initial_config():
    """Hexagonal lattice initial configuration for 11 unit hexagons."""
    # Arrange in a hexagonal pattern:
    # Center at origin, ring 1 (6 hexagons), ring 2 (need 4 more)
    # Hexagonal lattice spacing: distance between centers of adjacent hexagons = 2*apothem = sqrt(3)
    d = math.sqrt(3.0)  # center-to-center distance for touching hexagons
    positions = []
    angles = [0.0] * N

    # Center hexagon
    positions.append((0.0, 0.0))
    # Ring 1: 6 hexagons at distance d
    for k in range(6):
        th = math.radians(60 * k)
        positions.append((d * math.cos(th), d * math.sin(th)))
    # Ring 2: need 4 more. Place them at distance 2*d at angles 0, 60, 120, 180
    # But we need to avoid overlaps with ring 1.
    # Ring 2 hexagons at 2*d: they touch ring 1 hexagons.
    # Place 4 at angles 30, 90, 150, 210 (between ring-1 hexagons)
    for th_deg in [30, 90, 150, 210]:
        th = math.radians(th_deg)
        # Distance from center: 2*d / sin(30°) = 2*d / 0.5 = 4*d?  No.
        # In hexagonal lattice, ring 2 is at distance 2*d from center along the
        # directions of ring 1, or at distance d*sqrt(3) along the bisectors.
        # Let's just place them at 2*d along the 6 main directions, pick 4.
        positions.append((2.0 * d * math.cos(th), 2.0 * d * math.sin(th)))

    return np.array(positions), angles


def run_nm(x0, deadline, max_iters=2000):
    """Run Nelder-Mead from x0 until deadline."""
    start = time.time()
    res = minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxiter': max_iters, 'xatol': 1e-5, 'fatol': 1e-5,
                 'adaptive': True}
    )
    return res.x, res.fun


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    # Initial configuration
    pos0, ang0 = initial_config()
    x0 = np.concatenate([pos0.ravel(), np.array(ang0)])

    best_x = x0.copy()
    best_side = best_known_side()

    # Evaluate initial
    s0 = min_enclosing_hex_side(pos0, ang0)
    if config_valid(pos0, ang0):
        best_side = s0
        best_x = x0.copy()

    # Try multiple NM starts: initial + perturbed
    rng = np.random.default_rng(42)
    n_starts = 15
    for s in range(n_starts):
        if time.time() >= deadline - 10:
            break
        if s == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + rng.normal(0, 0.3, size=x0.shape)
            # Keep angles in [0, 60)
            x_start[2 * N:] = np.mod(x_start[2 * N:], 60.0)

        try:
            x_best, f_best = run_nm(x_start, deadline, max_iters=1500)
        except Exception:
            continue

        positions = x_best[:2 * N].reshape(N, 2)
        angles = x_best[2 * N:]
        if config_valid(positions, angles):
            side = min_enclosing_hex_side(positions, angles)
            if side < best_side - 1e-6:
                best_side = side
                best_x = x_best.copy()

    # Final: if we still have time, do a few more refined starts near best
    if time.time() < deadline - 5:
        for s in range(5):
            if time.time() >= deadline - 5:
                break
            x_start = best_x + rng.normal(0, 0.05, size=best_x.shape)
            x_start[2 * N:] = np.mod(x_start[2 * N:], 60.0)
            try:
                x_best, f_best = run_nm(x_start, deadline, max_iters=1000)
            except Exception:
                continue
            positions = x_best[:2 * N].reshape(N, 2)
            angles = x_best[2 * N:]
            if config_valid(positions, angles):
                side = min_enclosing_hex_side(positions, angles)
                if side < best_side - 1e-6:
                    best_side = side
                    best_x = x_best.copy()

    positions = best_x[:2 * N].reshape(N, 2)
    angles = best_x[2 * N:]

    inner = [[float(positions[i, 0]), float(positions[i, 1]), float(angles[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0


# EVOLVE-BLOCK-START
# (construct is defined above; keep the marker for the evolution harness)
# EVOLVE-BLOCK-END
