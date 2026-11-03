"""Rewritten program: hexagonal lattice packing with rotation optimization and safety margins."""
import time
import math
import numpy as np

_T0 = time.time()
DEADLINE = 150.0
MARGIN = 5.0

def _remaining():
    return DEADLINE - (time.time() - _T0) - MARGIN

def _min_enclosing_side(centers, cx, cy, theta):
    """Min side of regular hexagon (center, rotation theta rad) containing all unit-hex centers,
    with the unit hexagons having inradius sqrt(3)/2. Returns S such that inradius = S*sqrt(3)/2
    covers max projection."""
    normals = np.array([[math.cos(theta + math.radians(30 + 60 * k)),
                         math.sin(theta + math.radians(30 + 60 * k))] for k in range(6)])
    rel = centers - np.array([cx, cy])
    proj = rel @ normals.T
    max_proj = proj.max(axis=0).max()
    return (2.0 / math.sqrt(3.0)) * max_proj

def _lattice_positions(r_max, scale):
    pts = []
    for q in range(-r_max, r_max + 1):
        for r in range(-r_max, r_max + 1):
            if max(abs(q), abs(r), abs(q + r)) <= r_max:
                x = math.sqrt(3) * scale * (q + r / 2.0)
                y = 1.5 * scale * r
                pts.append((x, y))
    return np.array(pts)

def construct():
    n = 11
    GAP = 1.0 + 1e-4   # lattice scale to ensure disjoint (no touching)
    SAFETY = 1.0 + 1e-3 # extra on outer side for containment margin

    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_theta = 0.0

    for r_max in [1, 2, 3]:
        pts = _lattice_positions(r_max, GAP)
        d2 = (pts[:, 0] ** 2 + pts[:, 1] ** 2)
        order = np.argsort(d2)
        pts_sorted = pts[order]
        if len(pts_sorted) < n:
            continue
        sel = pts_sorted[:n]
        c = sel.mean(axis=0)
        for theta_deg in [0, 30, 60, 15, 45, 75, 10, 20, 35, 50, 25, 55]:
            if _remaining() <= 0:
                break
            theta = math.radians(theta_deg)
            S = _min_enclosing_side(sel, c[0], c[1], theta) * SAFETY
            if S < best_S:
                best_S = S
                best_inner = sel
                best_center = (float(c[0]), float(c[1]))
                best_theta = theta

    # Fallback straight line
    line = np.array([[k * math.sqrt(3) * GAP, 0.0] for k in range(n)])
    c = line.mean(axis=0)
    for theta_deg in [0, 30, 90]:
        if _remaining() <= 0:
            break
        theta = math.radians(theta_deg)
        S = _min_enclosing_side(line, c[0], c[1], theta) * SAFETY
        if S < best_S:
            best_S = S
            best_inner = line
            best_center = (float(c[0]), float(c[1]))
            best_theta = theta

    inner = [[float(p[0]), float(p[1]), 0.0] for p in best_inner]
    return inner, best_center, float(best_S), float(math.degrees(best_theta))
