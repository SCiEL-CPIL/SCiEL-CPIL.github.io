"""Improved packing: compact hexagonal cluster with optimal enclosing hexagon."""
import math
import time
import numpy as np

N = 11
_START = time.time()
DEADLINE = 150.0
MARGIN = 20.0

def _min_enclosing_hexagon_side(points, outer_rot_deg):
    """Min side s of regular hexagon (centered at origin, circumradius s) with given
    orientation containing all points. Hexagon = intersection of 6 half-planes,
    inradius = s*sqrt(3)/2. Face normals at outer_rot + 30 + 60k."""
    th = math.radians(outer_rot_deg)
    normals = np.array([[math.cos(th + math.radians(30 + 60 * k)),
                         math.sin(th + math.radians(30 + 60 * k))] for k in range(6)])
    proj = points @ normals.T
    max_proj = proj.max()
    return max_proj / (math.sqrt(3) / 2.0)

def _cluster_positions():
    """11 non-overlapping unit-hexagon centers: center + 6 ring1 + 4 of ring2.
    Neighbor spacing d = sqrt(3). Ring1 at distance d in 6 directions.
    Ring2 chosen at distance 2d in directions 0,60,120,180 (4 cells), each
    edge-adjacent to a ring1 cell, no overlaps."""
    d = math.sqrt(3)
    pts = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(60 * k)
        pts.append((d * math.cos(a), d * math.sin(a)))
    for k in range(4):
        a = math.radians(60 * k)
        pts.append((2 * d * math.cos(a), 2 * d * math.sin(a)))
    return pts

def construct():
    global _START, DEADLINE, MARGIN
    if time.time() - _START > DEADLINE - MARGIN:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    pts = np.array(_cluster_positions())
    best_score = float('inf')
    best_rot = 0.0

    # Coarse scan of cluster rotation vs fixed outer orientation (relative angle)
    for cluster_rot in range(0, 60, 1):
        if time.time() - _START > DEADLINE - MARGIN:
            break
        th = math.radians(cluster_rot)
        c, s_ = math.cos(th), math.sin(th)
        R = np.array([[c, -s_], [s_, c]])
        rot_pts = pts @ R.T
        side = _min_enclosing_hexagon_side(rot_pts, 0.0)
        if side < best_score:
            best_score = side
            best_rot = cluster_rot

    # Fine scan 0.1 deg over full range (600 iters, cheap)
    for cluster_rot in np.arange(0, 60, 0.1):
        if time.time() - _START > DEADLINE - MARGIN:
            break
        th = math.radians(cluster_rot)
        c, s_ = math.cos(th), math.sin(th)
        R = np.array([[c, -s_], [s_, c]])
        rot_pts = pts @ R.T
        side = _min_enclosing_hexagon_side(rot_pts, 0.0)
        if side < best_score:
            best_score = side
            best_rot = cluster_rot

    # Build final inner list
    th = math.radians(best_rot)
    c, s_ = math.cos(th), math.sin(th)
    R = np.array([[c, -s_], [s_, c]])
    rot_pts = pts @ R.T
    inner = [[float(rot_pts[i][0]), float(rot_pts[i][1]), float(best_rot)] for i in range(N)]
    return inner, (0.0, 0.0), float(best_score), 0.0
