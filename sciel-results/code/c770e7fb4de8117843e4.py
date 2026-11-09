"""Packing 12 unit hexagons in a minimal regular hexagon."""

import math
import time

N = 12

# EVOLVE-BLOCK-START

def _support_hex(theta):
    """Support function of a unit regular hexagon in direction theta.
    Vertices at angles 0, 60, 120, 180, 240, 300 degrees, radius 1.
    h(theta) = max_k cos(theta - k*pi/3)"""
    best = -2.0
    for k in range(6):
        v = math.cos(theta - k * math.pi / 3.0)
        if v > best:
            best = v
    return best


def _min_L_for_centers(centers, n_phi_steps=720):
    """Minimize L over outer rotation phi in [0, 60)."""
    best_L = float('inf')
    best_phi = 0.0
    for i in range(n_phi_steps):
        phi = (i / n_phi_steps) * (math.pi / 3.0)
        L = 0.0
        for k in range(3):
            ang = phi + k * math.pi / 3.0
            h = _support_hex(ang)
            # max over centers of |dot(c, n_k)|
            ax = math.cos(ang)
            ay = math.sin(ang)
            m = 0.0
            for (cx, cy) in centers:
                v = abs(cx * ax + cy * ay)
                if v > m:
                    m = v
            cand = h + m
            if cand > L:
                L = cand
        if L < best_L:
            best_L = L
            best_phi = phi
    return best_L, best_phi


def _hex_lattice_points(n, spacing=2.0):
    """Generate hexagonal lattice points, return n closest to origin."""
    pts = []
    ax = spacing
    ay = 0.0
    bx = spacing * 0.5
    by = spacing * math.sqrt(3) / 2.0
    R = 6
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            x = i * ax + j * bx
            y = i * ay + j * by
            pts.append((x, y))
    pts.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
    return pts[:n]


def _check_overlap(centers, eps=1e-9):
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            if math.hypot(dx, dy) < 2.0 - eps:
                return False
    return True


def _try_arrangement(centers):
    if not _check_overlap(centers):
        return None, None
    L, phi = _min_L_for_centers(centers)
    return L, phi


def construct():
    t0 = time.time()
    deadline = 120.0

    best_L = float('inf')
    best_centers = None
    best_phi = 0.0

    all_pts = _hex_lattice_points(200)

    # Candidate: 12 closest lattice points
    cand = all_pts[:12]
    L, phi = _try_arrangement(cand)
    if L is not None and L < best_L:
        best_L = L
        best_centers = list(cand)
        best_phi = phi

    # Try local swaps
    if best_centers is not None and time.time() - t0 < deadline - 20:
        base = list(best_centers)
        improved = True
        while improved and time.time() - t0 < deadline - 20:
            improved = False
            for idx in range(len(base)):
                for new_pt in all_pts[12:40]:
                    if new_pt in base:
                        continue
                    trial = list(base)
                    trial[idx] = new_pt
                    L2, phi2 = _try_arrangement(trial)
                    if L2 is not None and L2 < best_L - 1e-9:
                        best_L = L2
                        best_centers = trial
                        best_phi = phi2
                        base = list(trial)
                        improved = True
                        break
                if improved:
                    break

    # Fine-tune phi
    if best_centers is not None:
        L, phi = _min_L_for_centers(best_centers, n_phi_steps=7200)
        best_L = L
        best_phi = phi

    if best_centers is None:
        centers = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        best_L, best_phi = 40.0, 0.0
        best_centers = centers

    inner = [[cx, cy, 0.0] for (cx, cy) in best_centers]
    return inner, (0.0, 0.0), best_L, math.degrees(best_phi)

# EVOLVE-BLOCK-END
