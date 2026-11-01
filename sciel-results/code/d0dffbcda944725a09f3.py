"""Hybrid: analytical 6x2 grid baseline + SAT-verified local polishing.
Uses the current program's grid as a guaranteed-valid starting point, then
applies the inspiration's Separating Axis Theorem (SAT) validity checks to
safely minimize the outer side length via Nelder-Mead and coordinate descent."""

import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # safety margin before 150s wall-clock kill


# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()

    # --- Fast geometry helpers (NumPy-vectorized where possible) ---

    def hex_vertices(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        angles = a + np.array([0.0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3])
        return np.column_stack([cx + side * np.cos(angles),
                                cy + side * np.sin(angles)])

    def sat_disjoint(v1, v2, tol=1e-9):
        """Returns True if convex polygons v1, v2 (each Nx2) are disjoint."""
        # Build all edge normals (candidate separating axes)
        axes = []
        for v in (v1, v2):
            for i in range(len(v)):
                edge = v[(i + 1) % len(v)] - v[i]
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-12:
                    axes.append(n / norm)
        # Test each axis
        for ax in axes:
            proj1 = v1 @ ax
            proj2 = v2 @ ax
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return True
        return False

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx, dy = px - cx, py - cy
        a = math.radians(angle_deg)
        ca, sa = math.cos(-a), math.sin(-a)
        lx, ly = dx * ca - dy * sa, dx * sa + dy * ca
        apothem = side * math.sqrt(3) / 2
        for k in range(6):
            na = math.radians(30 + 60 * k)
            if lx * math.cos(na) + ly * math.sin(na) > apothem + 1e-9:
                return False
        return True

    def all_inside(inner, oc, os, oa):
        for x, y, a in inner:
            vs = hex_vertices(x, y, 1.0, a)
            for vx, vy in vs:
                if not point_in_hexagon(vx, vy, oc[0], oc[1], os, oa):
                    return False
        return True

    def pairwise_disjoint(inner):
        n = len(inner)
        for i in range(n):
            v1 = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            for j in range(i + 1, n):
                v2 = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                if not sat_disjoint(v1, v2):
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        return all_inside(inner, oc, os, oa) and pairwise_disjoint(inner)

    # --- Analytical baseline: 6 columns x 2 rows of flat-top unit hexagons ---
    # Flat-top unit hex: width 2, height sqrt(3).
    # Horizontal pitch 3 (flat edges touch), vertical pitch sqrt(3).
    cols, rows = 6, 2
    x0 = -(cols - 1) / 2.0 * 3.0          # -7.5
    y0 = -(rows - 1) / 2.0 * math.sqrt(3)  # -sqrt(3)/2

    inner = []
    for r in range(rows):
        for c in range(cols):
            inner.append([x0 + c * 3.0, y0 + r * math.sqrt(3), 0.0])

    # Flat-top outer hexagon: horizontal extent 2*S, vertical extent S*sqrt(3).
    # Grid width = 18 -> need 2*S >= 18 -> S >= 9.
    # Grid height = sqrt(3) -> need S*sqrt(3) >= sqrt(3) -> S >= 1.
    # So S=9 is the minimum for this grid orientation.
    oc = (0.0, 0.0)
    os = 9.0
    oa = 0.0

    # Verify baseline validity (should be True)
    if not is_valid(inner, oc, os, oa):
        # Fallback: expand slightly if numerical issues
        while not is_valid(inner, oc, os, oa) and os < 15.0:
            os += 0.01
        return inner, oc, os, oa

    best_inner = [list(p) for p in inner]
    best_oc = list(oc)
    best_os = os
    best_oa = oa

    # --- Local polishing: minimize outer side length via Nelder-Mead ---
    # Parameter vector: [x1,y1,a1, ..., x12,y12,a12, ox, oy, os, oa]
    def pack_params(inner, oc, os, oa):
        p = []
        for x, y, a in inner:
            p.extend([x, y, a])
        p.extend([oc[0], oc[1], os, oa])
        return np.array(p, dtype=float)

    def unpack_params(p):
        inner = [(p[3*i], p[3*i+1], p[3*i+2]) for i in range(N)]
        oc = (p[3*N], p[3*N+1])
        os = max(p[3*N+2], 0.5)  # keep positive
        oa = p[3*N+3]
        return inner, oc, os, oa

    def objective(p):
        inner, oc, os, oa = unpack_params(p)
        if is_valid(inner, oc, os, oa):
            return os
        # Penalty for invalidity
        pen = 0.0
        if not all_inside(inner, oc, os, oa):
            pen += 5.0
        if not pairwise_disjoint(inner):
            pen += 5.0
        return os + pen

    # Start Nelder-Mead from the baseline
    p0 = pack_params(inner, oc, os, oa)
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 2.0:
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6,
                                    'adaptive': True})
            if res.fun < best_os - 1e-9:
                inner_new, oc_new, os_new, oa_new = unpack_params(res.x)
                if is_valid(inner_new, oc_new, os_new, oa_new):
                    best_inner = [list(p) for p in inner_new]
                    best_oc = list(oc_new)
                    best_os = os_new
                    best_oa = oa_new
        except Exception:
            pass

    # --- Coordinate descent refinement on outer side length only ---
    # Try shrinking os in small steps while keeping inner fixed.
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 1.0:
        step = 0.01
        for _ in range(200):
            if time.time() - t0 > DEADLINE:
                break
            new_os = best_os - step
            if new_os < 1.0:
                break
            if is_valid(best_inner, best_oc, new_os, best_oa):
                best_os = new_os
            else:
                step *= 0.5
                if step < 1e-5:
                    break

    return best_inner, tuple(best_oc), best_os, best_oa
# EVOLVE-BLOCK-END
