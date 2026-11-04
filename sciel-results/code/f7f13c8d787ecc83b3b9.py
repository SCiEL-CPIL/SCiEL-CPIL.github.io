"""Packing of 11 unit hexagons into a minimal regular hexagon.
Structural approach: hexagonal lattice layout + global outer-rotation
optimization (golden section) + Nelder-Mead positional refinement.
"""
import math
import time

N = 11
_SQRT3 = math.sqrt(3.0)
_T = 2.0 * _SQRT3  # touching distance between unit hexagons centers
_DEADLINE = 120.0 - 12.0  # safety margin before the 120s kill


def _hex_vertex_local(angle_deg):
    """Local (untranslated) vertices of a unit hexagon rotated by angle_deg."""
    a = math.radians(angle_deg)
    return [(math.cos(a + k * math.pi / 3.0), math.sin(a + k * math.pi / 3.0))
            for k in range(6)]


def _outer_normal_axes(rotation_deg):
    """The three outer-hexagon normal axes (unit vectors) for its rotation."""
    r = math.radians(rotation_deg)
    axes = []
    for k in range(3):
        ang = r + k * math.pi / 3.0
        axes.append((math.cos(ang), math.sin(ang)))
    return axes


def _required_side(inner, outer_center, outer_rot_deg):
    """Minimal outer side length to contain all inner hexagons at outer_rot_deg."""
    ox, oy = outer_center
    axes = _outer_normal_axes(outer_rot_deg)
    best = 0.0
    for (ix, iy, ang) in inner:
        for (vx, vy) in _hex_vertex_local(ang):
            px, py = ix + vx - ox, iy + vy - oy
            for (nx, ny) in axes:
                d = abs(px * nx + py * ny)
                if d > best:
                    best = d
    return best


def _objective(params, outer_rot_deg):
    """Required outer side for a flat param vector (x_i, y_i, a_i)."""
    inner = []
    for i in range(N):
        inner.append([params[3 * i], params[3 * i + 1], params[3 * i + 2]])
    return _required_side(inner, (0.0, 0.0), outer_rot_deg)


def _nelder_mead(params, outer_rot_deg, time_budget):
    """Nelder-Mead on the required-side objective for fixed outer rotation."""
    import numpy as np
    n = 3 * N
    x = np.array(params, dtype=float)
    simplex = [x.copy() for _ in range(n + 1)]
    for i in range(n):
        simplex[i + 1] = x.copy()
        simplex[i + 1][i] *= 1.001
        simplex[i + 1][i] += 1e-3
    fvals = [_objective(s, outer_rot_deg) for s in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    t_end = time.time() + time_budget
    iters = 0
    while time.time() < t_end and iters < 4000:
        iters += 1
        order = sorted(range(n + 1), key=lambda k: fvals[k])
        worst = order[-1]
        best = order[0]
        if abs(fvals[worst] - fvals[best]) < 1e-10:
            break
        centroid = sum(simplex[k] for k in order[:-1]) / n
        xr = centroid + alpha * (centroid - simplex[worst])
        fr = _objective(xr, outer_rot_deg)
        if fr < fvals[best]:
            xe = centroid + gamma * (centroid - simplex[worst])
            fe = _objective(xe, outer_rot_deg)
            if fe < fr:
                simplex[worst] = xe
                fvals[worst] = fe
            else:
                simplex[worst] = xr
                fvals[worst] = fr
        elif fr < fvals[order[-2]]:
            simplex[worst] = xr
            fvals[worst] = fr
        else:
            xc = centroid + rho * (simplex[worst] - centroid)
            fc = _objective(xc, outer_rot_deg)
            if fc < fvals[worst]:
                simplex[worst] = xc
                fvals[worst] = fc
            else:
                for k in range(1, n + 1):
                    simplex[k] = simplex[best] + sigma * (simplex[k] - simplex[best])
                    fvals[k] = _objective(simplex[k], outer_rot_deg)
    order = sorted(range(n + 1), key=lambda k: fvals[k])
    return list(simplex[order[0]]), fvals[order[0]]


def _golden_section(f, a, b, iters):
    """Minimize unimodal f on [a, b]."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def _build_hex_lattice():
    """Compact hexagonal arrangement of 11 hexagons: rows 3,2,1,2,3."""
    rows = [3, 2, 1, 2, 3]
    pts = []
    for r, count in enumerate(rows):
        y = (r - 2) * _SQRT3  # vertical row spacing = sqrt(3)
        for k in range(count):
            x = (k - (count - 1) / 2.0) * _T
            pts.append((x, y))
    return pts


def construct():
    t0 = time.time()

    def elapsed_ok():
        return time.time() - t0 < _DEADLINE

    base_pts = _build_hex_lattice()
    base_inner = [[x, y, 0.0] for (x, y) in base_pts]

    # Baseline side with no rotation.
    base_side = _required_side(base_inner, (0.0, 0.0), 0.0)

    best_inner = [list(row) for row in base_inner]
    best_side = base_side
    best_rot = 0.0

    def consider(inner, rot, side):
        nonlocal best_inner, best_side, best_rot
        if side < best_side - 1e-12:
            best_inner = [list(row) for row in inner]
            best_side = side
            best_rot = rot

    # 1) Optimize global outer rotation via golden section (lattice has 60deg symmetry).
    if elapsed_ok():
        def frot(r):
            return _required_side(base_inner, (0.0, 0.0), r)
        # search a 60-degree window covering the symmetry class
        rot = _golden_section(frot, 0.0, 60.0, 40)
        side_rot = frot(rot)
        consider(base_inner, rot, side_rot)

    # 2) Nelder-Mead positional refinement at the best rotation (fixed rotation).
    if elapsed_ok():
        params = []
        for (x, y, a) in best_inner:
            params.extend([x, y, a])
        time_left = max(2.0, _DEADLINE - (time.time() - t0))
        new_params, new_side = _nelder_mead(params, best_rot, time_left)
        new_inner = [[new_params[3 * i], new_params[3 * i + 1], new_params[3 * i + 2]]
                     for i in range(N)]
        consider(new_inner, best_rot, new_side)

    # 3) Re-optimize outer rotation after positional refinement (one more pass).
    if elapsed_ok():
        def frot2(r):
            return _required_side(best_inner, (0.0, 0.0), r)
        rot2 = _golden_section(frot2, best_rot - 30.0, best_rot + 30.0, 30)
        side2 = frot2(rot2)
        consider(best_inner, rot2, side2)

    return best_inner, (0.0, 0.0), best_side, best_rot


# EVOLVE-BLOCK-START
# (construct defined above; this block kept for interface compatibility)
# EVOLVE-BLOCK-END
