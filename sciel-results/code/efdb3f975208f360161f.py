import numpy as np
import time

N = 11

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = np.radians(angle_deg)
    return np.array([(cx + side * np.cos(a + np.radians(60 * k)),
                      cy + side * np.sin(a + np.radians(60 * k))) for k in range(6)])

def _hex_radial_extent(lx, ly, S):
    """Radial distance from center to boundary of flat-top regular hexagon with side S."""
    if abs(lx) < 1e-15 and abs(ly) < 1e-15:
        return S
    theta = np.arctan2(ly, lx) % (2.0 * np.pi)
    k = int(np.floor(theta / (np.pi / 3.0))) % 6
    center_angle = k * np.pi / 3.0
    delta = theta - center_angle
    if delta > np.pi / 6.0:
        delta -= np.pi / 3.0
    elif delta < -np.pi / 6.0:
        delta += np.pi / 3.0
    inradius = S * np.sqrt(3.0) / 2.0
    c = np.cos(delta)
    if c < 1e-12:
        return 1e9
    return inradius / c

def _max_violation(centers, angles, outer_center, outer_side, outer_angle):
    ox, oy = outer_center
    S = outer_side
    a = np.radians(outer_angle)
    ca, sa = np.cos(-a), np.sin(-a)
    max_v = 0.0
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, angles[i])
        for v in verts:
            dx, dy = v[0] - ox, v[1] - oy
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            dist = np.hypot(lx, ly)
            r_b = _hex_radial_extent(lx, ly, S)
            v_val = dist - r_b
            if v_val > max_v:
                max_v = v_val
    return max_v

def _required_side(centers, angles, outer_center, outer_angle, tol=1e-9):
    lo, hi = 2.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _max_violation(centers, angles, outer_center, mid, outer_angle) < tol:
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    start_time = time.time()
    deadline = start_time + 125.0

    # Hexagonal cluster: 1 center + 6 ring1 + 4 ring2
    centers = np.array([(0.0, 0.0)])
    for k in range(6):
        theta = np.radians(60.0 * k)
        centers = np.vstack([centers, [[2.0 * np.cos(theta), 2.0 * np.sin(theta)]]])
    # 4 symmetric points from ring 2
    centers = np.vstack([centers, [
        [4.0, 0.0], [-4.0, 0.0],
        [0.0, 2.0 * np.sqrt(3.0)], [0.0, -2.0 * np.sqrt(3.0)]
    ]])
    assert len(centers) == N

    best_side = np.inf
    best_params = None

    outer_angles = [0.0, 30.0, 15.0, 45.0, 22.5, 67.5, 10.0, 50.0, 5.0, 35.0]
    inner_rots = [0.0, 30.0, 15.0, 45.0]

    for oa in outer_angles:
        if time.time() > deadline:
            break
        for ir in inner_rots:
            if time.time() > deadline:
                break
            angles = np.full(N, ir)
            S = _required_side(centers, angles, (0.0, 0.0), oa)
            if S < best_side:
                best_side = S
                best_params = (centers.copy(), angles.copy(), (0.0, 0.0), S, oa)

    if best_params is not None:
        c, a, oc, S, oa = best_params
        inner = [[c[i][0], c[i][1], a[i]] for i in range(N)]
        return inner, oc, S, oa

    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
