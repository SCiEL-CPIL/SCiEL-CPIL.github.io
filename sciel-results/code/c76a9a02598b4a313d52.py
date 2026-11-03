"""Packing 11 unit hexagons in a regular hexagon via simulated annealing."""
import numpy as np
import time

N = 11

def _hexagon_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        a = np.radians(30 + i * 60)
        dist = lx * np.cos(a) + ly * np.sin(a)
        if dist > apothem + 1e-9:
            return False
    return True

def _check_fits(inner_hexes, outer_cx, outer_cy, outer_side, outer_angle):
    for (ix, iy, iangle) in inner_hexes:
        verts = _hexagon_vertices(ix, iy, 1.0, iangle)
        for (vx, vy) in verts:
            if not _point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                return False
    return True

def _min_side(inner_hexes, outer_cx, outer_cy, outer_angle):
    lo, hi = 1.0, 50.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _check_fits(inner_hexes, outer_cx, outer_cy, mid, outer_angle):
            hi = mid
        else:
            lo = mid
    return hi

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    # Honeycomb-like initial arrangement
    s3 = np.sqrt(3)
    start_positions = []
    # 4-3-4 pattern
    for i in range(4):
        start_positions.append([i * 1.75, 0.0, 0.0])
    for i in range(3):
        start_positions.append([0.875 + i * 1.75, s3 * 0.9, 0.0])
    for i in range(4):
        start_positions.append([i * 1.75, s3 * 1.8, 0.0])

    inner = [list(p) for p in start_positions]
    outer_cx, outer_cy = 2.0, s3 * 0.9
    outer_angle = 0.0

    current_side = _min_side(inner, outer_cx, outer_cy, outer_angle)
    best_inner = [list(p) for p in inner]
    best_cx, best_cy, best_angle = outer_cx, outer_cy, outer_angle
    best_side = current_side

    T = 0.8
    cooling = 0.9997
    min_T = 0.0005
    step_size = 0.25

    rng = np.random.default_rng(123)

    while T > min_T and time.time() < deadline:
        new_inner = [list(p) for p in inner]
        new_cx, new_cy, new_angle = outer_cx, outer_cy, outer_angle

        r = rng.random()
        if r < 0.25:
            idx = rng.integers(0, N)
            new_inner[idx][0] += rng.normal(0, step_size)
            new_inner[idx][1] += rng.normal(0, step_size)
        elif r < 0.45:
            idx = rng.integers(0, N)
            new_inner[idx][2] += rng.uniform(-25, 25)
        elif r < 0.60:
            new_cx += rng.normal(0, step_size * 0.4)
            new_cy += rng.normal(0, step_size * 0.4)
        elif r < 0.70:
            new_angle += rng.uniform(-12, 12)
        else:
            dx = rng.normal(0, step_size * 0.25)
            dy = rng.normal(0, step_size * 0.25)
            for i in range(N):
                new_inner[i][0] += dx
                new_inner[i][1] += dy
            new_cx += dx * 0.5
            new_cy += dy * 0.5

        new_side = _min_side(new_inner, new_cx, new_cy, new_angle)
        delta = new_side - current_side
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            inner = new_inner
            outer_cx, outer_cy, outer_angle = new_cx, new_cy, new_angle
            current_side = new_side
            if current_side < best_side:
                best_side = current_side
                best_inner = [list(p) for p in inner]
                best_cx, best_cy, best_angle = outer_cx, outer_cy, outer_angle

        T *= cooling
        step_size *= 0.9995

    # Local refinement
    for _ in range(300):
        if time.time() > deadline:
            break
        trial_inner = [list(p) for p in best_inner]
        idx = rng.integers(0, N)
        trial_inner[idx][0] += rng.normal(0, 0.04)
        trial_inner[idx][1] += rng.normal(0, 0.04)
        trial_side = _min_side(trial_inner, best_cx, best_cy, best_angle)
        if trial_side < best_side:
            best_inner = trial_inner
            best_side = trial_side

    return best_inner, (best_cx, best_cy), best_side, best_angle
# EVOLVE-BLOCK-END
