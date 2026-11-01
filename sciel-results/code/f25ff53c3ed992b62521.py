"""Physics-based optimizer for packing 50 equal circles in a unit square."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 50
HALF = 0.5
EPS = 1e-12


def _compute_radius(pts):
    """Compute the largest common radius for a set of centres in [-1/2,1/2]^2."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts[:, 0] < -HALF - 1e-9) or np.any(pts[:, 0] > HALF + 1e-9) or \
       np.any(pts[:, 1] < -HALF - 1e-9) or np.any(pts[:, 1] > HALF + 1e-9):
        return 0.0
    # boundary term
    b = HALF - np.max(np.abs(pts), axis=1)
    r_bound = float(np.min(b))
    if r_bound <= 0:
        return 0.0
    # pairwise term
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=2.0 * max(r_bound, 1e-6) + 1e-6, output_type='ndarray')
    if len(pairs) == 0:
        return r_bound
    d = pts[pairs[:, 0]] - pts[pairs[:, 1]]
    dist = np.sqrt(np.sum(d * d, axis=1))
    r_pair = float(np.min(dist)) / 2.0
    return min(r_bound, r_pair)


def _init_hex(r0):
    """Hexagonal packing starting from a guess radius r0, clipped to square."""
    pts = []
    if r0 <= 0:
        r0 = 0.07
    dx = 2.0 * r0
    dy = np.sqrt(3.0) * r0
    x0 = -HALF + r0
    y0 = -HALF + r0
    row = 0
    while len(pts) < N:
        for col in range(100):
            x = x0 + col * dx
            y = y0 + row * dy + (0.5 * dy if row % 2 == 1 else 0.0)
            if x <= HALF - r0 + 1e-9 and y <= HALF - r0 + 1e-9 and x >= -HALF - 1e-9 and y >= -HALF - 1e-9:
                pts.append([x, y])
            if len(pts) >= N:
                break
        row += 1
    pts = np.array(pts[:N], dtype=np.float64)
    return pts


def _init_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _energy(pts, r):
    """Repulsive energy: large when circles overlap or are too close to walls."""
    if r <= 0:
        r = 1e-6
    # wall penalty
    wall = np.maximum(0.0, (r - (HALF - np.abs(pts[:, 0])))) + \
           np.maximum(0.0, (r - (HALF - np.abs(pts[:, 1]))))
    wall_pen = np.sum(wall ** 2)
    # pair overlap penalty
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=2.0 * r + 1e-6, output_type='ndarray')
    pair_pen = 0.0
    if len(pairs) > 0:
        d = pts[pairs[:, 0]] - pts[pairs[:, 1]]
        dist = np.sqrt(np.sum(d * d, axis=1))
        overlap = np.maximum(0.0, 2.0 * r - dist)
        pair_pen = np.sum(overlap ** 2)
    return wall_pen + pair_pen


def _step(pts, r, lr):
    """One gradient-ish step: push apart overlapping circles and away from walls."""
    new = pts.copy()
    # wall forces
    for i in range(N):
        for axis in range(2):
            coord = new[i, axis]
            if coord > HALF - r:
                new[i, axis] = min(coord - lr, HALF - r)
            elif coord < -HALF + r:
                new[i, axis] = max(coord + lr, -HALF + r)
    # pair forces
    tree = cKDTree(new)
    pairs = tree.query_pairs(r=2.0 * r + 1e-6, output_type='ndarray')
    if len(pairs) > 0:
        for a, b in pairs:
            d = new[a] - new[b]
            dist = np.sqrt(np.sum(d * d))
            if dist < 1e-12:
                d = np.array([1e-8, 0.0])
                dist = 1e-8
            overlap = 2.0 * r - dist
            if overlap > 0:
                push = 0.5 * lr * overlap / dist
                new[a] += d * push
                new[b] -= d * push
    # clip to square
    new = np.clip(new, -HALF + 1e-12, HALF - 1e-12)
    return new


def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # 20s safety margin

    # Start from a hexagonal layout with a reasonable radius guess
    best_pts = _init_hex(0.071)
    best_r = _compute_radius(best_pts)

    # Also try grid as a secondary candidate
    grid_pts = _init_grid()
    grid_r = _compute_radius(grid_pts)
    if grid_r > best_r:
        best_pts = grid_pts.copy()
        best_r = grid_r

    # Simulated annealing with repulsion
    current = best_pts.copy()
    current_r = best_r
    r_target = 0.072  # target radius to push toward
    lr = 0.02
    temp = 0.01
    steps_per_iter = 200
    iters = 400

    for it in range(iters):
        if time.time() > deadline - 5.0:
            break
        # Annotate: try to grow target radius slightly as we progress
        r_target = min(0.073, 0.071 + 0.002 * (it / max(1, iters)))
        for s in range(steps_per_iter):
            if time.time() > deadline - 5.0:
                break
            # Perturb a few random points
            n_pert = 2
            idx = np.random.choice(N, n_pert, replace=False)
            current[idx] += np.random.normal(0, temp, size=(n_pert, 2))
            current = np.clip(current, -HALF + 1e-12, HALF - 1e-12)
            # Gradient steps
            for _ in range(5):
                current = _step(current, r_target, lr)
            cur_r = _compute_radius(current)
            if cur_r > current_r:
                current_r = cur_r
                if cur_r > best_r:
                    best_r = cur_r
                    best_pts = current.copy()
        # Anneal
        temp *= 0.98
        lr *= 0.995

    # Final local polish with smaller target
    for polish in range(50):
        if time.time() > deadline - 2.0:
            break
        current = best_pts.copy()
        for _ in range(200):
            current = _step(current, best_r * 1.001, 0.005)
        pol_r = _compute_radius(current)
        if pol_r > best_r:
            best_r = pol_r
            best_pts = current.copy()

    return best_pts
