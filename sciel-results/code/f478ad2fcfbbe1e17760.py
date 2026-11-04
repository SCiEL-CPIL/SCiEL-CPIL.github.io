"""Circle packing in perimeter-4 rectangle: structural redesign.

Approach:
- Represent packing by centers (x_i, y_i) and radii (r_i) inside a W x H
  rectangle with 2*(W+H) <= 4.
- Start from a dense hexagonal lattice clipped to N circles.
- Optimize the rectangle aspect ratio for the layout (grid / hex).
- Local search: alternate
    (a) for fixed centers, compute the maximal radii under non-overlap
        (pairwise r_i + r_j <= dist_ij) and the perimeter constraint
        (r_i <= min(x_i, W - x_i, y_i, H - y_i)),
    (b) for fixed radii, move centers to increase the minimum slack
        (dist_ij - r_i - r_j) via gradient ascent (repulsion), with
        a tiny attraction to keep the packing compact.
- Deadline guard: return the best valid result before the limit.
"""
import time
import math
import numpy as np

N = 21
EPS = 1e-9
DEADLINE = 110.0  # safety margin under the 120 s wall-clock limit


def _hex_initial():
    """Build a hexagonal lattice, keep the N circles with largest
    'core' position (closest to rectangle center)."""
    # Hex lattice: rows offset by half spacing, vertical spacing = sqrt(3)/2 * s
    # We'll generate a generous field and pick the best N by distance to origin.
    # Use unit spacing s = 1 for now; radii will be scaled later.
    s = 1.0
    pts = []
    R = 8
    for row in range(-R, R + 1):
        y = row * math.sqrt(3) / 2 * s
        xoff = (0.5 * s) if (row % 2) else 0.0
        for col in range(-R, R + 1):
            x = col * s + xoff
            pts.append((x, y))
    pts = np.array(pts)
    # Keep the N closest to origin (this gives a compact cluster)
    d = np.hypot(pts[:, 0], pts[:, 1])
    order = np.argsort(d)
    pts = pts[order[:N]]
    return pts


def _grid_initial():
    """Build a compact square-ish grid of N circles (centers at integer lattice)."""
    side = int(math.ceil(math.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append((i, j))
    pts = np.array(pts[:N], dtype=float)
    # Center the grid at origin
    pts -= pts.mean(axis=0)
    return pts


def _aspect_for_centers(pts):
    """Given centers, find the best (W, H) with 2*(W+H)<=4 that maximizes
    the minimal 'clearance' min_i min(x_i, W-x_i, y_i, H-y_i).
    We choose W/H to balance the extents. Use a simple 1-D search over aspect."""
    xs = pts[:, 0]
    ys = pts[:, 1]
    # Extents (need at least this much)
    ex = max(xs.max() - xs.min(), 1e-6)
    ey = max(ys.max() - ys.min(), 1e-6)
    best = None
    # aspect a = W/H. Perimeter 4 => W + H = 2 => H = 2/(1+a), W = 2a/(1+a)
    # We want W >= ex + 2*r_min and H >= ey + 2*r_min, but r is what we're optimizing.
    # Instead, for a fixed aspect, the best centers placement is to center them,
    # and the maximal uniform clearance is limited by the smaller of the two
    # directions. We'll just center and compute clearance.
    for a in np.linspace(0.5, 2.0, 25):
        H = 2.0 / (1.0 + a)
        W = 2.0 * a / (1.0 + a)
        # Center the points within the rectangle
        cx = xs.mean()
        cy = ys.mean()
        # Place so that the bounding box is centered
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        # Shift so that (xmin+xmax)/2 -> W/2 and (ymin+ymax)/2 -> H/2
        shift_x = W / 2.0 - (xmin + xmax) / 2.0
        shift_y = H / 2.0 - (ymin + ymax) / 2.0
        px = xs + shift_x
        py = ys + shift_y
        # Clearance to walls
        clear = np.minimum(np.minimum(px, W - px), np.minimum(py, H - py))
        min_clear = clear.min()
        if min_clear < 0:
            continue
        if best is None or min_clear > best[0]:
            best = (min_clear, W, H, shift_x, shift_y)
    if best is None:
        # Fallback: square
        W = H = 1.0
        shift_x = W / 2.0 - (xs.min() + xs.max()) / 2.0
        shift_y = H / 2.0 - (ys.min() + ys.max()) / 2.0
        return W, H, shift_x, shift_y
    return best[1], best[2], best[3], best[4]


def _max_radii(pts, W, H):
    """For fixed centers in [0,W]x[0,H], compute maximal radii under
    pairwise non-overlap and wall constraints. Use a simple iterative
    water-filling: start with wall-limited radii, then enforce pair
    constraints r_i + r_j <= d_ij by reducing the larger of the two.
    Iterate until convergence."""
    n = len(pts)
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                 pts[:, None, 1] - pts[None, :, 1])
    # Wall constraint
    r = np.minimum(np.minimum(pts[:, 0], W - pts[:, 0]),
                   np.minimum(pts[:, 1], H - pts[:, 1]))
    r = np.maximum(r, 0.0)
    # Iteratively enforce pair constraints
    for _ in range(200):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                lim = d[i, j]
                s = r[i] + r[j]
                if s > lim + EPS:
                    # Reduce proportionally
                    if s > 0:
                        r[i] *= lim / s
                        r[j] *= lim / s
                    else:
                        r[i] = r[j] = 0.0
                    changed = True
        if not changed:
            break
    # Re-apply wall constraint (reduction may have helped, but ensure validity)
    r = np.minimum(r, np.minimum(np.minimum(pts[:, 0], W - pts[:, 0]),
                                 np.minimum(pts[:, 1], H - pts[:, 1])))
    r = np.maximum(r, 0.0)
    return r


def _gradient_step(pts, r, W, H, step, time_left):
    """Move centers to increase the minimum pairwise slack and wall clearance.
    Use a repulsion force from pairs that are too close, plus a weak pull
    toward the rectangle center to keep things compact. Returns new pts."""
    n = len(pts)
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                 pts[:, None, 1] - pts[None, :, 1])
    rsum = r[:, None] + r[None, :]
    slack = d - rsum
    # Force: for pairs with small slack, push apart. Use a smooth kernel.
    F = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dij = d[i, j]
            if dij < 1e-9:
                continue
            s = slack[i, j]
            if s < 0.5 * (r[i] + r[j]) + 0.1:
                # Repulsion strength grows as slack decreases
                w = 1.0 / (s + 0.05)
                dx = (pts[i, 0] - pts[j, 0]) / dij
                dy = (pts[i, 1] - pts[j, 1]) / dij
                F[i, 0] += w * dx
                F[i, 1] += w * dy
    # Weak attraction to center to prevent drift
    cx, cy = W / 2.0, H / 2.0
    F[:, 0] += 0.02 * (cx - pts[:, 0])
    F[:, 1] += 0.02 * (cy - pts[:, 1])
    # Normalize force magnitude to keep step controlled
    norm = np.linalg.norm(F, axis=1)
    norm = np.maximum(norm, 1e-9)
    F = F / norm[:, None]
    new_pts = pts + step * F
    # Clamp inside rectangle (with a tiny margin)
    new_pts[:, 0] = np.clip(new_pts[:, 0], 0.0, W)
    new_pts[:, 1] = np.clip(new_pts[:, 1], 0.0, H)
    return new_pts


def _is_valid(pts, r, W, H):
    n = len(pts)
    if len(pts) != N:
        return False
    if W <= 0 or H <= 0:
        return False
    if 2 * (W + H) > 4.0 + 1e-6:
        return False
    if np.any(r < -1e-9):
        return False
    # Wall
    if np.any(pts[:, 0] < -1e-7) or np.any(pts[:, 0] > W + 1e-7):
        return False
    if np.any(pts[:, 1] < -1e-7) or np.any(pts[:, 1] > H + 1e-7):
        return False
    if np.any(r > pts[:, 0] + 1e-6) or np.any(r > W - pts[:, 0] + 1e-6):
        return False
    if np.any(r > pts[:, 1] + 1e-6) or np.any(r > H - pts[:, 1] + 1e-6):
        return False
    # Pairwise
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                 pts[:, None, 1] - pts[None, :, 1])
    rsum = r[:, None] + r[None, :]
    if np.any(rsum > d + 1e-6):
        return False
    return True


def _score(pts, r, W, H):
    return float(np.sum(r))


def _local_search(pts0, W0, H0, time_limit):
    """Run local search starting from pts0 in W0 x H0. Return best (pts, r, W, H, score)."""
    pts = pts0.copy()
    W, H = W0, H0
    r = _max_radii(pts, W, H)
    best_score = _score(pts, r, W, H)
    best_state = (pts.copy(), r.copy(), W, H)
    step = 0.01
    t0 = time.time()
    no_improve = 0
    while time.time() - t0 < time_limit:
        if not _is_valid(pts, r, W, H):
            # Repair: shrink radii to validity
            r = _max_radii(pts, W, H)
            if not _is_valid(pts, r, W, H):
                break
        # Move centers
        pts = _gradient_step(pts, r, W, H, step, time_limit)
        # Recompute radii
        r = _max_radii(pts, W, H)
        if _is_valid(pts, r, W, H):
            s = _score(pts, r, W, H)
            if s > best_score + 1e-9:
                best_score = s
                best_state = (pts.copy(), r.copy(), W, H)
                no_improve = 0
                # Slightly increase step on improvement? Keep stable.
            else:
                no_improve += 1
                # Decay step if stuck
                if no_improve > 10:
                    step *= 0.9
                    no_improve = 0
        else:
            # Roll back to best
            pts, r, W, H = best_state
            step *= 0.8
    return best_state[0], best_state[1], best_state[2], best_state[3], best_score


def construct():
    t_start = time.time()
    best = None  # (pts, r, W, H, score)

    def _consider(pts, r, W, H):
        nonlocal best
        if not _is_valid(pts, r, W, H):
            return
        s = _score(pts, r, W, H)
        if best is None or s > best[4] + 1e-12:
            best = (pts, r, W, H, s)

    # --- Initialize from multiple layouts ---
    inits = []
    # Hex
    hex_pts = _hex_initial()
    W, H, sx, sy = _aspect_for_centers(hex_pts)
    inits.append(hex_pts + np.array([sx, sy]))
    # Grid
    grid_pts = _grid_initial()
    W, H, sx, sy = _aspect_for_centers(grid_pts)
    inits.append(grid_pts + np.array([sx, sy]))

    for pts0 in inits:
        if time.time() - t_start > DEADLINE - 5:
            break
        # Get aspect for this layout
        W, H, sx, sy = _aspect_for_centers(pts0 - np.array([W / 2, H / 2]))
        # Re-center properly: aspect_for_centers expects raw centers; we pass pts0
        # and it returns shifts to center within the chosen W,H.
        W, H, sx, sy = _aspect_for_centers(pts0)
        pts0 = pts0 + np.array([sx, sy])
        r = _max_radii(pts0, W, H)
        _consider(pts0, r, W, H)
        # Local search with a time budget
        budget = min(40.0, DEADLINE - (time.time() - t_start) - 5)
        if budget > 2:
            pts, r, W, H, s = _local_search(pts0, W, H, budget)
            _consider(pts, r, W, H)

    if best is None:
        # Fallback: simple square grid (valid)
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        arr = np.array(cells[:N])
        # Verify validity; if not, shrink radii
        W = H = 1.0
        r = arr[:, 2]
        # Ensure wall
        r = np.minimum(r, np.minimum(np.minimum(arr[:, 0], W - arr[:, 0]),
                                     np.minimum(arr[:, 1], H - arr[:, 1])))
        arr[:, 2] = r
        return arr

    pts, r, W, H, s = best
    out = np.zeros((N, 3))
    out[:, 0] = pts[:, 0]
    out[:, 1] = pts[:, 1]
    out[:, 2] = r
    return out
