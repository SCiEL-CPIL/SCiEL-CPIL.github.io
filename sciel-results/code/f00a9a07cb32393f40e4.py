"""Circle packing in perimeter-4 rectangle: force-relaxation search + SLSQP polish."""
import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 21
DEADLINE = 130  # safety margin before 150s limit


def _validate_and_score(circles):
    """Validate circles and compute score (sum of radii)."""
    if circles is None:
        return -1.0
    if len(circles) != N:
        return -1.0
    if np.any(circles[:, 2] < -1e-9):
        return -1.0
    if np.any(np.isnan(circles)) or np.any(np.isinf(circles)):
        return -1.0
    return float(np.sum(circles[:, 2]))


def _check_validity(circles, w, h):
    """Check if circles are valid (non-overlapping, within bounds)."""
    if circles is None:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r < -1e-9:
            return False
        if x - r < -1e-9 or x + r > w + 1e-9:
            return False
        if y - r < -1e-9 or y + r > h + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dist = math.hypot(circles[i, 0] - circles[j, 0],
                              circles[i, 1] - circles[j, 1])
            if dist < circles[i, 2] + circles[j, 2] - 1e-9:
                return False
    return True


def construct():
    start_time = time.time()
    rng = random.Random(20240607)
    best_score = -1.0
    best_result = None
    best_w = None
    best_h = None

    def _time_left():
        return DEADLINE - (time.time() - start_time)

    def _update_best(circles, w, h):
        nonlocal best_score, best_result, best_w, best_h
        if circles is None:
            return
        if _check_validity(circles, w, h):
            s = _validate_and_score(circles)
            if s > best_score + 1e-12:
                best_score = s
                best_result = np.array(circles, dtype=float).copy()
                best_w = float(w)
                best_h = float(h)

    def _grid_positions(rows, cols, w, h):
        positions = np.zeros((N, 2))
        for i in range(N):
            row = i // cols
            col = i % cols
            if row == rows - 1:
                k = N - (rows - 1) * cols
                if k < cols:
                    x = (col + 0.5) * w / k
                else:
                    x = (col + 0.5) * w / cols
            else:
                x = (col + 0.5) * w / cols
            y = (row + 0.5) * h / rows
            positions[i] = [x, y]
        return positions

    def _hex_positions(w, h):
        """Hexagonal (brick) arrangement, top-to-bottom rows."""
        positions = []
        r = 0.05
        y = r
        row = 0
        while y + r <= h + 1e-9 and len(positions) < N:
            offset = (r if row % 2 else 0.0)
            x = r + offset
            while x + r <= w + 1e-9 and len(positions) < N:
                positions.append([x, y])
                x += 2 * r
            y += r * math.sqrt(3)
            row += 1
        # If we ran out, fill remaining with a grid fallback
        if len(positions) < N:
            base = _grid_positions(5, 5, w, h)
            for k in range(len(positions), N):
                positions.append(list(base[k]))
        positions = np.array(positions[:N])
        return positions

    def _random_positions(w, h):
        positions = np.zeros((N, 2))
        for i in range(N):
            positions[i, 0] = rng.uniform(0.0, w)
            positions[i, 1] = rng.uniform(0.0, h)
        return positions

    def _feasible_radii(pos, w, h):
        """Largest uniform radius such that pos is feasible (boundary + pairwise)."""
        r = float(min(w, h))
        # boundary
        for i in range(N):
            r = min(r, pos[i, 0], w - pos[i, 0], pos[i, 1], h - pos[i, 1])
        # pairwise
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                r = min(r, d / 2.0)
        return max(r, 0.0)

    def force_relaxation(pos, w, h, steps=400, lr_pos=0.04, lr_rad=0.0006, seed=0):
        """
        Force-directed relaxation: grow radii, push apart overlaps, keep in bounds.
        Returns (positions, radii).
        """
        local = random.Random(seed)
        pos = np.array(pos, dtype=float)
        # start with a safe radius (uniform)
        r0 = _feasible_radii(pos, w, h) * 0.99
        rad = np.full(N, max(r0, 1e-4))

        for step in range(steps):
            # 1. grow radii (schedule decays)
            grow = lr_rad * (1.0 + 0.5 * math.cos(math.pi * step / max(1, steps)))
            rad += grow

            # 2. boundary clamp
            for i in range(N):
                if pos[i, 0] - rad[i] < 0:
                    pos[i, 0] = rad[i]
                if pos[i, 0] + rad[i] > w:
                    pos[i, 0] = w - rad[i]
                if pos[i, 1] - rad[i] < 0:
                    pos[i, 1] = rad[i]
                if pos[i, 1] + rad[i] > h:
                    pos[i, 1] = h - rad[i]
                # if still infeasible (w/h too small for this radius), shrink
                if pos[i, 0] - rad[i] < -1e-12 or pos[i, 0] + rad[i] > w + 1e-12 or \
                   pos[i, 1] - rad[i] < -1e-12 or pos[i, 1] + rad[i] > h + 1e-12:
                    rad[i] = max(0.0, min(pos[i, 0], w - pos[i, 0], pos[i, 1], h - pos[i, 1]))

            # 3. resolve overlaps (several sweeps per step)
            for sweep in range(2):
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = math.hypot(dx, dy)
                        min_d = rad[i] + rad[j]
                        if dist < min_d:
                            if dist < 1e-8:
                                dx = local.uniform(-1, 1)
                                dy = local.uniform(-1, 1)
                                dist = math.hypot(dx, dy)
                                if dist < 1e-8:
                                    continue
                            nx = dx / dist
                            ny = dy / dist
                            overlap = min_d - dist
                            push = overlap * 0.5 * lr_pos * 8.0
                            pos[i, 0] += nx * push
                            pos[i, 1] += ny * push
                            pos[j, 0] -= nx * push
                            pos[j, 1] -= ny * push
                            # if overlap is large, shrink both
                            if overlap > min_d * 0.15:
                                rad[i] *= 0.985
                                rad[j] *= 0.985
                # re-clamp after pushes
                for i in range(N):
                    if pos[i, 0] - rad[i] < 0:
                        pos[i, 0] = rad[i]
                    if pos[i, 0] + rad[i] > w:
                        pos[i, 0] = w - rad[i]
                    if pos[i, 1] - rad[i] < 0:
                        pos[i, 1] = rad[i]
                    if pos[i, 1] + rad[i] > h:
                        pos[i, 1] = h - rad[i]

        # final cleanup to strict validity
        for _ in range(8):
            ok = True
            for i in range(N):
                if pos[i, 0] - rad[i] < -1e-12 or pos[i, 0] + rad[i] > w + 1e-12 or \
                   pos[i, 1] - rad[i] < -1e-12 or pos[i, 1] + rad[i] > h + 1e-12:
                    rad[i] = max(0.0, min(pos[i, 0], w - pos[i, 0], pos[i, 1], h - pos[i, 1]))
                    ok = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = math.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    if d < rad[i] + rad[j] - 1e-12:
                        rad[i] *= 0.95
                        rad[j] *= 0.95
                        ok = False
            if ok:
                break

        return pos, rad

    def run_slsqp(w_init, x0, y0, r0, max_iter=200, ftol=1e-10):
        """Polish a packing with SLSQP; updates best if improved."""
        h = 2.0 - w_init
        if h <= 0.1 or w_init <= 0.1:
            return
        n_var = 3 * N + 1

        def objective(v):
            xs = v[0:3*N:3]
            ys = v[1:3*N:3]
            rs = v[2:3*N:3]
            w = v[3*N]
            hh = 2.0 - w
            if hh <= 0.05 or w <= 0.05:
                return 1e10
            penalty = 0.0
            for i in range(N):
                if xs[i] - rs[i] < 0:
                    penalty += 1000 * (xs[i] - rs[i])**2
                if xs[i] + rs[i] > w:
                    penalty += 1000 * (xs[i] + rs[i] - w)**2
                if ys[i] - rs[i] < 0:
                    penalty += 1000 * (ys[i] - rs[i])**2
                if ys[i] + rs[i] > hh:
                    penalty += 1000 * (ys[i] + rs[i] - hh)**2
            for i in range(N):
                for j in range(i + 1, N):
                    dist = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
                    min_d = rs[i] + rs[j]
                    if dist < min_d:
                        penalty += 1000 * (min_d - dist)**2
            return -np.sum(rs) + penalty

        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0))
            bounds.append((0.0, 2.0))
            bounds.append((0.0, 1.0))
        bounds.append((0.3, 1.7))

        v0 = np.zeros(n_var)
        for i in range(N):
            v0[i*3] = x0[i]
            v0[i*3+1] = y0[i]
            v0[i*3+2] = r0[i]
        v0[3*N] = w_init

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': ftol})
            if res.fun < objective(v0) + 1e-9:
                xs = res.x[0:3*N:3]
                ys = res.x[1:3*N:3]
                rs = res.x[2:3*N:3]
                w = res.x[3*N]
                hh = 2.0 - w
                circles = np.column_stack([xs, ys, rs])
                _update_best(circles, w, hh)
        except Exception:
            pass

    # ---------- Phase 1: force-relaxation over a diverse seed pool ----------
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
    # (w, h) pairs from perimeter 4
    wh_pairs = []
    for ar in aspect_ratios:
        h = 2.0 / (ar + 1.0)
        w = ar * h
        wh_pairs.append((w, h))

    # grid row/col options
    grid_opts = [(3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (2, 11), (11, 2),
                 (3, 8), (8, 3), (4, 7), (7, 4), (5, 6), (6, 5)]

    seed_pool = []
    for (w, h) in wh_pairs:
        # hex seed
        seed_pool.append(('hex', w, h, _hex_positions(w, h)))
        # grid seeds
        for (rows, cols) in grid_opts:
            if rows * cols >= N:
                seed_pool.append(('grid', w, h, _grid_positions(rows, cols, w, h)))

    # random seeds (a few)
    for (w, h) in wh_pairs[::2]:
        seed_pool.append(('rand', w, h, _random_positions(w, h)))

    # Shuffle deterministically for diversity
    rng.shuffle(seed_pool)

    # Relax each seed
    for idx, (kind, w, h, pos) in enumerate(seed_pool):
        if _time_left() < 45.0:
            break
        try:
            p2, rad = force_relaxation(pos, w, h, steps=350, lr_pos=0.04, lr_rad=0.0007, seed=idx)
            circles = np.column_stack([p2, rad])
            _update_best(circles, w, h)
        except Exception:
            continue

    # ---------- Phase 2: re-relax the best packing with more steps / higher lr ----------
    if best_result is not None and _time_left() > 30.0:
        w, h = best_w, best_h
        pos = best_result[:, :2]
        # several refined relaxations with varied parameters
        for (steps, lr_pos, lr_rad) in [(600, 0.03, 0.0005), (600, 0.05, 0.0008),
                                        (800, 0.02, 0.0004), (500, 0.06, 0.001)]:
            if _time_left() < 25.0:
                break
            try:
                p2, rad = force_relaxation(pos, w, h, steps=steps, lr_pos=lr_pos, lr_rad=lr_rad, seed=99)
                circles = np.column_stack([p2, rad])
                _update_best(circles, w, h)
            except Exception:
                continue

    # ---------- Phase 3: SLSQP polish on the best packing ----------
    if best_result is not None and _time_left() > 12.0:
        pos = best_result[:, :2]
        rad = best_result[:, 2]
        w, h = best_w, best_h
        # try polishing with the current w and a couple of nearby w values
        for w_try in [w, max(0.3, w - 0.1), min(1.7, w + 0.1)]:
            if _time_left() < 8.0:
                break
            run_slsqp(w_try, pos, rad, rad, max_iter=200, ftol=1e-10)

    # ---------- Fallback ----------
    if best_result is None:
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
        best_result = np.array(cells)

    return best_result
