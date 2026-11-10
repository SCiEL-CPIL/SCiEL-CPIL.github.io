# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid(centers, sides, tol=1e-7):
    """Vectorized validity check: inside unit square, positive sides, non-overlapping."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside unit square
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1.0 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1.0 + tol):
        return False
    # Non-overlapping (vectorized)
    dx = np.abs(centers[:, 0:1] - centers[0:1, 0])
    dy = np.abs(centers[:, 1:2] - centers[0:2, 1])
    t = (sides[:, None] + sides[None, :]) / 2.0 - tol
    if np.any((dx < t) & (dy < t)):
        return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def _clamp_to_unit(centers, sides):
    """Clamp centers so squares stay inside [0,1]^2."""
    half = sides / 2.0
    centers[:, 0] = np.clip(centers[:, 0], half, 1.0 - half)
    centers[:, 1] = np.clip(centers[:, 1], half, 1.0 - half)
    return centers


def _max_side_at(centers, sides, i, tol=1e-7, iters=60):
    """
    Vectorized max side for square i: binary search over side length,
    checking all other squares at once.
    """
    n = len(sides)
    cx, cy = centers[i]
    lo = sides[i]
    hi = min(1.0, 2.0 * min(cx, 1.0 - cx, cy, 1.0 - cy) + tol)
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        half = mid / 2.0
        if cx - half < -tol or cx + half > 1.0 + tol or cy - half < -tol or cy + half > 1.0 + tol:
            hi = mid
            continue
        # Vectorized check against all j != i
        t = (mid + sides) / 2.0 - tol
        t[i] = np.inf
        dx = np.abs(cx - centers[:, 0])
        dy = np.abs(cy - centers[:, 1])
        if np.any((dx < t) & (dy < t)):
            hi = mid
        else:
            lo = mid
    return lo


def _grow_all_iteratively(centers, sides, tol=1e-7, max_iters=300):
    """Iteratively grow each square to its maximum given others."""
    n = len(sides)
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            new_s = _max_side_at(centers, sides, i, tol)
            if new_s > sides[i] + 1e-12:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides


def _high_precision_polish(centers, sides, tol=1e-9, max_iters=500):
    """High-precision polish with tighter tolerance and more iterations."""
    n = len(sides)
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            new_s = _max_side_at(centers, sides, i, tol=tol, iters=80)
            if new_s > sides[i] + 1e-14:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides


def _force_based_separation(centers, sides, tol=1e-7, max_iters=300, lr=0.05):
    """Force-based repulsion to separate overlapping squares (vectorized inner loop)."""
    n = len(sides)
    for it in range(max_iters):
        forces = np.zeros((n, 2))
        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                ax, ay = abs(dx), abs(dy)
                if ax < t and ay < t:
                    ox = max(0.0, t - ax)
                    oy = max(0.0, t - ay)
                    max_overlap = max(max_overlap, ox, oy)
                    if ox < oy:
                        sgn = 1.0 if dx >= 0 else -1.0
                        forces[i, 0] += sgn * ox * 10
                        forces[j, 0] -= sgn * ox * 10
                    else:
                        sgn = 1.0 if dy >= 0 else -1.0
                        forces[i, 1] += sgn * oy * 10
                        forces[j, 1] -= sgn * oy * 10
        if max_overlap < 1e-10:
            break
        centers += lr * forces
        centers = _clamp_to_unit(centers, sides)
    return centers


def _push_to_corner(centers, sides, i, tol=1e-7):
    """Push square i to corners/edges, then regrow all."""
    n = len(sides)
    best_obj = _objective(centers, sides)
    best_c = centers.copy()
    best_s = sides.copy()
    half = sides[i] / 2.0
    targets = [
        (half, half), (1.0 - half, half), (half, 1.0 - half), (1.0 - half, 1.0 - half),
        (0.5, half), (0.5, 1.0 - half), (half, 0.5), (1.0 - half, 0.5),
    ]
    for tx, ty in targets:
        test_c = centers.copy()
        test_c[i, 0] = tx
        test_c[i, 1] = ty
        if not _is_valid(test_c, sides, tol):
            continue
        trial_s = sides.copy()
        for k in range(n):
            ns = _max_side_at(test_c, trial_s, k, tol)
            if ns > trial_s[k] + 1e-10:
                trial_s[k] = ns
        if _is_valid(test_c, trial_s, tol):
            obj = _objective(test_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = test_c.copy()
                best_s = trial_s.copy()
    return best_c, best_s


def _shake_and_regrow(centers, sides, tol=1e-7, rng=None):
    """Perturb all squares, resolve, regrow."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(sides)
    new_centers = centers.copy()
    dx = rng.uniform(-0.05, 0.05, n)
    dy = rng.uniform(-0.05, 0.05, n)
    new_centers[:, 0] = np.clip(centers[:, 0] + dx, sides / 2, 1.0 - sides / 2)
    new_centers[:, 1] = np.clip(centers[:, 1] + dy, sides / 2, 1.0 - sides / 2)
    new_centers = _force_based_separation(new_centers, sides, tol)
    if not _is_valid(new_centers, sides, tol):
        return None
    new_sides = _grow_all_iteratively(new_centers.copy(), sides.copy(), tol)
    if _is_valid(new_centers, new_sides, tol):
        return new_centers, new_sides
    return None


def _smooth_penalty(centers, sides, tol=1e-7, beta=50.0):
    """
    Smooth surrogate for the nonsmooth overlap constraint.
    For each pair, the forbidden region is |dx|<t AND |dy|<t.
    We use a soft product: penalty = sum over pairs of softmax of (t - dx, t - dy).
    Returns a scalar penalty (0 when no overlap, >0 when overlapping).
    """
    n = len(sides)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            # Overlap depth in each axis (positive means violation)
            vx = t - dx
            vy = t - dy
            if vx > 0 and vy > 0:
                # Soft-min of vx, vy via log-sum-exp; we want the min (bottleneck)
                m = min(vx, vy)
                penalty += m
    return penalty


def _simulated_annealing(centers, sides, tol=1e-7, rng=None, t_start=0.05, t_end=0.003,
                         n_iter=800, deadline=None):
    """
    SA on positions using a smooth penalty surrogate.
    Moves: random small displacement of a random square.
    Accepts moves that reduce the smooth penalty; also accepts worse with prob exp(-dE/T).
    Objective: maximize sum(sides) - lambda*penalty. We grow sides after each accepted move.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(sides)
    cur_c = centers.copy()
    cur_s = sides.copy()
    cur_penalty = _smooth_penalty(cur_c, cur_s, tol)
    cur_obj = _objective(cur_c, cur_s) - 5.0 * cur_penalty
    best_c = cur_c.copy()
    best_s = cur_s.copy()
    best_obj = cur_obj
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        T = t_start * (t_end / t_start) ** (it / max(1, n_iter - 1))
        # Propose move: displace random square
        i = rng.integers(n)
        dx = rng.uniform(-0.03, 0.03)
        dy = rng.uniform(-0.03, 0.03)
        new_c = cur_c.copy()
        new_c[i, 0] = np.clip(cur_c[i, 0] + dx, cur_s[i] / 2, 1.0 - cur_s[i] / 2)
        new_c[i, 1] = np.clip(cur_c[i, 1] + dy, cur_s[i] / 2, 1.0 - cur_s[i] / 2)
        new_penalty = _smooth_penalty(new_c, cur_s, tol)
        new_obj = _objective(new_c, cur_s) - 5.0 * new_penalty
        dE = new_obj - cur_obj
        if dE >= 0 or rng.random() < np.exp(dE / max(T, 1e-9)):
            cur_c = new_c
            cur_penalty = new_penalty
            cur_obj = new_obj
            # Try to grow
            trial_s = _grow_all_iteratively(cur_c.copy(), cur_s.copy(), tol)
            if _is_valid(cur_c, trial_s, tol):
                cur_s = trial_s
                cur_obj = _objective(cur_c, cur_s) - 5.0 * _smooth_penalty(cur_c, cur_s, tol)
                if _objective(cur_c, cur_s) > _objective(best_c, best_s) + 1e-9:
                    best_c = cur_c.copy()
                    best_s = cur_s.copy()
                    best_obj = cur_obj
    return best_c, best_s


def _local_optimize(centers, sides, tol=1e-7, max_rounds=80, seed=42, deadline=None):
    """Local optimization: grow, corner-push, SA escape, perturb, polish."""
    n = len(sides)
    best_sides = sides.copy()
    best_centers = centers.copy()
    best_obj = _objective(best_centers, best_sides)
    rng = np.random.default_rng(seed=seed)

    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        # Step 1: Grow all
        new_sides = _grow_all_iteratively(centers.copy(), sides.copy(), tol)
        if _is_valid(centers, new_sides, tol):
            obj = _objective(centers, new_sides)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_sides = new_sides.copy()
                best_centers = centers.copy()
                sides = new_sides.copy()

        # Step 2: Corner-push each square
        for i in range(n):
            new_c, new_s = _push_to_corner(centers, sides, i, tol)
            if _objective(new_c, new_s) > best_obj + 1e-9:
                best_obj = _objective(new_c, new_s)
                best_centers = new_c
                best_sides = new_s
                centers = new_c.copy()
                sides = new_s.copy()
                sides = _grow_all_iteratively(centers.copy(), sides.copy(), tol)
                if _is_valid(centers, sides, tol):
                    obj = _objective(centers, sides)
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_sides = sides.copy()
                        best_centers = centers.copy()

        # Step 3: SA escape (smooth surrogate)
        if rnd % 4 == 2:
            sa_c, sa_s = _simulated_annealing(
                centers, sides, tol, rng,
                t_start=0.04, t_end=0.003, n_iter=200, deadline=deadline)
            if _is_valid(sa_c, sa_s, tol):
                obj = _objective(sa_c, sa_s)
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_centers = sa_c.copy()
                    best_sides = sa_s.copy()
                    centers = sa_c.copy()
                    sides = sa_s.copy()

        # Step 4: Shake and regrow
        if rnd % 5 == 2:
            result = _shake_and_regrow(centers, sides, tol, rng)
            if result is not None:
                new_c, new_s = result
                obj = _objective(new_c, new_s)
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_centers = new_c.copy()
                    best_sides = new_s.copy()
                    centers = new_c.copy()
                    sides = new_s.copy()

        # Step 5: Targeted perturbations
        for i in range(n):
            for _ in range(2):
                dx = rng.uniform(-0.03, 0.03)
                dy = rng.uniform(-0.03, 0.03)
                new_c = centers.copy()
                new_c[i, 0] = np.clip(centers[i, 0] + dx, sides[i] / 2, 1.0 - sides[i] / 2)
                new_c[i, 1] = np.clip(centers[i, 1] + dy, sides[i] / 2, 1.0 - sides[i] / 2)
                if _is_valid(new_c, sides, tol):
                    trial_s = _grow_all_iteratively(new_c.copy(), sides.copy(), tol)
                    if _is_valid(new_c, trial_s, tol):
                        obj = _objective(new_c, trial_s)
                        if obj > best_obj + 1e-9:
                            best_obj = obj
                            best_centers = new_c.copy()
                            best_sides = trial_s.copy()
                            centers = new_c.copy()
                            sides = trial_s.copy()

        # Convergence check
        if rnd > 10 and abs(_objective(centers, sides) - best_obj) < 1e-14:
            can_improve = False
            for i in range(n):
                if _max_side_at(centers, sides, i, tol) > sides[i] + 1e-12:
                    can_improve = True
                    break
            if not can_improve:
                break

    return best_centers, best_sides


def _init_grid_packing():
    n = 10
    k = 3
    side = 1.0 / k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[:9] = np.stack([cx, cy], axis=1)
    sides[:9] = side
    centers[9] = [0.5, 0.5]
    sides[9] = 1.0 / 6.0
    return centers, sides


def _init_heterogeneous_1():
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0 / 6.0
    idx = 1
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_2():
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.2]
    sides[1] = 0.4
    side_s = 0.25
    idx = 2
    for row in range(2):
        for col in range(4):
            centers[idx, 0] = (col + 0.5) * side_s
            centers[idx, 1] = 0.4 + (row + 0.5) * side_s
            sides[idx] = side_s
            idx += 1
    return centers, sides


def _init_heterogeneous_4():
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.125, 0.625]
    sides[1] = 0.25
    centers[2] = [0.875, 0.625]
    sides[2] = 0.25
    side_s = 0.125
    idx = 3
    positions = [
        [0.375, 0.625], [0.625, 0.625],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
        [0.75, 0.25],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_5():
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.225, 0.225]
    sides[0] = 0.45
    centers[1] = [0.775, 0.775]
    sides[1] = 0.45
    side_s = 0.15
    idx = 2
    positions = [
        [0.225, 0.575], [0.225, 0.875],
        [0.525, 0.225], [0.825, 0.225],
        [0.525, 0.525],
        [0.05, 0.575], [0.05, 0.875],
        [0.825, 0.525],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_6():
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.165, 0.165]
    sides[0] = 0.33
    centers[1] = [0.835, 0.165]
    sides[1] = 0.33
    centers[2] = [0.165, 0.835]
    sides[2] = 0.33
    side_s = 0.15
    idx = 3
    positions = [
        [0.5, 0.165], [0.835, 0.5], [0.5, 0.5],
        [0.5, 0.835], [0.835, 0.835], [0.33, 0.5],
        [0.665, 0.33],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_random(seed=123):
    rng = np.random.default_rng(seed=seed)
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    for i in range(10):
        s = rng.uniform(0.1, 0.35)
        sides[i] = s
        centers[i, 0] = rng.uniform(s / 2, 1.0 - s / 2)
        centers[i, 1] = rng.uniform(s / 2, 1.0 - s / 2)
    return centers, sides


def _resolve_overlaps(centers, sides, tol=1e-7):
    return _force_based_separation(centers, sides, tol, max_iters=500)


def squares_in_square10() -> tuple:
    """Erdos 'squares in a square' problem for n=10. Returns (centers, sides)."""
    start_time = time.time()
    deadline = start_time + 570.0

    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_4(),
        _init_heterogeneous_5(),
        _init_heterogeneous_6(),
    ]
    for s in [123, 456, 789, 1011, 2020, 3030, 4040, 5050]:
        inits.append(_init_random(seed=s))

    best_centers = None
    best_sides = None
    best_obj = -1.0

    for idx, (centers, sides) in enumerate(inits):
        if time.time() > deadline:
            break
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        centers = _resolve_overlaps(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        if not _is_valid(centers, sides):
            centers = _force_based_separation(centers, sides, max_iters=1000)
            centers = _clamp_to_unit(centers, sides)
            if not _is_valid(centers, sides):
                continue
        opt_centers, opt_sides = _local_optimize(
            centers, sides, seed=42 + idx * 100, deadline=deadline)
        if _is_valid(opt_centers, opt_sides):
            obj = _objective(opt_centers, opt_sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()

        # Extra shake on best
        if best_obj > 1.15 and time.time() < deadline - 60:
            rng = np.random.default_rng(seed=999 + idx)
            for _ in range(4):
                result = _shake_and_regrow(best_centers, best_sides, rng=rng)
                if result is not None:
                    new_c, new_s = result
                    obj = _objective(new_c, new_s)
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_centers = new_c.copy()
                        best_sides = new_s.copy()

    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
        # Final high-precision polish
        if time.time() < deadline - 10:
            polished = _high_precision_polish(best_centers.copy(), best_sides.copy(),
                                              tol=1e-9, max_iters=300)
            if _is_valid(best_centers, polished, tol=1e-7):
                if _objective(best_centers, polished) > _objective(best_centers, best_sides):
                    best_sides = polished.copy()

    return best_centers, best_sides


# EVOLVE-BLOCK-END
