"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    for i in range(n):
        dx = pos[i, 0] - pos[i+1:, 0]
        dy = pos[i, 1] - pos[i+1:, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < radii[i] + radii[i+1:] - tol):
            return False
    return True


def _compute_penalty_and_grads(pos, radii):
    """
    Compute total penalty and gradients w.r.t. pos and radii.
    Uses quadratic penalty for overlaps and boundaries.
    Returns (penalty, grad_pos, grad_rad) where grad_pos/grad_rad are
    the forces that REDUCE penalty (i.e., -d(penalty)/d(x) for pos,
    and +d(penalty)/d(r) for radii since we want to shrink them).
    """
    n = len(pos)
    grad_pos = np.zeros_like(pos)
    grad_rad = np.zeros_like(radii)
    penalty = 0.0

    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(max(dist_sq, 1e-20))
                viol = min_dist - dist
                if viol > 0:
                    penalty += viol * viol
                    if dist < 1e-10:
                        angle = np.random.uniform(0, 2 * np.pi)
                        nx, ny = np.cos(angle), np.sin(angle)
                    else:
                        nx, ny = dx / dist, dy / dist
                    # Push i away from j, j away from i
                    grad_pos[i, 0] -= 2 * viol * nx
                    grad_pos[i, 1] -= 2 * viol * ny
                    grad_pos[j, 0] += 2 * viol * nx
                    grad_pos[j, 1] += 2 * viol * ny
                    # Reduce radii
                    grad_rad[i] += 2 * viol
                    grad_rad[j] += 2 * viol

    # Boundary penalties
    for i in range(n):
        r = radii[i]
        if pos[i, 0] < r:
            viol = r - pos[i, 0]
            penalty += viol * viol
            grad_pos[i, 0] += 2 * viol
            grad_rad[i] += 2 * viol
        if pos[i, 0] > 1 - r:
            viol = pos[i, 0] - (1 - r)
            penalty += viol * viol
            grad_pos[i, 0] -= 2 * viol
            grad_rad[i] += 2 * viol
        if pos[i, 1] < r:
            viol = r - pos[i, 1]
            penalty += viol * viol
            grad_pos[i, 1] += 2 * viol
            grad_rad[i] += 2 * viol
        if pos[i, 1] > 1 - r:
            viol = pos[i, 1] - (1 - r)
            penalty += viol * viol
            grad_pos[i, 1] -= 2 * viol
            grad_rad[i] += 2 * viol

    return penalty, grad_pos, grad_rad


def _gradient_optimize(pos0, radii0, time_budget=30.0, start_time=None, rng=None):
    """
    Gradient-based optimization with momentum and adaptive learning rates.
    Maximizes sum(radii) by balancing penalty reduction and radius growth.
    """
    if start_time is None:
        start_time = time.time()
    if rng is None:
        rng = np.random.RandomState(int(time.time() * 1000) % (2**31))

    n = len(pos0)
    pos = pos0.copy()
    radii = radii0.copy()

    lr_pos = 0.003
    lr_rad = 0.0005
    vel_pos = np.zeros_like(pos)
    vel_rad = np.zeros_like(radii)
    beta = 0.9

    best_score = -1e9
    best_pos = pos.copy()
    best_radii = radii.copy()

    if _check_validity(pos, radii):
        best_score = np.sum(radii)
        best_pos = pos.copy()
        best_radii = radii.copy()

    no_improve_count = 0
    max_no_improve = 3000

    for it in range(20000):
        if time.time() - start_time > time_budget:
            break

        penalty, grad_pos, grad_rad = _compute_penalty_and_grads(pos, radii)

        # Objective: maximize sum(radii) - penalty
        # d(obj)/d(pos) = -d(penalty)/d(pos) = grad_pos (our convention)
        # d(obj)/d(radii) = 1 - d(penalty)/d(radii) = 1 - grad_rad
        obj_grad_pos = grad_pos
        obj_grad_rad = 1.0 - grad_rad

        # Momentum
        vel_pos = beta * vel_pos + (1 - beta) * obj_grad_pos
        vel_rad = beta * vel_rad + (1 - beta) * obj_grad_rad

        pos += lr_pos * vel_pos
        radii += lr_rad * vel_rad

        # Clamp
        for i in range(n):
            radii[i] = max(0.005, min(0.49, radii[i]))
            pos[i, 0] = max(radii[i], min(1 - radii[i], pos[i, 0]))
            pos[i, 1] = max(radii[i], min(1 - radii[i], pos[i, 1]))

        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # Adaptive LR
        if it % 500 == 0 and it > 0:
            lr_pos *= 0.95
            lr_rad *= 0.95

        if no_improve_count > max_no_improve and penalty < 1e-10:
            break

    return best_pos, best_radii


def _coordinate_descent(pos0, radii0, time_budget=20.0, start_time=None, rng=None):
    """
    Coordinate descent: iterate over circles, adjust each one's position and radius
    to maximize its contribution while maintaining feasibility with others.
    """
    if start_time is None:
        start_time = time.time()
    if rng is None:
        rng = np.random.RandomState(int(time.time() * 1000) % (2**31))

    n = len(pos0)
    pos = pos0.copy()
    radii = radii0.copy()

    best_score = -1e9
    best_pos = pos.copy()
    best_radii = radii.copy()

    if _check_validity(pos, radii):
        best_score = np.sum(radii)
        best_pos = pos.copy()
        best_radii = radii.copy()

    for outer in range(200):
        if time.time() - start_time > time_budget:
            break

        for i in range(n):
            if time.time() - start_time > time_budget:
                    break

            # Find max radius for circle i given current positions of others
            max_r = 0.49
            # Boundary limits
            max_r = min(max_r, pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            # Overlap limits
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])

            # Try to grow radius
            target_r = min(0.49, max(0.005, max_r * 0.999))
            if target_r > radii[i]:
                radii[i] = target_r

            # Now try to reposition to allow even more growth
            # Simple: nudge toward center of available space
            # Compute centroid of feasible region (rough)
            # Just try small random nudges
            for _ in range(5):
                if time.time() - start_time > time_budget:
                    break
                orig_x, orig_y, orig_r = pos[i, 0], pos[i, 1], radii[i]
                pos[i, 0] += rng.uniform(-0.02, 0.02)
                pos[i, 1] += rng.uniform(-0.02, 0.02)
                pos[i, 0] = max(0.005, min(0.995, pos[i, 0]))
                pos[i, 1] = max(0.005, min(0.995, pos[i, 1]))

                # Recompute max_r at new position
                new_max_r = min(0.49, pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
                for j in range(n):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    new_max_r = min(new_max_r, dist - radii[j])

                new_r = min(0.49, max(0.005, new_max_r * 0.999))
                if new_r > radii[i]:
                    radii[i] = new_r
                else:
                    # Revert
                    pos[i, 0], pos[i, 1], radii[i] = orig_x, orig_y, orig_r
                    break

        # Check validity and update best
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()

    return best_pos, best_radii


def _simulated_annealing(pos0, radii0, time_budget=30.0, start_time=None, rng=None):
    """
    Simulated annealing with coordinate descent refinement.
    """
    if start_time is None:
        start_time = time.time()
    if rng is None:
        rng = np.random.RandomState(int(time.time() * 1000) % (2**31))

    n = len(pos0)
    pos = pos0.copy()
    radii = radii0.copy()

    best_score = -1e9
    best_pos = pos.copy()
    best_radii = radii.copy()

    if _check_validity(pos, radii):
        best_score = np.sum(radii)
        best_pos = pos.copy()
        best_radii = radii.copy()

    T_init = 0.03
    T_final = 0.0005
    cooling = 0.985
    T = T_init

    while time.time() - start_time < time_budget and T > T_final:
        # Perturb 1-3 circles
        n_perturb = rng.randint(1, 4)
        indices = rng.choice(n, size=min(n_perturb, n), replace=False)
        for idx in indices:
            step = T * 0.5
            pos[idx, 0] += rng.normal(0, step)
            pos[idx, 1] += rng.normal(0, step)
            pos[idx, 0] = max(radii[idx], min(1 - radii[idx], pos[idx, 0]))
            pos[idx, 1] = max(radii[idx], min(1 - radii[idx], pos[idx, 1]))
            radii[idx] += rng.normal(0, T * 0.08)
            radii[idx] = max(0.005, min(0.49, radii[idx]))
            pos[idx, 0] = max(radii[idx], min(1 - radii[idx], pos[idx, 0]))
            pos[idx, 1] = max(radii[idx], min(1 - radii[idx], pos[idx, 1]))

        # Refine with coordinate descent (short budget)
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 0.3:
            break
        refine_budget = min(0.8, remaining * 0.2)
        pos, radii = _coordinate_descent(pos, radii, time_budget=refine_budget,
                                          start_time=start_time, rng=rng)

        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()

        T *= cooling

    return best_pos, best_radii


def _build_hex(n):
    """Build a hexagonal initial configuration."""
    pos_list = []
    row = 0
    while len(pos_list) < n:
        cols = 6
        for col in range(cols):
            if len(pos_list) >= n:
                break
            x = (col + (0.5 if row % 2 else 0)) * 0.17
            y = (row + 0.5) * 0.149
            pos_list.append([x, y])
        row += 1
    pos = np.array(pos_list[:n])
    r = 0.08
    radii = np.full(n, r)
    return pos, radii


def _build_grid(n):
    """Build a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    r = 0.42 / side
    radii = np.full(n, r)
    return pos, radii


def _build_random(n, seed=0):
    """Build a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.07)
    return pos, radii


def _build_corner(n):
    """Place circles at corners and edges for a structured start."""
    pos_list = []
    r_corner = 0.12
    pos_list.append([r_corner, r_corner])
    pos_list.append([1 - r_corner, r_corner])
    pos_list.append([r_corner, 1 - r_corner])
    pos_list.append([1 - r_corner, 1 - r_corner])

    r_edge = 0.1
    pos_list.append([0.5, r_edge])
    pos_list.append([0.5, 1 - r_edge])
    pos_list.append([r_edge, 0.5])
    pos_list.append([1 - r_edge, 0.5])

    remaining = n - len(pos_list)
    if remaining > 0:
        side = int(np.ceil(np.sqrt(remaining)))
        for j in range(side):
            for i in range(side):
                if len(pos_list) >= n:
                    break
                x = 0.25 + (i + 0.5) * 0.5 / side
                y = 0.25 + (j + 0.5) * 0.5 / side
                pos_list.append([x, y])

    pos = np.array(pos_list[:n])
    radii = np.full(n, 0.08)
    return pos, radii


def _relax(pos, radii, max_iters=3000, time_budget=10.0, start_time=None):
    """Iterative overlap resolution with boundary clamping."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()

    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break

        for i in range(n):
            pos[i, 0] = max(radii[i], min(1 - radii[i], pos[i, 0]))
            pos[i, 1] = max(radii[i], min(1 - radii[i], pos[i, 1]))

        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]

                if dist < min_dist:
                    changed = True
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2
                        pos[i, 0] -= push * np.cos(angle)
                        pos[i, 1] -= push * np.sin(angle)
                        pos[j, 0] += push * np.cos(angle)
                        pos[j, 1] += push * np.sin(angle)
                    else:
                        push = (min_dist - dist) / 2
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] -= push * nx
                        pos[i, 1] -= push * ny
                        pos[j, 0] += push * nx
                        pos[j, 1] += push * ny

        if not changed:
            break

    for i in range(n):
        pos[i, 0] = max(radii[i], min(1 - radii[i], pos[i, 0]))
        pos[i, 1] = max(radii[i], min(1 - radii[i], pos[i, 1]))

    return pos, radii


def _grow_radii(pos, radii, time_budget=15.0, start_time=None, rng=None):
    """
    Iteratively try to grow all radii uniformly while maintaining feasibility.
    Uses binary search on the growth factor.
    """
    if start_time is None:
        start_time = time.time()
    if rng is None:
        rng = np.random.RandomState(int(time.time() * 1000) % (2**31))

    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()

    # Try to find max uniform scale factor
    lo, hi = 1.0, 2.0
    for _ in range(30):
        if time.time() - start_time > time_budget:
            break
        mid = (lo + hi) / 2
        test_radii = radii * mid
        # Check if we can place circles with these radii
        # Just check boundary feasibility
        feasible = True
        for i in range(n):
            if test_radii[i] > 0.49:
                feasible = False
                break
            if pos[i, 0] < test_radii[i] or pos[i, 0] > 1 - test_radii[i]:
                feasible = False
                break
            if pos[i, 1] < test_radii[i] or pos[i, 1] > 1 - test_radii[i]:
                feasible = False
                break
        if feasible:
            # Check overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < test_radii[i] + test_radii[j]:
                        feasible = False
                        break
                if not feasible:
                    break
        if feasible:
            lo = mid
        else:
            hi = mid

    scale = lo
    if scale > 1.001:
        radii = radii * scale
        # Re-relax
        pos, radii = _relax(pos, radii, max_iters=1000, time_budget=min(5.0, time_budget - (time.time() - start_time)), start_time=start_time)
        if _check_validity(pos, radii):
            return pos, radii

    return pos, radii


def construct():
    start_time = time.time()
    deadline = 140.0

    best_score = 0.0
    best_pos = None
    best_radii = None

    configs = []

    pos, radii = _build_hex(N)
    configs.append((pos.copy(), radii.copy(), "hex"))

    pos, radii = _build_grid(N)
    configs.append((pos.copy(), radii.copy(), "grid"))

    pos, radii = _build_corner(N)
    configs.append((pos.copy(), radii.copy(), "corner"))

    for seed in [42, 123, 456, 789, 1000, 2024, 31415]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos.copy(), radii.copy(), f"rand{seed}"))

    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))

    for ci, (pos, radii, name) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 8:
            break

        remaining = deadline - elapsed - 3

        # Phase 1: Relax overlaps
        time_alloc = min(5.0, remaining * 0.15)
        if time_alloc > 0.5:
            pos, radii = _relax(pos, radii, max_iters=3000,
                                 time_budget=time_alloc, start_time=start_time)

        # Phase 2: Coordinate descent (new structural component)
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 3
        time_alloc = min(15.0, remaining * 0.35)
        if time_alloc > 1:
            pos, radii = _coordinate_descent(pos, radii, time_budget=time_alloc,
                                              start_time=start_time, rng=rng)

        # Phase 3: Gradient optimization
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 3
        time_alloc = min(15.0, remaining * 0.3)
        if time_alloc > 1:
            pos, radii = _gradient_optimize(pos, radii, time_budget=time_alloc,
                                             start_time=start_time, rng=rng)

        # Phase 4: Grow radii
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 3
        time_alloc = min(8.0, remaining * 0.2)
        if time_alloc > 1:
            pos, radii = _grow_radii(pos, radii, time_budget=time_alloc,
                                      start_time=start_time, rng=rng)

        # Phase 5: Simulated annealing
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 3
        time_alloc = min(20.0, remaining * 0.4)
        if time_alloc > 1:
            pos, radii = _simulated_annealing(pos, radii, time_budget=time_alloc,
                                               start_time=start_time, rng=rng)

        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()

    # Final polish
    if best_pos is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 2
        if remaining > 5:
            backup_pos = best_pos.copy()
            backup_radii = best_radii.copy()
            new_pos, new_radii = _coordinate_descent(best_pos, best_radii,
                                                      time_budget=min(10.0, remaining * 0.4),
                                                      start_time=start_time, rng=rng)
            if _check_validity(new_pos, new_radii):
                new_score = np.sum(new_radii)
                if new_score > best_score:
                    best_pos = new_pos
                    best_radii = new_radii
                    best_score = new_score
            else:
                best_pos, best_radii = backup_pos, backup_radii

        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 2
        if remaining > 5:
            backup_pos = best_pos.copy()
            backup_radii = best_radii.copy()
            new_pos, new_radii = _grow_radii(best_pos, best_radii,
                                              time_budget=min(10.0, remaining * 0.5),
                                              start_time=start_time, rng=rng)
            if _check_validity(new_pos, new_radii):
                new_score = np.sum(new_radii)
                if new_score > best_score:
                    best_pos = new_pos
                    best_radii = new_radii
                    best_score = new_score
            else:
                best_pos, best_radii = backup_pos, backup_radii

    if best_pos is None:
        pos, radii = _build_grid(N)
        best_pos, best_radii = pos, radii

    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]

    return result_arr
# EVOLVE-BLOCK-END
