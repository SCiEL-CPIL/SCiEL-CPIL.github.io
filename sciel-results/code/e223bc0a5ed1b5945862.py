"""
Circle packing in unit square using a structural shift:
  1. Multi-start with diverse initial layouts (grid, hexagonal, random, golden-ratio)
  2. Fast greedy radius maximisation (O(N^2) per pass, cheap)
  3. Position-based local search with adaptive step size
  4. SLSQP refinement on the continuous landscape (small, targeted)
  5. Robust validation and repair
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _grid_positions(n, rows=None, cols=None):
    """Return (n,2) grid positions inside [0,1]^2."""
    if rows is None or cols is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    positions = np.zeros((n, 2))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            positions[idx, 0] = (j + 0.5) / cols
            positions[idx, 1] = (i + 0.5) / rows
            idx += 1
    return positions


def _hex_positions(n):
    """Return (n,2) hexagonal-ish positions inside [0,1]^2."""
    # Hexagonal packing: rows offset by half spacing
    # For 26 circles, try 5 rows: 5,6,5,6,4 = 26
    row_sizes = [5, 6, 5, 6, 4]
    positions = []
    for r_idx, count in enumerate(row_sizes):
        if len(positions) >= n:
            break
        y = (r_idx + 0.5) / len(row_sizes)
        if r_idx % 2 == 0:
            # even row
            for k in range(count):
                x = (k + 0.5) / count
                positions.append([x, y])
        else:
            # odd row, offset
            for k in range(count):
                x = (k + 0.5) / count
                positions.append([x, y])
    # Trim or pad
    while len(positions) < n:
        positions.append([0.5, 0.5])
    return np.array(positions[:n])


def _random_positions(n, seed):
    """Return (n,2) random positions with minimum separation."""
    rng = np.random.RandomState(seed)
    positions = np.zeros((n, 2))
    min_dist = 0.08
    for i in range(n):
        for attempt in range(200):
            pos = rng.uniform(0.1, 0.9, size=2)
            if i == 0:
                positions[i] = pos
                break
            ok = True
            for j in range(i):
                d = np.linalg.norm(pos - positions[j])
                if d < min_dist:
                    ok = False
                    break
            if ok:
                positions[i] = pos
                break
        else:
            positions[i] = pos  # accept last attempt
    return positions


def _golden_positions(n, seed):
    """Golden-ratio based positions for quasi-random distribution."""
    rng = np.random.RandomState(seed)
    positions = np.zeros((n, 2))
    golden = (1 + np.sqrt(5)) / 2
    for i in range(n):
        # Use golden angle for x, random for y with jitter
        x = (i * (1.0 / golden) + rng.uniform(-0.02, 0.02)) % 1.0
        x = 0.1 + 0.8 * x
        y = 0.1 + 0.8 * ((i * 0.61803 + rng.uniform(-0.05, 0.05)) % 1.0)
        positions[i] = [x, y]
    return positions


def _max_radius_for_circle(pos, radii, i, n):
    """Compute max possible radius for circle i given current positions and other radii."""
    x, y = pos[i]
    r_bound = min(x, y, 1.0 - x, 1.0 - y)
    if r_bound <= 0:
        return 0.0
    r_max = r_bound
    for j in range(n):
        if i == j:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        r_candidate = dist - radii[j]
        if r_candidate < r_max:
            r_max = r_candidate
    return max(0.0, r_max)


def _greedy_maximize_radii(pos, n, max_passes=30, deadline=None):
    """Iteratively set each radius to its maximum given current positions."""
    radii = np.zeros(n)
    # Initialize with boundary-limited radii
    for i in range(n):
        radii[i] = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
    radii = np.maximum(radii, 0.0)

    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            new_r = _max_radius_for_circle(pos, radii, i, n)
            if new_r > radii[i] + 1e-9:
                radii[i] = new_r
                improved = True
        if not improved:
            break
    return radii


def _local_search_positions(pos, radii, n, rng, deadline, max_iters=300):
    """
    Local search: randomly perturb a single circle's position,
    re-run greedy radius maximisation, accept if sum of radii improves.
    """
    pos = pos.copy()
    radii = radii.copy()
    current_sum = np.sum(radii)
    best_sum = current_sum
    step_size = 0.04
    no_improve = 0

    for it in range(max_iters):
        if time.time() > deadline:
            break

        i = rng.randint(n)
        orig = pos[i].copy()

        # Try several perturbations
        best_local_sum = current_sum
        best_local_pos = pos.copy()
        best_local_radii = radii.copy()

        for _ in range(8):
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            new_x = np.clip(orig[0] + dx, 0.0, 1.0)
            new_y = np.clip(orig[1] + dy, 0.0, 1.0)
            pos[i] = [new_x, new_y]

            trial_radii = _greedy_maximize_radii(pos, n, max_passes=10, deadline=deadline)
            trial_sum = np.sum(trial_radii)

            if trial_sum > best_local_sum + 1e-9:
                best_local_sum = trial_sum
                best_local_pos = pos.copy()
                best_local_radii = trial_radii.copy()

        # Restore original if no improvement found locally
        pos[i] = orig

        if best_local_sum > current_sum + 1e-9:
            pos = best_local_pos
            radii = best_local_radii
            current_sum = best_local_sum
            if current_sum > best_sum:
                best_sum = current_sum
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            if no_improve > 40:
                step_size *= 0.85
                no_improve = 0
                if step_size < 0.002:
                    break

    return pos, radii, best_sum


def _slsqp_refine(pos, radii, n, deadline):
    """
    Use SLSQP to refine positions and radii jointly.
    Objective: maximize sum(radii)
    Constraints:
      - x_i - r_i >= 0
      - y_i - r_i >= 0
      - x_i + r_i <= 1
      - y_i + r_i <= 1
      - dist(i,j) >= r_i + r_j
    """
    m = 2 * n + n  # x, y, r for each circle
    x0 = np.concatenate([pos.flatten(), radii])

    def objective(x):
        return -np.sum(x[2*n:])  # minimize negative sum of radii

    # Inequality constraints: g(x) >= 0
    cons = []
    # Boundary constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i + 1] - x[2*n + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i + 1] - x[2*n + i]})

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def overlap_constraint(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i + 1] - x[2*j + 1]
                return np.sqrt(dx*dx + dy*dy) - x[2*n + i] - x[2*n + j]
            cons.append({'type': 'ineq', 'fun': overlap_constraint})

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    try:
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        if result.success:
            new_pos = result.x[:2*n].reshape(n, 2)
            new_radii = result.x[2*n:]
            # Validate
            if _validate(new_pos, new_radii, n):
                return new_pos, new_radii
    except Exception:
        pass

    return pos, radii


def _validate(pos, radii, n, tol=1e-8):
    """Check validity: within bounds, no overlaps."""
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if r < -tol:
            return False
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _repair(pos, radii, n):
    """
    Repair a potentially invalid configuration by:
    1. Clamping radii to boundary limits.
    2. Iteratively reducing radii to resolve overlaps.
    3. If still invalid, fall back to greedy max radius.
    """
    pos = pos.copy()
    radii = radii.copy()

    # Clamp to boundary
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))

    # Fix overlaps
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                md = radii[i] + radii[j]
                if d < md - 1e-10:
                    excess = md - d
                    red = excess / 2.0 + 1e-7
                    radii[i] = max(0.0, radii[i] - red)
                    radii[j] = max(0.0, radii[j] - red)
                    changed = True
        if not changed:
            break

    # Final boundary clamp
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))

    if _validate(pos, radii, n):
        return pos, radii

    # Fallback: greedy max radius
    radii = _greedy_maximize_radii(pos, n, max_passes=20)
    if _validate(pos, radii, n):
        return pos, radii

    # Last resort: very small radii
    radii = np.full(n, 0.01)
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    best_result = None
    best_score = -np.inf

    # Generate diverse initial configurations
    configs = []

    # 1. Grid 5x5 (25) + 1 extra
    configs.append(_grid_positions(N, rows=5, cols=5))

    # 2. Grid 4x7
    configs.append(_grid_positions(N, rows=4, cols=7))

    # 3. Hexagonal
    configs.append(_hex_positions(N))

    # 4-7. Random with different seeds
    for seed in [0, 1, 2, 3]:
        configs.append(_random_positions(N, seed))

    # 8-9. Golden ratio
    for seed in [10, 20]:
        configs.append(_golden_positions(N, seed))

    # 10. Grid 3x9
    configs.append(_grid_positions(N, rows=3, cols=9))

    # 11. Grid 6x5
    configs.append(_grid_positions(N, rows=6, cols=5))

    for cfg_idx, pos in enumerate(configs):
        if time.time() > deadline:
            break

        # Ensure positions are within [0,1]
        pos = np.clip(pos, 0.0, 1.0)

        # Greedy radius maximisation
        radii = _greedy_maximize_radii(pos, N, max_passes=30, deadline=deadline)
        current_sum = np.sum(radii)

        # Local search on positions
        rng = np.random.RandomState(100 + cfg_idx)
        pos, radii, current_sum = _local_search_positions(
            pos, radii, N, rng, deadline, max_iters=200
        )

        # SLSQP refinement if time permits
        if time.time() < deadline - 5.0:
            pos, radii = _slsqp_refine(pos, radii, N, deadline)
            current_sum = np.sum(radii)

        # Repair and validate
        pos, radii = _repair(pos, radii, N)
        final_sum = np.sum(radii)

        if final_sum > best_score:
            best_score = final_sum
            best_result = np.column_stack([pos, radii])

    # Fallback: simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        best_result = np.array(cells[:N])

    return best_result

# EVOLVE-BLOCK-END
