"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    best_sum = 0.0
    best_result = None

    def check_validity(positions, radii, w, h):
        """Check if packing is valid and return sum of radii if so."""
        if np.any(positions[:, 0] < radii - 1e-9):
            return -1.0
        if np.any(positions[:, 0] > w - radii + 1e-9):
            return -1.0
        if np.any(positions[:, 1] < radii - 1e-9):
            return -1.0
        if np.any(positions[:, 1] > h - radii + 1e-9):
            return -1.0

        dx = positions[:, 0][:, None] - positions[:, 0][None, :]
        dy = positions[:, 1][:, None] - positions[:, 1][None, :]
        d = np.sqrt(dx * dx + dy * dy)
        r_sum = radii[:, None] + radii[None, :]
        if np.any(d < r_sum - 1e-9):
            return -1.0

        return np.sum(radii)

    def smooth_objective(z, penalty_weight=500.0):
        """
        Smooth vectorized objective with penalty terms.
        z: [x0, y0, r0, x1, y1, r1, ..., x_{N-1}, y_{N-1}, r_{N-1}, w]
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2:
            return 1e8

        r_clipped = np.clip(r, 0.001, min(w, h) / 2 - 0.001)

        obj = -np.sum(r_clipped)

        # Pairwise overlap penalty (smooth)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_clipped[:, None] + r_clipped[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-15))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        overlap_penalty = penalty_weight * np.sum(overlap ** 2)

        # Wall constraint penalties (vectorized)
        x_low_viol = np.maximum(r_clipped - x, 0.0)
        x_high_viol = np.maximum(x - (w - r_clipped), 0.0)
        y_low_viol = np.maximum(r_clipped - y, 0.0)
        y_high_viol = np.maximum(y - (h - r_clipped), 0.0)

        wall_penalty = penalty_weight * (np.sum(x_low_viol ** 2) + np.sum(x_high_viol ** 2) +
                                         np.sum(y_low_viol ** 2) + np.sum(y_high_viol ** 2))

        small_r_penalty = penalty_weight * np.sum(np.maximum(0.01 - r_clipped, 0.0) ** 2)

        return obj + overlap_penalty + wall_penalty + small_r_penalty

    def force_relax(positions, radii, w, h, n_iters=200, strength=0.3):
        """Push overlapping circles apart using force-directed relaxation."""
        positions = positions.copy().astype(float)
        radii = radii.copy().astype(float)

        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
            forces = np.zeros_like(positions)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < 1e-12:
                        dist = 1e-12
                        dx, dy = 1e-6, 0.0
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        fx = strength * overlap * dx / dist
                        fy = strength * overlap * dy / dist
                        forces[i] += [fx, fy]
                        forces[j] -= [fx, fy]

            # Wall forces
            for i in range(N):
                if positions[i, 0] < radii[i]:
                    forces[i, 0] += strength * (radii[i] - positions[i, 0])
                if positions[i, 0] > w - radii[i]:
                    forces[i, 0] -= strength * (positions[i, 0] - (w - radii[i]))
                if positions[i, 1] < radii[i]:
                    forces[i, 1] += strength * (radii[i] - positions[i, 1])
                if positions[i, 1] > h - radii[i]:
                    forces[i, 1] -= strength * (positions[i, 1] - (h - radii[i]))

            positions += forces * 0.5

            # Clamp to walls
            positions[:, 0] = np.clip(positions[:, 0], radii, w - radii)
            positions[:, 1] = np.clip(positions[:, 1], radii, h - radii)

        return positions

    def get_grid_initialization(rows, cols, w, h):
        """
        Grid-based cellular decomposition: partition rectangle into rows x cols cells,
        assign circles to cells, and set initial radii based on cell inradius.
        """
        positions = np.zeros((N, 2))
        radii = np.zeros(N)

        cell_w = w / cols
        cell_h = h / rows

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N:
                    break
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                max_r = min(cell_w, cell_h) / 2.0 * 0.90

                remaining_circles = N - idx
                remaining_cells = (rows - r) * cols - c

                if remaining_circles > remaining_cells:
                    r1 = max_r * 0.55
                    r2 = max_r * 0.35
                    offset_x = (r1 + r2) * 0.6

                    if cell_w > cell_h:
                        positions[idx, 0] = cx - offset_x
                        positions[idx, 1] = cy
                        radii[idx] = max(r1, 0.01)
                        idx += 1
                        if idx < N:
                            positions[idx, 0] = cx + offset_x
                            positions[idx, 1] = cy
                            radii[idx] = max(r2, 0.01)
                            idx += 1
                    else:
                        positions[idx, 0] = cx
                        positions[idx, 1] = cy - (r1 + r2) * 0.3
                        radii[idx] = max(r1, 0.01)
                        idx += 1
                        if idx < N:
                            positions[idx, 0] = cx
                            positions[idx, 1] = cy + (r1 + r2) * 0.3
                            radii[idx] = max(r2, 0.01)
                            idx += 1
                else:
                    positions[idx, 0] = cx
                    positions[idx, 1] = cy
                    radii[idx] = max(max_r, 0.01)
                    idx += 1

        while idx < N:
            positions[idx, 0] = w / 2.0
            positions[idx, 1] = h / 2.0
            radii[idx] = 0.01
            idx += 1

        return positions, radii

    def get_initial_guesses():
        """Generate diverse initial guesses using grid-based cellular decomposition."""
        guesses = []

        grid_configs = [
            (3, 7), (7, 3),
            (4, 6), (6, 4),
            (5, 5),
            (3, 8), (8, 3),
            (4, 5), (5, 4),
            (2, 11), (11, 2),
            (3, 5), (5, 3),
        ]

        aspect_ratios = [
            1.0, 1.2, 0.8, 1.5, 0.5, 1.3, 0.7, 1.4, 0.6, 1.1, 0.9,
            1.6, 0.4, 1.7, 0.3, 1.8, 0.25
        ]

        for w_frac in aspect_ratios:
            w = w_frac
            h = 2.0 - w
            if w < 0.2 or h < 0.2:
                continue

            for rows, cols in grid_configs:
                positions, radii = get_grid_initialization(rows, cols, w, h)

                score = check_validity(positions, radii, w, h)
                if score > 0:
                    guesses.append((w, h, positions, radii))

        return guesses

    def polish_result(w, h, positions, radii, max_iter=300, ftol=1e-12):
        """Polish a valid packing using SLSQP with smooth objective."""
        z0 = np.concatenate([positions.flatten(), [w]])

        bounds = []
        for i in range(N):
            bounds.append((max(0.001, positions[i, 0] - 0.02), min(w - 0.001, positions[i, 0] + 0.02)))
            bounds.append((max(0.001, positions[i, 1] - 0.02), min(h - 0.001, positions[i, 1] + 0.02)))
            bounds.append((max(0.005, radii[i] * 0.9), min(min(w, h) / 2.0 - 0.001, radii[i] * 1.15)))
        bounds.append((max(0.25, w - 0.05), min(1.75, w + 0.05)))

        try:
            result = minimize(
                lambda z: smooth_objective(z),
                z0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol}
            )

            if result.fun < 0:
                z_opt = result.x
                x_opt = z_opt[0::3][:N]
                y_opt = z_opt[1::3][:N]
                r_opt = z_opt[2::3][:N]
                w_opt = z_opt[-1]
                h_opt = 2.0 - w_opt

                positions_opt = np.column_stack([x_opt, y_opt])

                score = check_validity(positions_opt, r_opt, w_opt, h_opt)
                if score > 0:
                    return positions_opt, r_opt, w_opt, h_opt, score
        except Exception:
            pass

        return None

    def high_precision_polish(w, h, positions, radii, max_iter=800, ftol=1e-14):
        """Final high-precision polish pass with tighter tolerance."""
        z0 = np.concatenate([positions.flatten(), [w]])

        bounds = []
        for i in range(N):
            bounds.append((max(0.001, positions[i, 0] - 0.03), min(w - 0.001, positions[i, 0] + 0.03)))
            bounds.append((max(0.001, positions[i, 1] - 0.03), min(h - 0.001, positions[i, 1] + 0.03)))
            bounds.append((max(0.005, radii[i] * 0.95), min(min(w, h) / 2.0 - 0.001, radii[i] * 1.1)))
        bounds.append((max(0.25, w - 0.05), min(1.75, w + 0.05)))

        try:
            result = minimize(
                lambda z: smooth_objective(z, penalty_weight=2000.0),
                z0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol}
            )

            if result.fun < 0:
                z_opt = result.x
                x_opt = z_opt[0::3][:N]
                y_opt = z_opt[1::3][:N]
                r_opt = z_opt[2::3][:N]
                w_opt = z_opt[-1]
                h_opt = 2.0 - w_opt

                positions_opt = np.column_stack([x_opt, y_opt])

                score = check_validity(positions_opt, r_opt, w_opt, h_opt)
                if score > 0:
                    return positions_opt, r_opt, w_opt, h_opt, score
        except Exception:
            pass

        return None

    def vectorized_expansion(w, h, positions, radii, max_rounds=40):
        """Vectorized radius expansion with position search."""
        positions = positions.copy().astype(float)
        radii = radii.copy().astype(float)

        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                old_r = radii[i]
                max_possible = min(w, h) / 2.0 - 0.001
                for dr in [0.004, 0.002, 0.001, 0.0005]:
                    new_r = old_r + dr
                    if new_r > max_possible:
                        continue

                    # Candidate positions: current + grid of offsets
                    offsets = np.array([[0, 0], [0.004, 0], [-0.004, 0],
                                        [0, 0.004], [0, -0.004],
                                        [0.004, 0.004], [-0.004, -0.004],
                                        [0.008, 0], [-0.008, 0],
                                        [0, 0.008], [0, -0.008]])
                    cand_x = positions[i, 0] + offsets[:, 0]
                    cand_y = positions[i, 1] + offsets[:, 1]

                    # Wall check
                    valid_mask = (cand_x >= new_r) & (cand_x <= w - new_r) & \
                                 (cand_y >= new_r) & (cand_y <= h - new_r)

                    found = False
                    for k in range(len(cand_x)):
                        if not valid_mask[k]:
                            continue
                        nx, ny = cand_x[k], cand_y[k]
                        # Check against all other circles
                        dx_o = nx - positions[:, 0]
                        dy_o = ny - positions[:, 1]
                        dists = np.sqrt(dx_o * dx_o + dy_o * dy_o)
                        dists[i] = np.inf
                        if np.all(dists >= new_r + radii - 1e-9):
                            positions[i, 0] = nx
                            positions[i, 1] = ny
                            radii[i] = new_r
                            improved = True
                            found = True
                            break
                    if found:
                        break

            if not improved:
                break

        score = check_validity(positions, radii, w, h)
        if score > 0:
            return positions, radii, w, h, score
        return None

    # Main search loop
    guesses = get_initial_guesses()
    guesses.sort(key=lambda g: -np.sum(g[3]))

    for w, h, positions, radii in guesses:
        if time.time() > deadline - 10.0:
            break

        # Force relax to fix any minor overlaps before optimization
        pos_relaxed = force_relax(positions, radii, w, h)
        score_relaxed = check_validity(pos_relaxed, radii, w, h)
        if score_relaxed > 0:
            positions = pos_relaxed

        # Try polishing
        result = polish_result(w, h, positions, radii)
        if result is not None:
            pos_opt, r_opt, w_opt, h_opt, score = result
            if score > best_sum:
                best_sum = score
                best_result = np.column_stack([pos_opt, r_opt])

                # Try high-precision polish on the best result
                if time.time() < deadline - 15.0:
                    hp_result = high_precision_polish(w_opt, h_opt, pos_opt, r_opt)
                    if hp_result is not None:
                        pos_hp, r_hp, w_hp, h_hp, score_hp = hp_result
                        if score_hp > best_sum:
                            best_sum = score_hp
                            best_result = np.column_stack([pos_hp, r_hp])

        # Try vectorized expansion
        if time.time() < deadline - 5.0:
            result2 = vectorized_expansion(w, h, positions, radii)
            if result2 is not None:
                pos2, r2, w2, h2, score2 = result2
                if score2 > best_sum:
                    best_sum = score2
                    best_result = np.column_stack([pos2, r2])

    # If we haven't found anything good, return a simple grid solution
    if best_result is None:
        w, h = 1.0, 1.0
        positions, radii = get_grid_initialization(5, 5, w, h)
        best_result = np.column_stack([positions, radii])

    return best_result
# EVOLVE-BLOCK-END
