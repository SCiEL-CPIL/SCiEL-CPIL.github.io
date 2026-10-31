"""Circle packing in perimeter-4 rectangle: Simulated Annealing + Vectorized Expand polish."""
import numpy as np
import time
import math

N = 21
# Precompute upper-triangle index pairs for vectorized distance computation
_IU = np.triu_indices(N, k=1)


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s kill

    def check_validity(x, y, r, w, h, tol=1e-7):
        if w <= 0.01 or h <= 0.01:
            return False, 0.0
        if np.any(r < -tol):
            return False, 0.0
        r = np.maximum(r, 0.0)
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        # Vectorized overlap check
        dx = x[_IU[0]] - x[_IU[1]]
        dy = y[_IU[0]] - y[_IU[1]]
        d2 = dx * dx + dy * dy
        r_sum = r[_IU[0]] + r[_IU[1]]
        if np.any(d2 < r_sum * r_sum - tol):
            return False, 0.0
        return True, float(np.sum(r))

    def expand_to_max(x, y, r, w, h, max_rounds=80):
        """Iteratively expand each circle to its maximum feasible radius."""
        x = x.copy()
        y = y.copy()
        r = np.maximum(r, 0.0).copy()
        for _ in range(max_rounds):
            improved = False
            for i in range(N):
                # Wall constraint
                r_max = min(x[i], w - x[i], y[i], h - y[i])
                # Other circles
                dx = x[i] - x
                dy = y[i] - y
                d = np.sqrt(dx * dx + dy * dy)
                d[i] = np.inf
                r_allowed = d - r
                r_allowed[i] = np.inf
                r_max = min(r_max, np.min(r_allowed))
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-10:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return x, y, r

    def sa_packing(w0, h0, time_budget=20.0, seed=42):
        """Simulated annealing to maximize sum of radii in rectangle w0 x h0."""
        rng = np.random.default_rng(seed)
        n = N

        # Initialize: grid layout with small radii
        side = int(math.ceil(math.sqrt(n)))
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)
        for i in range(n):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w0
            y[i] = (row + 0.5) / side * h0
            r[i] = min(w0 / side, h0 / side) * 0.35

        # Clamp to boundaries
        x = np.clip(x, 0.01, w0 - 0.01)
        y = np.clip(y, 0.01, h0 - 0.01)

        # Temperature schedule
        T_start = 0.05
        T_end = 0.001
        # We'll run until time budget
        t_sa = start_time
        T = T_start
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_val = -np.sum(r)  # we minimize negative sum

        # Penalty weight for overlaps
        penalty_w = 200.0

        while time.time() - t_sa < time_budget:
            # Compute current energy
            dx = x[_IU[0]] - x[_IU[1]]
            dy = y[_IU[0]] - y[_IU[1]]
            d = np.sqrt(dx * dx + dy * dy)
            r_sum = r[_IU[0]] + r[_IU[1]]
            overlap = np.maximum(0.0, r_sum - d)
            # Boundary violations
            bv = 0.0
            bv += np.sum(np.maximum(0.0, r - x) ** 2)
            bv += np.sum(np.maximum(0.0, x + r - w0) ** 2)
            bv += np.sum(np.maximum(0.0, r - y) ** 2)
            bv += np.sum(np.maximum(0.0, y + r - h0) ** 2)
            energy = -np.sum(r) + penalty_w * (np.sum(overlap * overlap) + bv)

            if energy < best_val:
                best_val = energy
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

            # Pick a random circle and perturb it
            i = rng.integers(n)
            # Perturbation magnitude scales with temperature
            mag = T * 0.5
            dx_p = rng.uniform(-mag, mag)
            dy_p = rng.uniform(-mag, mag)
            dr_p = rng.uniform(-mag * 0.3, mag * 0.3)

            new_x = x[i] + dx_p
            new_y = y[i] + dy_p
            new_r = r[i] + dr_p

            # Clamp
            new_x = np.clip(new_x, 0.005, w0 - 0.005)
            new_y = np.clip(new_y, 0.005, h0 - 0.005)
            new_r = max(0.01, min(new_r, min(new_x, w0 - new_x, new_y, h0 - new_y)))

            # Compute energy change (only affected pairs)
            # New distances for circle i
            dx_new = new_x - x
            dy_new = new_y - y
            d_new = np.sqrt(dx_new * dx_new + dy_new * dy_new)
            d_new[i] = np.inf
            r_sum_new = new_r + r
            r_sum_new[i] = np.inf
            overlap_new_i = np.maximum(0.0, r_sum_new - d_new)
            overlap_new_i[i] = 0.0
            new_overlap_sum = np.sum(overlap_new_i * overlap_new_i)

            # Old overlap sum for pairs involving i
            old_overlap_sum = 0.0
            for j in range(n):
                if j == i:
                    continue
                d_old = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                rs = r[i] + r[j]
                ov = max(0.0, rs - d_old)
                old_overlap_sum += ov * ov

            # Boundary
            bv_old = 0.0
            bv_old += max(0.0, r[i] - x[i]) ** 2
            bv_old += max(0.0, x[i] + r[i] - w0) ** 2
            bv_old += max(0.0, r[i] - y[i]) ** 2
            bv_old += max(0.0, y[i] + r[i] - h0) ** 2

            bv_new = 0.0
            bv_new += max(0.0, new_r - new_x) ** 2
            bv_new += max(0.0, new_x + new_r - w0) ** 2
            bv_new += max(0.0, new_r - new_y) ** 2
            bv_new += max(0.0, new_y + new_r - h0) ** 2

            # Unchanged pairs overlap
            # Total old overlap for all pairs
            # We only changed pairs involving i, so:
            # delta_overlap = new_overlap_sum (for i's pairs) - old_overlap_sum (for i's pairs)
            delta_overlap = new_overlap_sum - old_overlap_sum
            delta_bv = bv_new - bv_old
            delta_r = new_r - r[i]

            delta_e = -delta_r + penalty_w * (delta_overlap + delta_bv)

            # Accept
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                x[i] = new_x
                y[i] = new_y
                r[i] = new_r

            # Cool down
            T *= 0.9995
            if T < T_end:
                T = T_end

        # Final: expand to max
        x, y, r = expand_to_max(best_x, best_y, best_r, w0, h0)
        return x, y, r, w0, h0

    def local_shake(x, y, r, w, h, time_budget=5.0, seed=99):
        """Small random perturbations to escape local optima, then re-expand."""
        rng = np.random.default_rng(seed)
        n = N
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_val = float(np.sum(r))

        t0 = time.time()
        while time.time() - t0 < time_budget:
            # Pick 2-3 random circles and nudge them
            for _ in range(rng.integers(2, 5)):
                i = rng.integers(n)
                dx = rng.uniform(-0.01, 0.01)
                dy = rng.uniform(-0.01, 0.01)
                x[i] = np.clip(x[i] + dx, 0.005, w - 0.005)
                y[i] = np.clip(y[i] + dy, 0.005, h - 0.005)
                # Slightly shrink to allow movement
                r[i] *= 0.98

            # Re-expand
            x2, y2, r2 = expand_to_max(x, y, r, w, h, max_rounds=40)
            val = float(np.sum(r2))
            if val > best_val:
                best_val = val
                best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
                x, y, r = x2, y2, r2
            else:
                # Revert
                x, y, r = best_x.copy(), best_y.copy(), best_r.copy()

        return best_x, best_y, best_r

    # Main: try multiple aspect ratios with SA
    best_value = 0.0
    best_result = None

    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.75, 1.4, 0.7, 1.5, 0.65, 1.6, 0.6, 1.7, 0.55, 1.8, 0.5, 2.0, 0.45, 2.2, 0.4, 2.5, 0.35, 3.0, 0.3]

    # Sort by aspect ratio to try diverse ones
    rng_main = np.random.default_rng(7)
    # Shuffle for diversity
    indices = list(range(len(aspect_ratios)))
    rng_main.shuffle(indices)

    # Prioritize square-ish first (known good region), then others
    # Actually, let's try them in a mixed order
    for idx in indices:
        if time.time() > deadline - 20:
            break

        w0 = aspect_ratios[idx]
        h0 = 2.0 - w0
        if h0 < 0.25 or w0 < 0.25:
            continue

        remaining = deadline - time.time()
        if remaining < 8:
            break

        # Time budget for SA: use most of remaining, leave 15s for polish
        sa_time = min(remaining - 15, 35.0)
        if sa_time < 5:
            continue

        seed = 42 + idx * 7
        x, y, r, w, h = sa_packing(w0, h0, time_budget=sa_time, seed=seed)

        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

        # Local shake on the best so far if we have time
        if best_result is not None and time.time() < deadline - 10:
            bx, by, br, bw, bh = best_result
            shake_time = min(5.0, deadline - time.time() - 5)
            if shake_time > 2:
                bx2, by2, br2 = local_shake(bx, by, br, bw, bh, time_budget=shake_time, seed=200 + idx)
                valid2, val2 = check_validity(bx2, by2, br2, bw, bh)
                if valid2 and val2 > best_value:
                    best_value = val2
                    best_result = (bx2.copy(), by2.copy(), br2.copy(), bw, bh)

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 5:
        x, y, r, w, h = best_result
        x, y, r = expand_to_max(x, y, r, w, h, max_rounds=100)
        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    if best_result is None:
        # Fallback: simple grid in square
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r0 = 0.5 / side - 0.005
            cells.append((cx, cy, r0))
        return np.array(cells)

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
