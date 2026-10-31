"""Circle packing in perimeter-4 rectangle: Optimized SLSQP + Vectorized Force-Directed Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    best_value = 0.0
    best_result = None

    # --- Vectorized objective for penalty-based SLSQP ---
    penalty_weight = 1000.0

    def objective(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.25 or h < 0.25 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        # Vectorized pairwise overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        obj += penalty_weight * np.sum(overlap * overlap)

        # Vectorized boundary penalty
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)

        obj += penalty_weight * (
            np.sum(left_viol * left_viol) +
            np.sum(right_viol * right_viol) +
            np.sum(bottom_viol * bottom_viol) +
            np.sum(top_viol * top_viol)
        )

        # Keep aspect ratio in reasonable range
        if w < 0.5:
            obj += penalty_weight * (0.5 - w) ** 2
        elif w > 1.5:
            obj += penalty_weight * (w - 1.5) ** 2

        return obj

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    # --- Generate diverse initial guesses ---
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.1, 0.9, 0.7, 1.4, 0.65, 1.6]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

            # Hexagonal variant
            z_hex = np.zeros(3 * N + 1)
            hex_r = min(w0, h0) / (side * 1.8)
            hex_dx = hex_r * 2.0
            hex_dy = hex_r * np.sqrt(3)
            count = 0
            for row in range(side + 2):
                for col in range(side + 2):
                    if count >= N:
                        break
                    cx = col * hex_dx + (row % 2) * (hex_dx / 2) + hex_r
                    cy = row * hex_dy + hex_r
                    if cx + hex_r <= w0 and cy + hex_r <= h0:
                        z_hex[3 * count] = cx
                        z_hex[3 * count + 1] = cy
                        z_hex[3 * count + 2] = hex_r
                        count += 1
            if count == N:
                z_hex[-1] = w0
                guesses.append(z_hex)

        rng = np.random.default_rng(42)
        for _ in range(8):
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.11)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # --- Phase 1: Penalty-based SLSQP from multiple starts ---
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 15:
            break

        z0_relaxed = _force_relax_unified(z0, max_iter=50)

        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 15
        if remaining < 5:
            continue
        maxiter = int(min(200, max(50, remaining * 20)))

        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.93
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Fast vectorized force-directed polish ---
    if best_result is not None and time.time() < deadline - 10:
        polished = _fast_polish(best_result, deadline - (time.time() - start_time))
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # --- Phase 3: Second polish pass with tighter tolerance ---
    if best_result is not None and time.time() < deadline - 5:
        polished2 = _fast_polish(best_result, deadline - (time.time() - start_time), tight=True)
        if polished2 is not None:
            valid, val = check_validity(polished2)
            if valid and val > best_value:
                best_value = val
                best_result = polished2

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]

    # Tighter shrink to preserve value
    r = r * (1 - 1e-9)

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=100):
    """Force-directed relaxation in unified parameter space."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001

    for _ in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 100
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 100
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 100
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 100

        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                min_d = r[i] + r[j]
                if d_sq < min_d * min_d and d_sq > 1e-12:
                    d = np.sqrt(d_sq)
                    overlap = min_d - d
                    max_overlap = max(max_overlap, overlap)
                    f = overlap * 15
                    fx[i] += (dx / d) * f
                    fy[i] += (dy / d) * f
                    fx[j] -= (dx / d) * f
                    fy[j] -= (dy / d) * f

        if max_overlap < 1e-8:
            break

        force_mag = np.max(np.sqrt(fx * fx + fy * fy))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _fast_polish(z_init, time_budget, tight=False):
    """Fast force-directed polish with fully vectorized operations."""
    if time_budget < 2:
        return None

    z = z_init.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Vectorized force-directed relaxation
    dt = 0.003 if not tight else 0.001
    max_iter = int(min(1000 if not tight else 2000, time_budget * 100))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Boundary forces (vectorized)
        left_viol = np.maximum(r - x, 0)
        right_viol = np.maximum(x + r - w, 0)
        bottom_viol = np.maximum(r - y, 0)
        top_viol = np.maximum(y + r - h, 0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        # Pairwise overlap forces (fully vectorized accumulation)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        # Only compute forces for overlapping pairs
        overlap_mask = d_sq_pairs < r_sum_pairs * r_sum_pairs
        if np.any(overlap_mask):
            d_pairs = np.sqrt(np.maximum(d_sq_pairs[overlap_mask], 1e-12))
            overlaps = r_sum_pairs[overlap_mask] - d_pairs
            dirs_x = dx[iu][overlap_mask] / d_pairs
            dirs_y = dy[iu][overlap_mask] / d_pairs

            # Accumulate forces using np.add.at
            idx_i = iu[0][overlap_mask]
            idx_j = iu[1][overlap_mask]
            f_mag = overlaps * 25

            np.add.at(fx, idx_i, dirs_x * f_mag)
            np.add.at(fy, idx_i, dirs_y * f_mag)
            np.add.at(fx, idx_j, -dirs_x * f_mag)
            np.add.at(fy, idx_j, -dirs_y * f_mag)

        max_overlap = np.max(overlaps) if np.any(overlap_mask) else 0.0
        if max_overlap < 1e-9:
            break

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    if not _is_valid(x, y, r, w, h):
        return None

    # Adaptive radius growth
    grow_factor = 1.002 if not tight else 1.0005
    min_grow = 1.0001
    max_grow_attempts = int(min(1000 if not tight else 2000, time_budget * 100))

    for attempt in range(max_grow_attempts):
        if time.time() > time_budget:
            break
        if grow_factor < min_grow:
            break

        # Try uniform growth first
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
            grow_factor = max(min_grow, grow_factor * 0.93)
            # Quick relaxation after growth
            for _ in range(10):
                fx = np.zeros(n)
                fy = np.zeros(n)
                left_viol = np.maximum(r - x, 0)
                right_viol = np.maximum(x + r - w, 0)
                bottom_viol = np.maximum(r - y, 0)
                top_viol = np.maximum(y + r - h, 0)
                fx -= left_viol * 50
                fx += right_viol * 50
                fy -= bottom_viol * 50
                fy += top_viol * 50
                
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                d_sq = dx * dx + dy * dy
                r_sum = r[:, None] + r[None, :]
                iu = np.triu_indices(n, k=1)
                d_sq_pairs = d_sq[iu]
                r_sum_pairs = r_sum[iu]
                overlap_mask = d_sq_pairs < r_sum_pairs * r_sum_pairs
                if np.any(overlap_mask):
                    d_pairs = np.sqrt(np.maximum(d_sq_pairs[overlap_mask], 1e-12))
                    overlaps = r_sum_pairs[overlap_mask] - d_pairs
                    dirs_x = dx[iu][overlap_mask] / d_pairs
                    dirs_y = dy[iu][overlap_mask] / d_pairs
                    idx_i = iu[0][overlap_mask]
                    idx_j = iu[1][overlap_mask]
                    f_mag = overlaps * 25
                    np.add.at(fx, idx_i, dirs_x * f_mag)
                    np.add.at(fy, idx_i, dirs_y * f_mag)
                    np.add.at(fx, idx_j, -dirs_x * f_mag)
                    np.add.at(fy, idx_j, -dirs_y * f_mag)
                
                x += 0.002 * fx
                y += 0.002 * fy
                x = np.clip(x, r, w - r)
                y = np.clip(y, r, h - r)
        else:
            # Try growing individual circles (smallest first)
            sorted_idx = np.argsort(r)
            success = False
            for k in range(5, 0, -1):
                r_trial2 = r.copy()
                r_trial2[sorted_idx[:k]] *= grow_factor
                if _is_valid(x, y, r_trial2, w, h):
                    r = r_trial2
                    success = True
                    # Relax
                    for _ in range(5):
                        fx = np.zeros(n)
                        fy = np.zeros(n)
                        left_viol = np.maximum(r - x, 0)
                        right_viol = np.maximum(x + r - w, 0)
                        bottom_viol = np.maximum(r - y, 0)
                        top_viol = np.maximum(y + r - h, 0)
                        fx -= left_viol * 50
                        fx += right_viol * 50
                        fy -= bottom_viol * 50
                        fy += top_viol * 50
                        
                        dx = x[:, None] - x[None, :]
                        dy = y[:, None] - y[None, :]
                        d_sq = dx * dx + dy * dy
                        r_sum = r[:, None] + r[None, :]
                        iu = np.triu_indices(n, k=1)
                        d_sq_pairs = d_sq[iu]
                        r_sum_pairs = r_sum[iu]
                        overlap_mask = d_sq_pairs < r_sum_pairs * r_sum_pairs
                        if np.any(overlap_mask):
                            d_pairs = np.sqrt(np.maximum(d_sq_pairs[overlap_mask], 1e-12))
                            overlaps = r_sum_pairs[overlap_mask] - d_pairs
                            dirs_x = dx[iu][overlap_mask] / d_pairs
                            dirs_y = dy[iu][overlap_mask] / d_pairs
                            idx_i = iu[0][overlap_mask]
                            idx_j = iu[1][overlap_mask]
                            f_mag = overlaps * 25
                            np.add.at(fx, idx_i, dirs_x * f_mag)
                            np.add.at(fy, idx_i, dirs_y * f_mag)
                            np.add.at(fx, idx_j, -dirs_x * f_mag)
                            np.add.at(fy, idx_j, -dirs_y * f_mag)
                        
                        x += 0.002 * fx
                        y += 0.002 * fy
                        x = np.clip(x, r, w - r)
                        y = np.clip(y, r, h - r)
                    break
            if not success:
                grow_factor *= 0.85

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True
