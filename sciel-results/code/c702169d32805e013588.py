"""Circle packing in perimeter-4 rectangle: SLSQP + Contact-Graph Optimization + DE + LSE."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 21


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
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

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _lse_refine(z, time_budget):
    """
    Smooth LSE (log-sum-exp) epigraph refinement with adaptive temperature.
    The LSE beta increases as violations shrink, tightening the surrogate.
    """
    z = z.copy().astype(float)
    n = N
    iu = np.triu_indices(n, k=1)
    
    # Adaptive beta: starts moderate, increases as we converge
    beta_start = 20.0
    beta_end = 200.0
    
    def lse_objective(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w
        
        if w < 0.2 or h < 0.2:
            return 1e6
        
        obj = -np.sum(r)
        
        # LSE smooth penalty for circle-circle
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        r_sum = r[iu[0]] + r[iu[1]]
        violation = r_sum - d  # positive means overlap
        
        # Adaptive beta based on max violation
        max_viol = float(np.max(violation)) if len(violation) > 0 else 0.0
        if max_viol > 0.01:
            beta = beta_start
        elif max_viol > 0.001:
            beta = (beta_start + beta_end) / 2
        else:
            beta = beta_end
        
        v_clipped = np.maximum(violation, -10.0)
        max_v = np.max(v_clipped)
        if max_v > 0:
            lse_val = max_v + np.log(np.sum(np.exp(beta * (v_clipped - max_v)))) / beta
        else:
            lse_val = 0.0
        obj += 100.0 * lse_val
        
        # LSE for walls
        wall_viol = np.concatenate([
            r - x,
            x + r - w,
            r - y,
            y + r - h
        ])
        max_wv = float(np.max(wall_viol)) if len(wall_viol) > 0 else 0.0
        if max_wv > 0.01:
            beta_w = beta_start
        elif max_wv > 0.001:
            beta_w = (beta_start + beta_end) / 2
        else:
            beta_w = beta_end
        
        wv_clipped = np.maximum(wall_viol, -10.0)
        max_wv_c = np.max(wv_clipped)
        if max_wv_c > 0:
            lse_wall = max_wv_c + np.log(np.sum(np.exp(beta_w * (wv_clipped - max_wv_c)))) / beta_w
        else:
            lse_wall = 0.0
        obj += 100.0 * lse_wall
        
        # Aspect ratio penalty
        if w < 0.5:
            obj += 1000.0 * (0.5 - w) ** 2
        elif w > 1.5:
            obj += 1000.0 * (w - 1.5) ** 2
        
        return obj
    
    bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]
    maxiter = int(min(1500, time_budget * 150))
    
    try:
        res = minimize(lse_objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w
        
        # Repair if needed
        if not _is_valid(x, y, r, w, h):
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99, 0.98]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break
        
        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _contact_graph_optimize(z0, time_budget):
    """
    Contact-graph-aware local optimization.
    Identifies active constraints (near-contact circles/walls) and solves
    a smaller, better-conditioned problem on just those variables.
    """
    n = N
    x = z0[0::3][:n].copy().astype(float)
    y = z0[1::3][:n].copy().astype(float)
    r = z0[2::3][:n].copy().astype(float)
    w = float(z0[-1])
    h = 2.0 - w

    if not _is_valid(x, y, r, w, h):
        return None

    best_val = float(np.sum(r))
    best_state = (x.copy(), y.copy(), r.copy(), w)
    contact_tol = 0.02

    for iteration in range(30):
        if time.time() > time_budget:
            break

        # Identify contact graph: pairs that are near-touching
        iu = np.triu_indices(n, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d = np.sqrt(dx * dx + dy * dy)
        r_sum = r[iu[0]] + r[iu[1]]
        gaps = d - r_sum

        # Active circle-circle contacts
        active_cc = np.where(gaps < contact_tol)[0]

        # Active wall contacts
        active_left = np.where(x - r < contact_tol)[0]
        active_right = np.where(w - x - r < contact_tol)[0]
        active_bottom = np.where(y - r < contact_tol)[0]
        active_top = np.where(h - y - r < contact_tol)[0]

        # If no active constraints, we're at a local optimum for this config
        if len(active_cc) == 0 and len(active_left) == 0 and len(active_right) == 0 \
           and len(active_bottom) == 0 and len(active_top) == 0:
            break

        # Try per-circle greedy expansion
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Max radius for circle i given current positions
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                d_ij = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                r_allowed = d_ij - r[j]
                r_max = min(r_max, r_allowed)
            r_max = max(0.001, r_max)

            if r_max > r[i] + 1e-8:
                # Binary search for max valid radius
                lo, hi = r[i], r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True

        if improved:
            val = float(np.sum(r))
            if val > best_val:
                best_val = val
                best_state = (x.copy(), y.copy(), r.copy(), w)

        # Try moving circles that are constrained
        # Focus on circles with active constraints
        constrained = set()
        for idx in active_cc:
            constrained.add(iu[0][idx])
            constrained.add(iu[1][idx])
        constrained.update(active_left.tolist())
        constrained.update(active_right.tolist())
        constrained.update(active_bottom.tolist())
        constrained.update(active_top.tolist())

        # Small random perturbations for constrained circles
        rng = np.random.default_rng(hash(tuple(r.round(4))) % (2**32))
        for i in sorted(constrained):
            if time.time() > time_budget:
                break
            # Try small moves in random directions
            for _ in range(5):
                angle = rng.uniform(0, 2 * np.pi)
                step = rng.uniform(0.005, 0.03)
                nx = x[i] + step * np.cos(angle)
                ny = y[i] + step * np.sin(angle)
                if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                    continue
                x_test = x.copy()
                y_test = y.copy()
                x_test[i] = nx
                y_test[i] = ny
                if _is_valid(x_test, y_test, r, w, h):
                    x[i] = nx
                    y[i] = ny
                    # Try to expand after move
                    r_wall = min(x[i], w - x[i], y[i], h - y[i])
                    r_max = r_wall
                    for j in range(n):
                        if i == j:
                            continue
                        d_ij = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                        r_max = min(r_max, d_ij - r[j])
                    r_max = max(0.001, r_max)
                    if r_max > r[i] + 1e-8:
                        lo, hi = r[i], r_max
                        for _ in range(15):
                            mid = (lo + hi) / 2.0
                            r_test = r.copy()
                            r_test[i] = mid
                            if _is_valid(x, y, r_test, w, h):
                                lo = mid
                            else:
                                hi = mid
                        if lo > r[i] + 1e-9:
                            r[i] = lo

        val = float(np.sum(r))
        if val > best_val:
            best_val = val
            best_state = (x.copy(), y.copy(), r.copy(), w)

    if _is_valid(best_state[0], best_state[1], best_state[2], best_state[3], 2.0 - best_state[3]):
        z_out = np.empty(3 * N + 1)
        z_out[0::3][:n] = best_state[0]
        z_out[1::3][:n] = best_state[1]
        z_out[2::3][:n] = best_state[2]
        z_out[-1] = best_state[3]
        return z_out
    return None


def _de_global_search(time_budget):
    """
    Differential Evolution global search to find diverse high-quality starting points.
    Uses a compact representation: only the aspect ratio w and initial grid perturbations.
    """
    if time.time() > time_budget:
        return None

    n = N
    side = int(np.ceil(np.sqrt(n)))

    def de_objective(params):
        w = params[0]
        h = 2.0 - w
        if w < 0.3 or h < 0.3:
            return 1e6

        # Grid-based initial placement with perturbations
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)

        for i in range(n):
            row, col = divmod(i, side)
            base_cx = (col + 0.5) / side * w
            base_cy = (row + 0.5) / side * h
            base_r = min(w / side, h / side) * 0.45

            # Add perturbation from params
            px = params[1 + 2 * i] if 1 + 2 * i < len(params) else 0
            py = params[2 + 2 * i] if 2 + 2 * i < len(params) else 0

            x[i] = base_cx + px * w * 0.1
            y[i] = base_cy + py * h * 0.1
            r[i] = base_r * (1.0 + 0.1 * (params[3 + 2 * i] if 3 + 2 * i < len(params) else 0))

        # Quick validity check
        if not _is_valid(x, y, r, w, h):
            # Shrink radii
            for factor in [0.95, 0.9, 0.85, 0.8]:
                if _is_valid(x, y, r * factor, w, h):
                    r = r * factor
                    break
            else:
                return 1e6

        return -np.sum(r)

    # DE bounds: [w, perturbations...]
    bounds = [(0.4, 1.6)]
    for _ in range(n):
        bounds.extend([(-1, 1), (-1, 1)])

    maxiter = int(min(50, time_budget * 5))
    try:
        result = differential_evolution(
            de_objective, bounds,
            seed=42, maxiter=maxiter, tol=1e-8,
            popsize=8, mutation=(0.3, 0.8), recombination=0.7,
            polish=False, init='latinhypercube'
        )
        if result.success or result.fun < 1e5:
            params = result.x
            w = params[0]
            h = 2.0 - w
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row, col = divmod(i, side)
                base_cx = (col + 0.5) / side * w
                base_cy = (row + 0.5) / side * h
                base_r = min(w / side, h / side) * 0.45
                px = params[1 + 2 * i]
                py = params[2 + 2 * i]
                x[i] = base_cx + px * w * 0.1
                y[i] = base_cy + py * h * 0.1
                r[i] = base_r * (1.0 + 0.1 * (params[3 + 2 * i] if 3 + 2 * i < len(params) else 0))

            if _is_valid(x, y, r, w, h):
                z = np.empty(3 * N + 1)
                z[0::3][:n] = x
                z[1::3][:n] = y
                z[2::3][:n] = r
                z[-1] = w
                return z
    except Exception:
        pass
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s kill

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

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

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8]
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

        rng = np.random.default_rng(42)
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    # --- Phase 1: SLSQP from multiple starts ---
    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        z0_relaxed = _force_relax_unified(z0, max_iter=300)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 40)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 1.5: DE global search for diverse starts ---
    if time.time() < deadline - 35:
        de_result = _de_global_search(deadline - (time.time() - start_time) - 30)
        if de_result is not None:
            valid, val = check_validity(de_result)
            if valid and val > best_value:
                best_value = val
                best_result = de_result.copy()
            # Also run SLSQP from DE result
            if time.time() < deadline - 20:
                try:
                    res = minimize(objective, de_result, method='SLSQP',
                                   bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                                   options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 60})
                    z_opt = res.x
                    valid, val = check_validity(z_opt)
                    if valid and val > best_value:
                        best_value = val
                        best_result = z_opt.copy()
                except Exception:
                    pass

    # --- Phase 2: Multi-method polish pipeline ---
    if best_result is not None:
        # 1. Contact-graph optimization (new structural method)
        if time.time() < deadline - 20:
            cg_result = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 15)
            if cg_result is not None:
                valid, val = check_validity(cg_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg_result

        # 2. Force-directed position polish
        if best_result is not None and time.time() < deadline - 18:
            polished = _polish_result(best_result, deadline - (time.time() - start_time) - 13)
            if polished is not None:
                valid, val = check_validity(polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished

        # 3. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 15:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 10:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. LSE epigraph refinement
        if best_result is not None and time.time() < deadline - 6:
            lse = _lse_refine(best_result, deadline - (time.time() - start_time) - 3)
            if lse is not None:
                valid, val = check_validity(lse)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse

        # 6. Second contact-graph pass
        if best_result is not None and time.time() < deadline - 3:
            cg_result2 = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 1)
            if cg_result2 is not None:
                valid, val = check_validity(cg_result2)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg_result2

        # 7. Final expansion pass
        if best_result is not None and time.time() < deadline - 1:
            expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time))
            if expanded2 is not None:
                valid, val = check_validity(expanded2)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded2

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety shrink
    for _ in range(30):
        r_trial = r * (1 - 1e-7)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=500):
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        if max_overlap < convergence_tol:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
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


def _polish_result(z, time_budget):
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.002
    max_iter = int(min(5000, time_budget * 200))
    prev_best = float(np.sum(r))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-12:
            f_pairs = overlap * 20.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        elif max_force > 0.01:
            dt = 0.001
        else:
            dt = 0.002

        x_new = x + dt * fx
        y_new = y + dt * fy
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)

        if _is_valid(x_new, y_new, r, w, h):
            x = x_new
            y = y_new

    for _ in range(100):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
            r = r * lo
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _expansion_relaxation_polish(z, time_budget):
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]

    def try_expand_circle(i):
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        r_max = max(0.0, r_max)
        
        if r_max > r[i] + 1e-6:
            lo = r[i]
            hi = r_max
            for _ in range(25):
                mid = (lo + hi) / 2.0
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-7:
                r[i] = lo
                return True
        return False

    def try_move_circle(i):
        best_dir = None
        best_score = -float('inf')
        
        for dx_d, dy_d in dirs:
            new_x = x[i] + dx_d * 0.01
            new_y = y[i] + dy_d * 0.01
            if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                continue
            
            min_gap = float('inf')
            for j in range(n):
                if i == j:
                    continue
                dx = new_x - x[j]
                dy = new_y - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
            
            wall_dist = min(new_x - r[i], w - new_x - r[i], new_y - r[i], h - new_y - r[i])
            score = min(min_gap, wall_dist)
            
            if score > best_score:
                best_score = score
                best_dir = (dx_d, dy_d)
        
        if best_dir is None:
            return False
        
        dx_d, dy_d = best_dir
        
        for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]:
            new_x = x[i] + dx_d * step
            new_y = y[i] + dy_d * step
            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                x[i] = new_x
                y[i] = new_y
                return True
        
        return False

    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            
            if try_expand_circle(i):
                improved = True
            
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            
            if r_max - r[i] < 0.01:
                if try_move_circle(i):
                    improved = True
        
        if not improved:
            break

    for rnd in range(20):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            if try_expand_circle(i):
                improved = True
        
        if not improved:
            break

    for _ in range(100):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
            r = r * lo
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-9:
            return False
    return True


def _constraint_refine(z, time_budget):
    z = z.copy().astype(float)
    n = N
    n_pairs = n * (n - 1) // 2

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    def make_circle_constraint(i):
        def constraint(zv):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            dx = x[i] - x
            dy = y[i] - y
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            result = dist - (r[i] + r)
            result[i] = 1.0
            return result
        return constraint

    def make_wall_constraint(axis, side):
        def constraint(zv):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            w = zv[-1]
            h = 2.0 - w
            
            if axis == 'x':
                pos = x
            else:
                pos = y
            
            if side == 'left' or side == 'bottom':
                return pos - r
            else:
                if axis == 'x':
                    return w - pos - r
                else:
                    return h - pos - r
        return constraint

    constraints = []
    
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_circle_constraint(i)})
    
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('x', 'left')})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('x', 'right')})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('y', 'bottom')})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('y', 'top')})

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1500, time_budget * 300))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 200})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if not _is_valid(x, y, r, w, h):
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break
        
        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
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
