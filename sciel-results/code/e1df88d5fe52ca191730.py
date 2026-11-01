"""Circle packing in perimeter-4 rectangle: DE + LogSumExp Epigraph + Contact Graph."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay

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

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _logsumexp_objective(z, beta=50.0):
    """
    Log-sum-exp epigraph formulation: smooth approximation of the non-convex
    distance constraints.  The objective is -sum(r) + (1/beta)*log(sum(exp(beta*violation)))
    for all constraints.  This is much better conditioned than hard inequality
    constraints for SLSQP.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2:
        return 1e6

    # Collect all constraint violations (should be <= 0)
    violations = []

    # Boundary constraints
    violations.append(np.maximum(r - x, 0.0))
    violations.append(np.maximum(x + r - w, 0.0))
    violations.append(np.maximum(r - y, 0.0))
    violations.append(np.maximum(y + r - h, 0.0))

    # Pairwise overlap constraints
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    violations.append(overlap)

    # Stack all violations
    all_v = np.concatenate(violations)
    max_v = np.max(all_v)
    if max_v < 1e-12:
        penalty = 0.0
    else:
        # Log-sum-exp for numerical stability
        shifted = all_v - max_v
        penalty = (max_v + np.log(np.sum(np.exp(beta * shifted)))) / beta

    return -np.sum(r) + penalty


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

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
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
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
        for _ in range(10):
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

    # === Phase 1: Differential Evolution (global search) ===
    if time.time() < deadline - 80:
        bounds_de = [(0.01, 1.99)] * (3 * N) + [(0.3, 1.7)]
        # DE with penalty objective for global exploration
        try:
            de_res = differential_evolution(
                _vectorized_objective, bounds_de,
                seed=42, maxiter=200, tol=1e-8,
                popsize=15, mutation=(0.3, 1.0), recombination=0.9,
                polish=False, updating='deferred',
                init='latinhypercube',
                constraints=[{'type': 'ineq', 'fun': lambda z: 0.3 - z[-1]},
                             {'type': 'ineq', 'fun': lambda z: z[-1] - 1.7}],
                options={'maxiter': 200}
            )
            if de_res.x is not None:
                z_de = de_res.x
                valid, val = check_validity(z_de)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_de.copy()
        except Exception:
            pass

    # === Phase 2: Multi-start SLSQP with log-sum-exp epigraph ===
    guesses = get_initial_guesses()
    # Add DE result if available
    if best_result is not None:
        guesses.insert(0, best_result.copy())

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 40:
            break

        # Force-directed relaxation first
        z0_relaxed = _force_relax_unified(z0, max_iter=300)

        # SLSQP with log-sum-exp objective (smooth, no hard constraints needed)
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(100, remaining * 40)))
        try:
            res = minimize(
                _logsumexp_objective, z0_relaxed, method='SLSQP',
                bounds=[(0.005, 1.99)] * (3 * N) + [(0.3, 1.7)],
                options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 60}
            )
            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Try repair
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.98
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # === Phase 3: Contact-graph local optimization ===
    if best_result is not None and time.time() < deadline - 30:
        # Contact graph optimization: identify touching circles and optimize
        # positions/radii on the contact graph
        polished = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 20)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # === Phase 4: Greedy per-circle binary search expansion ===
    if best_result is not None and time.time() < deadline - 15:
        expanded = _greedy_expansion(best_result, deadline - (time.time() - start_time) - 10)
        if expanded is not None:
            valid, val = check_validity(expanded)
            if valid and val > best_value:
                best_value = val
                best_result = expanded

    # === Phase 5: Final SLSQP with explicit constraints (high precision) ===
    if best_result is not None and time.time() < deadline - 10:
        refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 5)
        if refined is not None:
            valid, val = check_validity(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined

    # === Phase 6: Final polish ===
    if best_result is not None and time.time() < deadline - 3:
        polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
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
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step."""
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


def _contact_graph_optimize(z, time_budget):
    """
    Contact-graph based local optimization: build the graph of touching/near-touching
    circles, then iteratively adjust positions and radii to maximize sum(r) while
    maintaining the contact structure.  This is a different optimization family
    from SLSQP - it uses a combinatorial + local search approach.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    def build_contact_graph(tol=0.01):
        """Build adjacency: circles i,j are connected if distance <= r_i+r_j+tol."""
        contacts = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d <= r[i] + r[j] + tol:
                    contacts[i].append(j)
                    contacts[j].append(i)
        return contacts

    def try_expand(i, contacts):
        """Try to expand circle i while maintaining validity."""
        # Max radius from walls
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        # Max radius from neighbors
        for j in contacts.get(i, []):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        # Also check all circles (not just contacts) for safety
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        return max(0.0, r_max)

    # Iterative expansion: repeatedly find the most expandable circle and expand it
    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        contacts = build_contact_graph()
        improved = False

        # Order by smallest radius (most room to grow)
        order = np.argsort(r)
        for i in order:
            if time.time() > time_budget:
                break

            r_max = try_expand(i, contacts)
            if r_max > r[i] + 1e-7:
                # Binary search for exact max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

        if not improved:
            break

    # Position optimization on contact graph
    for rnd in range(20):
        if time.time() > time_budget:
            break

        contacts = build_contact_graph()
        improved = False

        for i in range(n):
            if time.time() > time_budget:
                break

            # Try small perturbations in 8 directions
            best_pos = None
            best_r = r[i]

            for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                for step in [0.05, 0.03, 0.015, 0.008]:
                    new_x = x[i] + step * np.cos(angle)
                    new_y = y[i] + step * np.sin(angle)

                    if new_x < r[i] or new_x > w - r[i]:
                        continue
                    if new_y < r[i] or new_y > h - r[i]:
                        continue

                    # Check validity
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        d2 = dx * dx + dy * dy
                        r_sum = r[i] + r[j]
                        if d2 < r_sum * r_sum - 1e-9:
                            valid = False
                            break

                    if not valid:
                        continue

                    # Compute max radius from this position
                    r_wall_new = min(new_x, w - new_x, new_y, h - new_y)
                    r_max_new = r_wall_new
                    for j in range(n):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        r_allowed = d - r[j]
                        if r_allowed < r_max_new:
                            r_max_new = r_allowed
                    r_max_new = max(0.0, r_max_new)

                    if r_max_new > best_r + 1e-5:
                        best_r = r_max_new
                        best_pos = (new_x, new_y)

            if best_pos is not None and best_r > r[i] + 1e-5:
                x[i] = best_pos[0]
                y[i] = best_pos[1]
                r[i] = best_r
                improved = True

        if not improved:
            break

    # Final uniform expansion
    lo = 1.0
    hi = 1.03
    for _ in range(25):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2.0
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w, h):
            lo = mid
        else:
            hi = mid
    r = r * lo

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _greedy_expansion(z, time_budget):
    """
    Greedy per-circle binary search expansion: for each circle in order of
    smallest radius, find the maximum valid radius via binary search.
    This is a different search strategy from SLSQP - it's a combinatorial
    greedy with exact feasibility checks.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Multiple passes
    for pass_num in range(10):
        if time.time() > time_budget:
            break

        improved = False
        order = np.argsort(r)
        for i in order:
            if time.time() > time_budget:
                break

            # Upper bound
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

            if r_max > r[i] + 1e-8:
                lo = r[i]
                hi = r_max
                for _ in range(30):
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

        if not improved:
            break

    # Uniform scaling
    lo = 1.0
    hi = 1.02
    for _ in range(20):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2.0
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w, h):
            lo = mid
        else:
            hi = mid
    r = r * lo

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed position adjustment and radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(2000, time_budget * 100))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 35
        fx += right_viol * 35
        fy -= bottom_viol * 35
        fy += top_viol * 35

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 18.0
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
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly
    grow_factor = 1.0003
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
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


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit constraints."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    constraints = []
    bounds = []

    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    def make_boundary_constraint(idx, axis, side):
        if axis == 0:
            if side == 0:
                return lambda p: p[3 * idx] - p[3 * idx + 2]
            else:
                return lambda p: p[-1] - p[3 * idx] - p[3 * idx + 2]
        else:
            if side == 0:
                return lambda p: p[3 * idx + 1] - p[3 * idx + 2]
            else:
                return lambda p: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 1)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 1)})

    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    maxiter = int(min(600, time_budget * 120))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
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
