"""Circle packing in unit square: smooth penalty optimization + force-directed refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s kill limit

    best_sum = 0.0
    best_result = None

    def _time_left():
        return deadline - time.time()

    def check_validity(x, y, r, tol=1e-9):
        """Check if packing is valid (all constraints satisfied)."""
        if np.any(x < r - tol) or np.any(x > 1 - r + tol) or np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def _logsumexp_penalty(terms, beta=50.0):
        """Smooth approximation of sum(max(0, terms)) using log-sum-exp."""
        # For each term t, max(0,t) ≈ (1/beta)*log(1+exp(beta*t)) - (1/beta)*log(2) for t near 0
        # But simpler: use softplus-like: (1/beta)*log(1 + exp(beta*max(0,t)))
        # Actually, let's use: for violation v = max(0, t), smooth version = (1/beta)*log(1+exp(beta*t)) when t>0
        # Better: use the fact that max(0,t) = t*sigmoid(beta*t) approximately, but that's not differentiable well
        # Let's use: smooth_max(0, t) = (1/beta) * log(1 + exp(beta * t)) - (1/beta)*log(2) for t >= 0
        # But for t < 0, we want 0. So: smooth = (1/beta)*log(1 + exp(beta*max(0,t))) - (1/beta)*log(2)
        # This is still not great. Let me use a different approach.
        # Actually, for the objective, I'll use quadratic penalties with a smooth component.
        pass

    def objective(z, penalty_weight=500.0):
        """Objective: minimize -sum(r) + penalty for violations."""
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:]

        # Clip radii to positive values
        r_safe = np.maximum(r, 0.001)

        sum_r = np.sum(r_safe)

        # Boundary violations (quadratic)
        v_bound = 0.0
        v_bound += np.sum(np.maximum(0, r_safe - x)**2)
        v_bound += np.sum(np.maximum(0, r_safe - (1 - x))**2)
        v_bound += np.sum(np.maximum(0, r_safe - y)**2)
        v_bound += np.sum(np.maximum(0, r_safe - (1 - y))**2)

        # Collision violations - vectorized for speed
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist_sq = x_diff**2 + y_diff**2
        sum_r_mat = r_safe[:, None] + r_safe[None, :]
        overlap_terms = np.maximum(0, sum_r_mat**2 - dist_sq)
        np.fill_diagonal(overlap_terms, 0)
        v_coll = np.sum(overlap_terms**2)

        return -sum_r + penalty_weight * (v_bound + v_coll)

    def objective_smooth(z, penalty_weight=500.0, beta=20.0):
        """Objective with smooth (log-sum-exp based) penalties for better gradients."""
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:]

        r_safe = np.maximum(r, 0.001)
        sum_r = np.sum(r_safe)

        # Smooth boundary violations
        # For each constraint c_i >= 0, penalty = (1/beta)*log(1 + exp(-beta*c_i))
        # where c_i = x_i - r_i, etc.
        c_left = x - r_safe
        c_right = 1 - x - r_safe
        c_bottom = y - r_safe
        c_top = 1 - y - r_safe

        def smooth_penalty(c, beta=20.0):
            # Penalty for c < 0: (1/beta)*log(1 + exp(-beta*c))
            # For numerical stability
            c_clipped = np.clip(c, -50, 50)
            return (1.0 / beta) * np.log1p(np.exp(-beta * c_clipped))

        v_bound = np.sum(smooth_penalty(c_left, beta)) + np.sum(smooth_penalty(c_right, beta) \
                   + smooth_penalty(c_bottom, beta) + smooth_penalty(c_top, beta))

        # Smooth collision violations
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist_sq = x_diff**2 + y_diff**2
        sum_r_mat = r_safe[:, None] + r_safe[None, :]
        # Violation: (r_i + r_j)^2 - dist^2 > 0
        viol = sum_r_mat**2 - dist_sq
        np.fill_diagonal(viol, -1e10)  # No self-collision

        # Smooth penalty for viol > 0
        v_coll = np.sum(smooth_penalty(-viol, beta))

        return -sum_r + penalty_weight * (v_bound + v_coll)

    def local_optimize(x0, y0, r0, max_iter=300, use_smooth=True):
        """Run local optimization from initial guess."""
        z0 = np.concatenate([x0, y0, r0])

        try:
            if use_smooth:
                res = minimize(objective_smooth, z0, method='L-BFGS-B',
                              options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            else:
                res = minimize(objective, z0, method='L-BFGS-B',
                              options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})

            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]

            r_opt = np.maximum(r_opt, 0.001)

            # Check validity and adjust if needed
            if check_validity(x_opt, y_opt, r_opt):
                return x_opt, y_opt, r_opt
            else:
                # Try to fix by shrinking radii
                for scale in [0.99, 0.98, 0.97, 0.95, 0.90]:
                    r_adjusted = r_opt * scale
                    if check_validity(x_opt, y_opt, r_adjusted):
                        return x_opt, y_opt, r_adjusted
                return None
        except Exception:
            return None

    def force_anneal(x, y, r, max_steps=3000, initial_temp=0.05, final_temp=0.001):
        """Force-directed simulated annealing to improve packing density."""
        x = x.copy()
        y = y.copy()
        r = r.copy()

        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_sum = np.sum(r)

        r_mean = np.mean(r)

        for step in range(max_steps):
            if _time_left() < 5.0:
                break

            # Cooling schedule (geometric)
            temp = initial_temp * (final_temp / initial_temp) ** (step / max_steps)

            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Repulsive forces between overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]

                    if dist < min_dist and dist > 1e-12:
                        overlap = min_dist - dist
                        force = 15.0 * overlap / r_mean
                        fx[i] -= force * dx / dist
                        fy[i] -= force * dy / dist
                        fx[j] += force * dx / dist
                        fy[j] += force * dy / dist

            # Boundary forces
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += 10.0 * (r[i] - x[i]) / r_mean
                if x[i] > 1 - r[i]:
                    fx[i] -= 10.0 * (x[i] - (1 - r[i])) / r_mean
                if y[i] < r[i]:
                    fy[i] += 10.0 * (r[i] - y[i]) / r_mean
                if y[i] > 1 - r[i]:
                    fy[i] -= 10.0 * (y[i] - (1 - r[i])) / r_mean

            # Update positions
            x += temp * fx
            y += temp * fy

            # Clamp to valid range
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)

            # Try to grow radii if no overlaps
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    if d < r[i] + r[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                growth = 0.001 * temp
                new_r = r + growth
                valid_growth = True
                for i in range(N):
                    if new_r[i] > min(x[i], 1-x[i], y[i], 1-y[i]):
                        valid_growth = False
                        break
                    for j in range(i + 1, N):
                        d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                        if d < new_r[i] + new_r[j] - 1e-9:
                            valid_growth = False
                            break
                    if not valid_growth:
                        break

                if valid_growth:
                    r = new_r

            current_sum = np.sum(r)
            if current_sum > best_sum:
                best_sum = current_sum
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

        return best_x, best_y, best_r

    def hexagonal_init(n_circles):
        """Create hexagonal packing initialization."""
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)

        cols = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / cols))

        col_spacing = 1.0
        row_spacing = 0.866

        total_width = (cols - 1) * col_spacing
        total_height = (rows - 1) * row_spacing

        margin_x = 0.05
        margin_y = 0.05

        scale_x = (1 - 2*margin_x) / max(total_width, 1e-6)
        scale_y = (1 - 2*margin_y) / max(total_height, 1e-6)
        scale = min(scale_x, scale_y, 1.0)

        for i in range(n_circles):
            row = i // cols
            col = i % cols
            x[i] = margin_x + col * col_spacing * scale + (row % 2) * 0.5 * col_spacing * scale
            y[i] = margin_y + row * row_spacing * scale
            r[i] = 0.5 * min(col_spacing, row_spacing) * scale * 0.95

        for i in range(n_circles):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)

        return x, y, r

    def random_init():
        """Random initialization."""
        r = np.random.uniform(0.03, 0.1, N)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        return x, y, r

    def grid_init():
        """Grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)

        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.95

        return x, y, r

    def center_ring_init():
        """Center-ring initialization."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)

        x[0] = 0.5
        y[0] = 0.5
        r[0] = 0.12

        n_ring = N - 1
        ring_radius = 0.35
        for i in range(1, N):
            angle = 2 * np.pi * (i - 1) / n_ring
            x[i] = 0.5 + ring_radius * np.cos(angle)
            y[i] = 0.5 + ring_radius * np.sin(angle)
            r[i] = 0.08

        for i in range(N):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)

        return x, y, r

    def two_ring_init():
        """Two-ring initialization."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)

        x[0] = 0.5
        y[0] = 0.5
        r[0] = 0.10

        n_inner = 8
        inner_radius = 0.25
        for i in range(1, 1 + n_inner):
            angle = 2 * np.pi * (i - 1) / n_inner
            x[i] = 0.5 + inner_radius * np.cos(angle)
            y[i] = 0.5 + inner_radius * np.sin(angle)
            r[i] = 0.07

        n_outer = N - 1 - n_inner
        outer_radius = 0.42
        for i in range(1 + n_inner, N):
            idx = i - 1 - n_inner
            angle = 2 * np.pi * idx / n_outer + np.pi / n_outer
            x[i] = 0.5 + outer_radius * np.cos(angle)
            y[i] = 0.5 + outer_radius * np.sin(angle)
            r[i] = 0.06

        for i in range(N):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)

        return x, y, r

    def try_optimize(x0, y0, r0, use_smooth=True):
        """Try to optimize from given initial state."""
        nonlocal best_sum, best_result
        result = local_optimize(x0, y0, r0, max_iter=400, use_smooth=use_smooth)
        if result is not None:
            x_opt, y_opt, r_opt = result
            current_sum = np.sum(r_opt)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_opt, y_opt, r_opt))

    def polish(x, y, r, max_iter=500):
        """High-precision polish."""
        z0 = np.concatenate([x, y, r])

        try:
            # First pass with smooth objective
            res = minimize(objective_smooth, z0, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]

            # Second pass with quadratic objective for precision
            z1 = np.concatenate([x_opt, y_opt, r_opt])
            res2 = minimize(objective, z1, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
            x_opt = res2.x[:N]
            y_opt = res2.x[N:2*N]
            r_opt = res2.x[2*N:]

            r_opt = np.maximum(r_opt, 0.001)

            if check_validity(x_opt, y_opt, r_opt):
                return x_opt, y_opt, r_opt
            else:
                for scale in [0.999, 0.998, 0.997, 0.995]:
                    r_adjusted = r_opt * scale
                    if check_validity(x_opt, y_opt, r_adjusted):
                        return x_opt, y_opt, r_adjusted
                return x, y, r
        except Exception:
            return x, y, r

    # Multi-start optimization with different initializations
    # Hexagonal starts
    for k in range(25):
        if _time_left() < 10.0:
            break
        x0, y0, r0 = hexagonal_init(N)
        x0 += np.random.uniform(-0.02, 0.02, N)
        y0 += np.random.uniform(-0.02, 0.02, N)
        r0 *= np.random.uniform(0.85, 1.15, N)
        try_optimize(x0, y0, r0, use_smooth=(k % 2 == 0))

    # Random starts
    for k in range(40):
        if _time_left() < 10.0:
            break
        x0, y0, r0 = random_init()
        try_optimize(x0, y0, r0, use_smooth=(k % 2 == 0))

    # Grid starts
    for k in range(8):
        if _time_left() < 10.0:
            break
        x0, y0, r0 = grid_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)

    # Center-ring starts
    for k in range(12):
        if _time_left() < 10.0:
            break
        x0, y0, r0 = center_ring_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.85, 1.15, N)
        try_optimize(x0, y0, r0, use_smooth=(k % 2 == 0))

    # Two-ring starts
    for k in range(12):
        if _time_left() < 10.0:
            break
        x0, y0, r0 = two_ring_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.85, 1.15, N)
        try_optimize(x0, y0, r0, use_smooth=(k % 2 == 0))

    # Force-directed annealing from best result
    if best_result is not None and _time_left() > 15.0:
        x_best = best_result[:, 0]
        y_best = best_result[:, 1]
        r_best = best_result[:, 2]

        for temp_init in [0.03, 0.05, 0.1, 0.15]:
            if _time_left() < 10.0:
                break
            x_anneal, y_anneal, r_anneal = force_anneal(x_best, y_best, r_best,
                                                        max_steps=2500,
                                                        initial_temp=temp_init)
            current_sum = np.sum(r_anneal)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_anneal, y_anneal, r_anneal))

    # Final high-precision polish
    if best_result is not None and _time_left() > 8.0:
        x_best = best_result[:, 0]
        y_best = best_result[:, 1]
        r_best = best_result[:, 2]
        x_polish, y_polish, r_polish = polish(x_best, y_best, r_best, max_iter=400)
        current_sum = np.sum(r_polish)
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = np.column_stack((x_polish, y_polish, r_polish))

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
