import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Precompute pair indices for vectorized operations
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)

    def project_valid(x, y, r, max_iter=200):
        """
        Hard-constraint projection: iteratively push overlapping circles apart
        and clamp to boundaries until no overlaps remain.
        Returns (x, y, r) that are valid (or as close as possible).
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()

        # First: clamp radii to boundary limits
        max_r_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, max_r_bound)
        r = np.maximum(r, 1e-8)

        for iteration in range(max_iter):
            # Compute all pairwise distances
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx * dx + dy * dy)
            sum_r = r[i_idx] + r[j_idx]

            # Find violations: pairs where dist < sum_r
            violations = sum_r - dist
            # Only consider actual overlaps (with small tolerance)
            overlap_mask = violations > 1e-10

            if not np.any(overlap_mask):
                break

            # For each overlapping pair, push them apart
            # We accumulate displacement forces
            fx = np.zeros(N)
            fy = np.zeros(N)

            vi = i_idx[overlap_mask]
            vj = j_idx[overlap_mask]
            dx_v = x[vi] - x[vj]
            dy_v = y[vi] - y[vj]
            dist_v = np.sqrt(dx_v * dx_v + dy_v * dy_v)
            viol_v = sum_r[overlap_mask] - dist_v

            # Direction to push apart (from j to i)
            # If dist is ~0, use random direction
            safe_dist = np.where(dist_v > 1e-10, dist_v, 1.0)
            dir_x = dx_v / safe_dist
            dir_y = dy_v / safe_dist
            # For near-zero distance, assign random directions
            zero_mask = dist_v < 1e-10
            if np.any(zero_mask):
                dir_x[zero_mask] = np.random.randn(np.sum(zero_mask))
                dir_y[zero_mask] = np.random.randn(np.sum(zero_mask))
                norms = np.sqrt(dir_x[zero_mask]**2 + dir_y[zero_mask]**2)
                norms = np.where(norms > 1e-10, norms, 1.0)
                dir_x[zero_mask] /= norms
                dir_y[zero_mask] /= norms

            # Each circle gets pushed by half the violation
            push = viol_v * 0.5

            # Accumulate forces (using np.add.at for repeated indices)
            np.add.at(fx, vi, dir_x * push)
            np.add.at(fy, vi, dir_y * push)
            np.add.at(fx, vj, -dir_x * push)
            np.add.at(fy, vj, -dir_y * push)

            # Apply displacement with damping
            damping = 0.5
            x += damping * fx
            y += damping * fy

            # Clamp to boundaries
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)

            # Re-clamp radii to boundary
            max_r_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
            r = np.minimum(r, max_r_bound)
            r = np.maximum(r, 1e-8)

        # Final boundary clamp
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        max_r_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, max_r_bound)
        r = np.maximum(r, 1e-8)

        return x, y, r

    def grow_radii(x, y, r, growth_steps=50, growth_factor=1.01):
        """
        Greedily grow radii while maintaining validity.
        At each step, try to increase all radii by a small factor,
        then project back to validity.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()

        for step in range(growth_steps):
            # Try to grow all radii
            r_new = r * growth_factor

            # Check boundary
            max_r_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
            r_new = np.minimum(r_new, max_r_bound)

            # Check pairwise: if any overlap, reduce growth
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx * dx + dy * dy)
            sum_r_new = r_new[i_idx] + r_new[j_idx]

            if np.any(sum_r_new > dist + 1e-10):
                # Scale back: find max allowed scale
                # We want r_new[i] + r_new[j] <= dist for all pairs
                # Try a smaller growth
                r_new = r * (1 + (growth_factor - 1) * 0.5)
                max_r_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
                r_new = np.minimum(r_new, max_r_bound)

                # If still overlapping, don't grow this step
                dx = x[i_idx] - x[j_idx]
                dy = y[i_idx] - y[j_idx]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r_new = r_new[i_idx] + r_new[j_idx]
                if np.any(sum_r_new > dist + 1e-10):
                    # Can't grow, try to move circles to make room
                    # Use push algorithm to create space
                    x, y, r = project_valid(x, y, r, max_iter=50)
                    # Re-try growth
                    r_new = r * growth_factor
                    max_r_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
                    r_new = np.minimum(r_new, max_r_bound)
                    dx = x[i_idx] - x[j_idx]
                    dy = y[i_idx] - y[j_idx]
                    dist = np.sqrt(dx * dx + dy * dy)
                    sum_r_new = r_new[i_idx] + r_new[j_idx]
                    if np.any(sum_r_new > dist + 1e-10):
                        continue

            r = r_new
            # Project to ensure validity
            x, y, r = project_valid(x, y, r, max_iter=30)

        return x, y, r

    def push_and_grow(x, y, r, time_limit):
        """
        Combined push-apart and radius growth algorithm.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()

        # First project to validity
        x, y, r = project_valid(x, y, r, max_iter=100)

        # Then grow radii
        x, y, r = grow_radii(x, y, r, growth_steps=80, growth_factor=1.015)

        # Final polish: a few more push+grow cycles
        for _ in range(5):
            if time.time() > time_limit:
                break
            x, y, r = project_valid(x, y, r, max_iter=50)
            x, y, r = grow_radii(x, y, r, growth_steps=20, growth_factor=1.01)

        return x, y, r

    def soft_penalty_objective(z):
        """
        Smooth penalty objective using soft-max (log-sum-exp) for penalties.
        z = [x_1..x_N, y_1..y_N, r_1..r_N]
        """
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:]

        # Objective: maximize sum(r) -> minimize -sum(r)
        obj = -np.sum(r)

        # Boundary violations (smooth soft-max)
        v_left = r - x
        v_right = r + x - 1.0
        v_bottom = r - y
        v_top = r + y - 1.0

        # Soft-max approximation of max(0, v)^2 using log-sum-exp
        # For a vector v, soft_max(v, beta) ≈ (1/beta) * log(sum(exp(beta*v)))
        # We want sum of max(0, v_i)^2, approximated smoothly
        def soft_penalty(v, beta=50.0):
            # Smooth approximation of max(0, v)^2
            # Use: (v + softplus(-v))^2 where softplus(t) = log(1+exp(t))
            # max(0,v) ≈ v + softplus(-v) - softplus(0) ... actually simpler:
            # max(0, v) ≈ softplus(v * beta) / beta
            sp = np.logaddexp(0, beta * v) / beta  # smooth max(0, v)
            return np.sum(sp * sp)

        pen_boundary = soft_penalty(v_left) + soft_penalty(v_right) + \
                       soft_penalty(v_bottom) + soft_penalty(v_top)

        # Pairwise violations
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist_sq = dx * dx + dy * dy
        r_sum = r[i_idx] + r[j_idx]
        r_sum_sq = r_sum * r_sum

        # violation = r_sum_sq - dist_sq (positive when overlapping)
        # We want to penalize max(0, violation)
        violation_pairs = r_sum_sq - dist_sq
        pen_pairs = soft_penalty(violation_pairs, beta=20.0)

        M = 500.0
        total = obj + M * (pen_boundary + pen_pairs)

        return total

    def local_optimize_scipy(z0, time_limit):
        """
        Run L-BFGS-B with smooth penalty, then polish with push-and-grow.
        """
        x0 = z0[:N]
        y0 = z0[N:2*N]
        r0 = z0[2*N:]

        # Basic boundary clipping
        max_r_bound = np.minimum(np.minimum(x0, 1 - x0), np.minimum(y0, 1 - y0))
        r0 = np.minimum(r0, max_r_bound)
        r0 = np.maximum(r0, 1e-4)

        z0 = np.concatenate([x0, y0, r0])

        bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N

        best_z = z0.copy()
        best_score = 0.0

        # Stage 1: L-BFGS-B with smooth penalty
        try:
            res = minimize(
                soft_penalty_objective,
                z0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 300,
                    'maxfun': 1500,
                    'ftol': 1e-10,
                    'gtol': 1e-7
                }
            )
            best_z = res.x
        except Exception:
            pass

        # Stage 2: Polish with SLSQP on a tighter region
        if time.time() < time_limit - 3.0:
            x1 = best_z[:N]
            y1 = best_z[N:2*N]
            r1 = best_z[2*N:]
            # Tighten bounds around current solution
            eps = 0.05
            bounds_tight = []
            for i in range(N):
                bounds_tight.append((max(0, x1[i] - eps), min(1, x1[i] + eps)))
                bounds_tight.append((max(0, y1[i] - eps), min(1, y1[i] + eps)))
                bounds_tight.append((max(0, r1[i] * 0.8), min(0.5, r1[i] * 1.2)))

            try:
                res2 = minimize(
                    soft_penalty_objective,
                    best_z,
                    method='SLSQP',
                    bounds=bounds_tight,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-12
                    }
                )
                if res2.fun < soft_penalty_objective(best_z):
                    best_z = res2.x
            except Exception:
                pass

        # Stage 3: Push-and-grow polish (the key differentiator)
        x_final = best_z[:N]
        y_final = best_z[N:2*N]
        r_final = best_z[2*N:]

        # Project to validity
        x_final, y_final, r_final = project_valid(x_final, y_final, r_final, max_iter=100)

        # Grow radii
        x_final, y_final, r_final = grow_radii(x_final, y_final, r_final,
                                                 growth_steps=60, growth_factor=1.015)

        # Additional polish cycles
        for _ in range(3):
            if time.time() > time_limit:
                break
            x_final, y_final, r_final = project_valid(x_final, y_final, r_final, max_iter=50)
            x_final, y_final, r_final = grow_radii(x_final, y_final, r_final,
                                                     growth_steps=15, growth_factor=1.01)

        score = np.sum(r_final)

        # Also try push-and-grow from the initial state
        x_push = x0.copy()
        y_push = y0.copy()
        r_push = r0.copy()
        x_push, y_push, r_push = push_and_grow(x_push, y_push, r_push, time_limit)
        score_push = np.sum(r_push)

        if score_push > score:
            x_final = x_push
            y_final = y_push
            r_final = r_push
            score = score_push

        return np.concatenate([x_final, y_final, r_final]), score

    best_global_state = None
    best_global_score = 0.0

    # Multi-Start Configurations
    num_starts = 12
    time_per_start = (deadline - start_time) / num_starts
    time_per_start = max(time_per_start, 8.0)

    for k in range(num_starts):
        if time.time() > deadline - 15.0:
            break

        start_k = time.time()
        time_limit_k = min(start_k + time_per_start, deadline - 10.0)

        # Generate Initial State z0 (shape 3N)
        if k % 3 == 0:
            # Grid
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            count = 0
            for i in range(side):
                for j in range(side):
                    if count < N:
                        x[count] = (i + 0.5) / side
                        y[count] = (j + 0.5) / side
                        r[count] = 0.08
                        count += 1
        elif k % 3 == 1:
            # Random
            x = np.random.rand(N) * 0.8 + 0.1
            y = np.random.rand(N) * 0.8 + 0.1
            r = np.full(N, 0.05)
        else:
            # Ring: 1 center, 8 inner, 17 outer
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.full(N, 0.05)

            # Center
            x[0] = 0.5
            y[0] = 0.5

            # Inner ring (8)
            angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            rad1 = 0.15
            x[1:9] = 0.5 + rad1 * np.cos(angles1)
            y[1:9] = 0.5 + rad1 * np.sin(angles1)

            # Outer ring (17)
            angles2 = np.linspace(0, 2 * np.pi, 17, endpoint=False)
            rad2 = 0.35
            x[9:] = 0.5 + rad2 * np.cos(angles2)
            y[9:] = 0.5 + rad2 * np.sin(angles2)

        # Add noise to break symmetry
        x += np.random.normal(0, 0.005, N)
        y += np.random.normal(0, 0.005, N)

        # Clip to [0.01, 0.99] to stay away from boundaries initially
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)

        z0 = np.concatenate([x, y, r])

        # Run optimization
        candidate_z, candidate_score = local_optimize_scipy(z0, time_limit_k)

        if candidate_score > best_global_score:
            best_global_score = candidate_score
            best_global_state = candidate_z.copy()

    # Fallback if no valid state found
    if best_global_state is None:
        x = np.full(N, 0.5)
        y = np.full(N, 0.5)
        r = np.full(N, 0.01)
        best_global_state = np.concatenate([x, y, r])
        best_global_score = np.sum(r)

    # Final safety projection
    x = best_global_state[:N]
    y = best_global_state[N:2*N]
    r = best_global_state[2*N:]

    # Boundary
    max_r_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    r = np.minimum(r, max_r_bound)

    # Pairwise: use push algorithm for final validity
    x, y, r = project_valid(x, y, r, max_iter=100)

    # One final growth attempt
    if time.time() < deadline - 5.0:
        x, y, r = grow_radii(x, y, r, growth_steps=10, growth_factor=1.005)
        x, y, r = project_valid(x, y, r, max_iter=50)

    # Construct output array
    final_state = np.column_stack([x, y, r])

    return final_state
