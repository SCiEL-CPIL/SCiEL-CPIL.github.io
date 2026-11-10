import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 295  # 5s safety margin before 300s limit

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with Sobol-like initialization
        # Use a quasi-random sequence for better coverage
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1

        # Generate initial points using Halton sequence for better spread
        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i = i // base
            return result

        # Generate 20 initial points using Halton sequence
        num_initial = min(20, budget - 1)
        for k in range(1, num_initial + 1):
            if calls >= budget or time.time() > self.deadline:
                break
            x_new = np.zeros(dim)
            for d in range(dim):
                x_new[d] = -5 + 10 * halton(k, d + 2)  # base 2,3,4,5,6 for dim 5
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls += 1
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

        # Phase 2: Trust-region with quadratic surrogate
        # Build a local quadratic model from recent evaluations
        # Store recent points for surrogate construction
        recent_points = [(best_x.copy(), best_f)]
        max_recent = 15  # Keep last 15 points for surrogate

        # Initial step size for trust region
        tr_radius = 1.0
        min_tr_radius = 1e-8
        max_tr_radius = 5.0

        # Main trust-region loop
        while calls < budget and time.time() < self.deadline:
            if tr_radius < min_tr_radius:
                break

            # Build quadratic surrogate from recent points
            # Use points within the current trust region
            tr_points = [best_x.copy()]
            tr_fvals = [best_f]

            # Add points from recent history that are within trust region
            for px, pf in recent_points:
                if np.linalg.norm(px - best_x) <= tr_radius * 1.5:
                    if not np.array_equal(px, best_x):
                        tr_points.append(px)
                        tr_fvals.append(pf)

            # If not enough points, evaluate new ones
            while len(tr_points) < min(dim + 1, 8) and calls < budget and time.time() < self.deadline:
                # Generate point in trust region
                direction = rng.normal(0, 1, dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                x_new = best_x + direction * tr_radius * 0.5
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls += 1
                tr_points.append(x_new)
                tr_fvals.append(f_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

            if len(tr_points) < dim + 1:
                break

            # Fit quadratic model: f(x) ≈ f0 + g^T(x-x0) + 0.5(x-x0)^T H (x-x0)
            # Use least squares
            X = np.array(tr_points)
            f = np.array(tr_fvals)
            x0_model = best_x

            # Compute relative positions
            D = X - x0_model  # (n, dim)
            delta_f = f - f[0]

            # Build design matrix for quadratic fit
            # Columns: identity (for g), then H terms
            n = len(X)
            dim_h = dim * (dim + 1) // 2  # Number of unique H entries

            # Design matrix A: n x (1 + dim + dim_h)
            A = np.zeros((n, 1 + dim + dim_h))
            A[:, 0] = 1.0  # Constant term (should be ~0)
            A[:, 1:1+dim] = D  # Linear terms

            # Quadratic terms
            col = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        A[:, col] = D[:, i] ** 2
                    else:
                        A[:, col] = 2 * D[:, i] * D[:, j]
                    col += 1

            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, delta_f, rcond=None)
            except:
                break

            # Extract gradient and Hessian
            g = coeffs[1:1+dim]
            H_flat = coeffs[1+dim:]

            # Reconstruct Hessian
            H = np.zeros((dim, dim))
            col = 0
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        H[i, i] = H_flat[col]
                    else:
                        H[i, j] = H_flat[col] / 2
                        H[j, i] = H_flat[col] / 2
                    col += 1

            # Ensure H is symmetric
            H = (H + H.T) / 2

            # Find trust-region step: minimize q(s) = g^T s + 0.5 s^T H s subject to ||s|| <= tr_radius
            # Use simple approach: try Newton step, clip to trust region
            try:
                # Newton step: H s = -g
                s_newton = np.linalg.solve(H + 1e-10 * np.eye(dim), -g)
            except:
                s_newton = -g / (np.linalg.norm(g) + 1e-10) * tr_radius

            s_norm = np.linalg.norm(s_newton)
            if s_norm > tr_radius:
                s_newton = s_newton * (tr_radius / s_norm)
            elif s_norm < 1e-10:
                # If step is tiny, use gradient direction
                g_norm = np.linalg.norm(g)
                if g_norm > 1e-10:
                    s_newton = -g * (tr_radius / g_norm)
                else:
                    break  # Converged

            # Evaluate at proposed point
            x_trial = best_x + s_newton
            x_trial = np.clip(x_trial, -5, 5)
            f_trial = func(x_trial)
            calls += 1

            # Compute actual vs predicted improvement
            predicted_improvement = -g @ s_newton - 0.5 * (s_newton @ H @ s_newton)
            actual_improvement = f[0] - f_trial

            # Compute trust region ratio
            if abs(predicted_improvement) > 1e-10:
                rho_ratio = actual_improvement / predicted_improvement
            else:
                rho_ratio = 1.0 if actual_improvement > 0 else 0.0

            # Update trust region
            if rho_ratio > 0.7:
                tr_radius = min(2 * tr_radius, max_tr_radius)
            elif rho_ratio > 0.25:
                tr_radius = tr_radius  # Keep same
            else:
                tr_radius = max(0.5 * tr_radius, min_tr_radius)

            # Update best if improved
            if f_trial < best_f:
                best_f = f_trial
                best_x = x_trial.copy()

            # Update recent points
            recent_points.append((x_trial.copy(), f_trial))
            if len(recent_points) > max_recent:
                recent_points.pop(0)

        # Phase 3: Adaptive coordinate descent for final polish
        if calls < budget and time.time() < self.deadline:
            # Estimate step sizes based on trust region radius
            step_sizes = np.full(dim, max(tr_radius / 2, 1e-4))

            x = best_x.copy()
            fx = best_f

            # Adaptive coordinate descent with finer steps
            successes = np.zeros(dim)
            total = np.zeros(dim)
            success_threshold = 0.3
            min_step = 1e-8
            max_step = 1.0

            max_cd_iterations = 200
            for cd_iter in range(max_cd_iterations):
                if calls >= budget or time.time() > self.deadline:
                    break

                progress = False
                for i in range(dim):
                    if calls >= budget or time.time() > self.deadline:
                        break

                    # Try positive step
                    x_new = x.copy()
                    x_new[i] += step_sizes[i]
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls += 1

                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        successes[i] += 1
                        progress = True
                    total[i] += 1

                    # Try negative step
                    x_new = x.copy()
                    x_new[i] -= step_sizes[i]
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls += 1

                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        successes[i] += 1
                        progress = True
                    total[i] += 1

                    # Adapt step size based on success rate
                    if total[i] >= 3:
                        rate = successes[i] / total[i]
                        if rate > success_threshold:
                            step_sizes[i] *= 1.2
                        elif rate < success_threshold:
                            step_sizes[i] *= 0.6
                        successes[i] = 0
                        total[i] = 0

                    # Clamp step size
                    step_sizes[i] = np.clip(step_sizes[i], min_step, max_step)

                # If no progress in a full cycle, reduce all step sizes
                if not progress:
                    step_sizes *= 0.4
                    if np.all(step_sizes < min_step):
                        break

        return best_x, best_f
