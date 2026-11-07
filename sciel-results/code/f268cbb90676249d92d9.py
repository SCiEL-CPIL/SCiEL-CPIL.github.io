"""Model-based optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-and-optimize approach with BFGS refinement:
    - Build a local quadratic model from a set of evaluations.
    - Use the model to propose the next point.
    - Adapt the model as new evaluations come in.
    - BFGS-based local refinement for faster convergence.
    - Deadline guard to ensure we return before the time limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # safety margin before 330s kill
        rng = np.random.default_rng(123)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline - 2:
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Store all evaluations for model building
        # Each entry: (x, f)
        evals = []

        def record_eval(x, fx):
            if fx is not None:
                evals.append((x.copy(), fx))

        # Phase 1: Initial sampling to build a model
        # Start at origin and a few random points
        initial_starts = [np.zeros(d)]
        for _ in range(8):
            initial_starts.append(rng.uniform(-3, 3, d))

        for s in initial_starts:
            fx = try_eval(s)
            if fx is None:
                break
            record_eval(s, fx)

        if calls_used >= max_calls or time.time() > deadline - 5:
            return best_x, best_f

        # Phase 2: Model-based optimization loop
        # We maintain a local quadratic model: f(x) ≈ f0 + g^T (x - x0) + 0.5 (x-x0)^T H (x-x0)
        # We use a trust-region approach: propose x_new = x0 - H^{-1} g within a trust radius

        x0 = best_x.copy()
        fx0 = best_f

        # Build initial model from evaluations near x0
        def build_model(center, radius):
            """Build a quadratic model from evaluations within radius of center."""
            # Select points within radius
            selected = []
            for (x, f) in evals:
                if np.linalg.norm(x - center) <= radius * 1.5:
                    selected.append((x, f))

            if len(selected) < d + 1:
                return None

            # Fit quadratic model using least squares
            # f(x) = f0 + g^T (x - center) + 0.5 (x-center)^T H (x-center)
            # Parameters: f0 (1), g (d), H (d*(d+1)/2) -> total 1 + d + d*(d+1)/2
            n_params = 1 + d + d * (d + 1) // 2

            if len(selected) < n_params:
                return None

            # Build design matrix
            A = []
            b = []
            for (x, f) in selected:
                dx = x - center
                row = [1.0]
                row.extend(dx)
                # Hessian terms: H_{ij} for i <= j
                for i in range(d):
                    for j in range(i, d):
                        if i == j:
                            row.append(0.5 * dx[i] * dx[j])
                        else:
                            row.append(dx[i] * dx[j])
                A.append(row)
                b.append(f)

            A = np.array(A)
            b = np.array(b)

            try:
                params, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None

            # Check if model is reasonable
            if rank < n_params:
                # Degenerate, but still use it
                pass

            f0 = params[0]
            g = params[1:1 + d]
            h_flat = params[1 + d:]

            # Reconstruct H
            H = np.zeros((d, d))
            idx = 0
            for i in range(d):
                for j in range(i, d):
                    H[i, j] = h_flat[idx]
                    H[j, i] = h_flat[idx]
                    idx += 1

            return f0, g, H

        # Propose next point using the model
        def propose_from_model(model, center, trust_radius):
            if model is None:
                return None
            f0, g, H = model

            # Regularize H for invertibility
            H_reg = H + 1e-8 * np.eye(d)

            try:
                step = np.linalg.solve(H_reg, g)
            except np.linalg.LinAlgError:
                try:
                    step = np.linalg.lstsq(H_reg, g, rcond=None)[0]
                except:
                    return None

            # Clip step to trust region
            step_norm = np.linalg.norm(step)
            if step_norm > trust_radius:
                step = step * (trust_radius / step_norm)

            x_new = center + step
            x_new = np.clip(x_new, -5, 5)
            return x_new

        # BFGS refinement function
        def bfgs_refine(start_x, start_f, max_iters=200, tol=1e-12):
            """
            BFGS refinement using finite difference gradients.
            Returns (best_x, best_f, calls_used_in_refine)
            """
            nonlocal calls_used, best_x, best_f

            x = start_x.copy()
            f = start_f
            H = np.eye(d)  # BFGS Hessian approximation
            h = 1e-4  # finite difference step

            for it in range(max_iters):
                if calls_used >= max_calls:
                    break
                if time.time() > deadline - 3:
                    break

                # Compute gradient using central differences
                grad = np.zeros(d)
                for i in range(d):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += h
                    x_minus[i] -= h

                    fx_plus = try_eval(x_plus)
                    if fx_plus is None:
                        break
                    record_eval(x_plus, fx_plus)

                    fx_minus = try_eval(x_minus)
                    if fx_minus is None:
                        break
                    record_eval(x_minus, fx_minus)

                    grad[i] = (fx_plus - fx_minus) / (2 * h)

                if calls_used >= max_calls or time.time() > deadline - 3:
                    break

                # Check convergence
                if np.linalg.norm(grad) < tol:
                    break

                # Compute BFGS step: s = -H * grad
                s = -H @ grad

                # Line search to find step size
                alpha = 1.0
                improved = False
                for ls_iter in range(30):
                    x_trial = x + alpha * s
                    x_trial = np.clip(x_trial, -5, 5)

                    f_trial = try_eval(x_trial)
                    if f_trial is None:
                        break
                    record_eval(x_trial, f_trial)

                    if f_trial < f:
                        # Accept step
                        # Update BFGS
                        s_actual = x_trial - x
                        f_diff = f_trial - f
                        g_diff = None

                        # Recompute gradient at new point for BFGS update
                        grad_new = np.zeros(d)
                        for i in range(d):
                            x_plus = x_trial.copy()
                            x_minus = x_trial.copy()
                            x_plus[i] += h
                            x_minus[i] -= h

                            fx_plus = try_eval(x_plus)
                            if fx_plus is None:
                                break
                            record_eval(x_plus, fx_plus)

                            fx_minus = try_eval(x_minus)
                            if fx_minus is None:
                                break
                            record_eval(x_minus, fx_minus)

                            grad_new[i] = (fx_plus - fx_minus) / (2 * h)

                        if calls_used >= max_calls or time.time() > deadline - 3:
                            break

                        g_diff = grad_new - grad
                        y = g_diff
                        sy = s_actual @ y

                        if sy > 1e-10:
                            # BFGS update
                            Hy = H @ y
                            H = H + (s_actual @ s_actual.T) / sy - (Hy @ Hy.T) / (y @ Hy)

                        x = x_trial
                        f = f_trial
                        improved = True
                        break
                    else:
                        alpha *= 0.5

                if not improved:
                    # Try smaller step or restart
                    alpha = 0.01
                    x_trial = x + alpha * s
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = try_eval(x_trial)
                    if f_trial is not None:
                        record_eval(x_trial, f_trial)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                    else:
                        break

                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            return x, f

        # Main loop: trust-region model-based optimization
        trust_radius = 1.0
        no_improve_count = 0
        max_no_improve = 10

        while calls_used < max_calls and time.time() < deadline - 15:
            # Build model
            model = build_model(x0, trust_radius)

            if model is None:
                # Can't build model, try a random step
                x_new = np.clip(x0 + trust_radius * rng.standard_normal(d), -5, 5)
            else:
                x_new = propose_from_model(model, x0, trust_radius)
                if x_new is None:
                    x_new = np.clip(x0 + trust_radius * rng.standard_normal(d), -5, 5)

            fx_new = try_eval(x_new)
            if fx_new is None:
                break
            record_eval(x_new, fx_new)

            if fx_new < fx0 - 1e-10:
                # Improvement
                fx0 = fx_new
                x0 = x_new.copy()
                no_improve_count = 0
                # Expand trust region
                trust_radius = min(trust_radius * 1.3, 3.0)
            else:
                # No improvement
                no_improve_count += 1
                # Shrink trust region
                trust_radius = max(trust_radius * 0.8, 0.01)

                if no_improve_count >= max_no_improve:
                    # Try BFGS refinement
                    if calls_used < max_calls - 50 and time.time() < deadline - 30:
                        x_refined, f_refined = bfgs_refine(x0, fx0, max_iters=100)
                        if f_refined < fx0:
                            fx0 = f_refined
                            x0 = x_refined.copy()
                            no_improve_count = 0
                            trust_radius = 1.0
                        else:
                            no_improve_count = 0
                            trust_radius = max(trust_radius * 0.5, 0.005)
                    else:
                        # Try a larger random jump
                        jump_size = 1.0
                        x_new = np.clip(x0 + jump_size * rng.standard_normal(d), -5, 5)
                        fx_new = try_eval(x_new)
                        if fx_new is None:
                            break
                        record_eval(x_new, fx_new)

                        if fx_new < fx0:
                            fx0 = fx_new
                            x0 = x_new.copy()
                            no_improve_count = 0
                            trust_radius = 1.0
                        else:
                            no_improve_count = 0
                            trust_radius = max(trust_radius * 0.5, 0.005)

        # Phase 3: Final BFGS polish pass
        if calls_used < max_calls - 30 and time.time() < deadline - 5 and best_x is not None:
            x_polished, f_polished = bfgs_refine(best_x, best_f, max_iters=150, tol=1e-14)
            if f_polished < best_f:
                best_f = f_polished
                best_x = x_polished.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
