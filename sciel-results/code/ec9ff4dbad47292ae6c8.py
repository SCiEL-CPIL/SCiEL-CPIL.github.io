"""Derivative-free trust-region BFGS optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DF-TBFGS: Derivative-free trust-region BFGS for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline - 3:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start to find good initial basins
        n_starts = 8
        starts = [np.zeros(d)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, d))

        for s in starts:
            s_clipped = np.clip(s, -5, 5)
            fx = try_eval(s_clipped)
            if fx is None:
                break

        if calls_used >= max_calls or time.time() > deadline - 15:
            return best_x, best_f

        # Phase 2: DF-TBFGS from the best starting point
        x = best_x.copy()
        fx = best_f

        # Finite-difference gradient estimation
        eps = 1e-5  # step for finite differences

        def estimate_gradient(x, fx):
            """Estimate gradient using forward differences. Costs d function evaluations."""
            grad = np.zeros(d)
            for i in range(d):
                if calls_used >= max_calls or time.time() > deadline - 5:
                    return None
                x_pert = x.copy()
                x_pert[i] += eps
                x_pert = np.clip(x_pert, -5, 5)
                fx_pert = try_eval(x_pert)
                if fx_pert is None:
                    return None
                grad[i] = (fx_pert - fx) / eps
            return grad

        # BFGS inverse Hessian approximation
        H = np.eye(d)  # H^{-1} approximation
        trust_radius = 2.0  # initial trust region radius

        # Run DF-TBFGS iterations
        max_tbfgs_iters = int((max_calls - calls_used) / (d + 2))  # rough budget per iteration
        min_trust_radius = 1e-6
        max_trust_radius = 5.0

        for it in range(max_tbfgs_iters):
            if calls_used >= max_calls or time.time() > deadline - 10:
                break

            # Estimate gradient at current point
            grad = estimate_gradient(x, fx)
            if grad is None:
                break

            # If gradient is small, we're near a minimum
            if np.linalg.norm(grad) < 1e-7:
                break

            # Compute search direction: p = -H @ grad
            p = -H @ grad
            p_norm = np.linalg.norm(p)

            if p_norm < 1e-10:
                # Direction is too small, reset H and use steepest descent
                H = np.eye(d)
                p = -grad
                p_norm = np.linalg.norm(p)
                if p_norm < 1e-10:
                    break

            # Scale p to be within trust region
            if p_norm > trust_radius:
                p = p * (trust_radius / p_norm)

            # Propose new point
            x_new = x + p
            x_new = np.clip(x_new, -5, 5)
            fx_new = try_eval(x_new)
            if fx_new is None:
                break

            # Compute actual reduction
            actual_reduction = fx - fx_new

            # Compute predicted reduction using quadratic model
            # Model: f(x) + grad^T p + 0.5 p^T (H^{-1})^{-1} p
            # Since we have H ≈ (H^{-1}), the Hessian approximation is H^{-1}
            # For simplicity, use: predicted = grad^T p + 0.5 * p^T @ np.linalg.inv(H) @ p
            # But inverting H each iteration is expensive. Use a simpler prediction.
            # Use: predicted = grad^T p (linear model) as a lower bound
            predicted_reduction = grad @ p

            # Check if the step was successful
            rho = actual_reduction / max(abs(predicted_reduction), 1e-10) if predicted_reduction != 0 else actual_reduction

            if rho > 0.25:
                # Successful step: update x, fx, and BFGS H approximation
                s = x_new - x
                y = estimate_gradient(x_new, fx_new)
                if y is None:
                    x, fx = x_new, fx_new
                    trust_radius = min(2.0 * trust_radius, max_trust_radius)
                    continue

                sy = s @ y
                if sy > 1e-10:
                    # BFGS update for H^{-1}
                    Hy = H @ y
                    H = H + (sy + y @ Hy) / (sy ** 2) * np.outer(s, s) - \
                          1.0 / sy * (np.outer(s, Hy) + np.outer(Hy, s))
                    H = (H + H.T) / 2  # enforce symmetry

                x, fx = x_new, fx_new
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            elif rho > 0.1:
                # Marginal step: accept but don't expand
                x, fx = x_new, fx_new
                # Don't update H for marginal steps to avoid instability
            else:
                # Unsuccessful step: shrink trust region
                trust_radius *= 0.5

            if trust_radius < min_trust_radius:
                # Trust region too small, try to escape local minimum
                # Reset and try a random perturbation
                trust_radius = 1.0
                H = np.eye(d)
                # Small random perturbation to escape
                x = x + 0.5 * rng.standard_normal(d)
                x = np.clip(x, -5, 5)
                fx = try_eval(x)
                if fx is None:
                    break

        # Phase 3: Polish with (1+1)-ES for final convergence
        if calls_used < max_calls and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05

            for _ in range(max_calls - calls_used):
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.85

                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
