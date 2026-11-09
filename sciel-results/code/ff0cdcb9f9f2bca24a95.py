"""Stochastic quasi-Newton (Broyden) optimizer with Nelder-Mead refinement and restarts."""
import numpy as np
import time


class Optimizer:
    """Broyden quasi-Newton with NM refinement and multiple restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = np.full(n, np.nan)
        best_f = np.inf
        calls_used = 0

        def time_ok():
            return time.time() < deadline

        # Number of restarts and budget per restart
        n_restarts = 6
        budget_per_restart = max(1, (total_budget - 10) // n_restarts)

        for restart in range(n_restarts):
            if not time_ok() or calls_used >= total_budget:
                break

            # Initial point: first restart at origin, others random
            if restart == 0:
                x = np.zeros(n)
            else:
                x = rng.uniform(-4, 4, n)

            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # --- Broyden quasi-Newton phase ---
            # We'll use a simplified Broyden's method with line search
            # Approximate Jacobian initially with finite differences
            # But since we only have function values, we use a gradient-free
            # quasi-Newton approach: Broyden's method for minimization

            # Initialize: use a simple gradient approximation
            # For Broyden, we need to track the Jacobian approximation H
            # and the search direction

            # Simpler approach: use Nelder-Mead for the first phase,
            # then Broyden for refinement. Or just use a robust
            # derivative-free method.

            # Let's do: Nelder-Mead with adaptive simplex, then
            # a local Broyden-like refinement.

            # Actually, let's implement a proper stochastic quasi-Newton:
            # We maintain a Hessian approximation B and update it using
            # secant conditions. Since we don't have gradients, we'll
            # estimate gradients via finite differences.

            # Phase 1: Nelder-Mead (robust, no gradient needed)
            remaining = total_budget - calls_used
            nm_budget = min(budget_per_restart, max(30, remaining // 2))

            x_nm = self._nelder_mead(func, x, fx, nm_budget, rng, calls_used)
            calls_used = x_nm[1]
            if x_nm[0] < best_f:
                best_f, best_x = x_nm[0], x_nm[2]

            if not time_ok() or calls_used >= total_budget:
                break

            # Phase 2: Broyden quasi-Newton refinement
            remaining = total_budget - calls_used
            bn_budget = min(budget_per_restart - nm_budget, max(10, remaining // 2))
            if bn_budget > 5:
                x_bn = self._broyden(func, x_nm[2], x_nm[0], bn_budget, rng, calls_used)
                calls_used = x_bn[1]
                if x_bn[0] < best_f:
                    best_f, best_x = x_bn[0], x_bn[2]

            if not time_ok():
                break

        # Final: use any remaining budget for a few more NM iterations around best
        remaining = total_budget - calls_used
        if remaining > 20 and time_ok():
            x_final = self._nelder_mead(func, best_x, best_f, min(remaining, 50), rng, calls_used)
            if x_final[0] < best_f:
                best_f, best_x = x_final[0], x_final[2]

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, budget, rng, calls_offset):
        """Nelder-Mead simplex method. Returns (best_f, calls_used, best_x)."""
        n = self.dim
        calls = calls_offset

        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [f0]

        # Add perturbed points
        for i in range(n):
            xi = x0.copy()
            step = 0.25 if abs(x0[i]) < 1e-10 else 0.05 * abs(x0[i])
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            calls += 1
            simplex.append(xi)
            fvals.append(fi)

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma_shrink = 0.5

        best_f = min(fvals)
        best_x = simplex[np.argmin(fvals)].copy()

        max_iter = max(1, budget - (n + 1))
        for _ in range(max_iter):
            if calls >= calls_offset + budget:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            if fvals[-1] - fvals[0] < 1e-12:
                break

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fi = func(simplex[i])
                        calls += 1
                        fvals[i] = fi

            # Track best
            cur_best = min(fvals)
            if cur_best < best_f:
                best_f = cur_best
                best_x = simplex[np.argmin(fvals)].copy()

            if calls >= calls_offset + budget:
                break

        return best_f, calls, best_x

    def _broyden(self, func, x0, f0, budget, rng, calls_offset):
        """Broyden quasi-Newton with finite-difference gradients. Returns (best_f, calls_used, best_x)."""
        n = self.dim
        calls = calls_offset
        x = x0.copy()
        fx = f0

        # Initial Hessian approximation: identity
        B = np.eye(n)

        # Gradient estimation via central differences
        def grad(x):
            g = np.zeros(n)
            h = 1e-5
            for i in range(n):
                xp = x.copy()
                xm = x.copy()
                xp[i] = min(5.0, xp[i] + h)
                xm[i] = max(-5.0, xm[i] - h)
                fp = func(xp)
                fm = func(xm)
                calls_holder[0] += 2
                g[i] = (fp - fm) / (xp[i] - xm[i])
            return g

        calls_holder = [0]

        best_f = fx
        best_x = x.copy()
        step_size = 1.0

        max_iter = max(1, budget)
        for it in range(max_iter):
            if calls + calls_holder[0] >= calls_offset + budget:
                break
            if not (time.time() < 280.0):
                break

            g = grad(x)
            calls += calls_holder[0]
            calls_holder[0] = 0

            if np.linalg.norm(g) < 1e-8:
                break

            # Solve B * d = -g
            try:
                d = np.linalg.solve(B, -g)
            except np.linalg.LinAlgError:
                d = -g

            # Line search
            alpha = step_size
            for _ in range(20):
                x_new = np.clip(x + alpha * d, -5, 5)
                f_new = func(x_new)
                calls += 1
                if f_new < fx:
                    break
                alpha *= 0.5
            else:
                # No improvement, shrink
                step_size *= 0.5
                if step_size < 1e-6:
                    break
                continue

            # Update B using Broyden's formula
            dx = x_new - x
            g_new = grad(x_new)
            calls += calls_holder[0]
            calls_holder[0] = 0
            dg = g_new - g

            if np.linalg.norm(dg) > 1e-12 and np.linalg.norm(dx) > 1e-12:
                B = B + np.outer(dg - B @ dx, dx) / (dx @ dx)

            x = x_new
            fx = f_new
            step_size = min(1.0, alpha * 1.2)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            if calls >= calls_offset + budget:
                break

        return best_f, calls, best_x
