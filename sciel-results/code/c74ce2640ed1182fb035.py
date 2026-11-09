"""Rewritten optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with Broyden quasi-Newton refinement for f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls_made = 1

        # Phase 1: Adaptive (1+1)-ES with 1/5 success rule
        # Initial sigma large to explore [-5,5]^5
        sigma = 3.0
        x = best_x.copy()
        fx = best_f
        success_count = 0
        window_size = 5  # 1/5 success rule window
        min_sigma = 1e-6
        max_sigma = 10.0

        # Phase 1: Run (1+1)-ES until we have ~150 calls left for refinement
        es_budget = max(1, budget - 150)

        while calls_made < es_budget and time.time() < deadline:
            # Generate candidate
            direction = rng.standard_normal(dim)
            candidate = x + sigma * direction
            candidate = np.clip(candidate, -5, 5)

            if time.time() >= deadline:
                break

            cf = func(candidate)
            calls_made += 1

            # Update best
            if cf < best_f:
                best_f = cf
                best_x = candidate.copy()

            # (1+1)-ES update
            if cf < fx:
                # Success
                x = candidate.copy()
                fx = cf
                success_count += 1
            else:
                # Failure: keep x, fx
                pass

            # 1/5 success rule
            if success_count >= window_size:
                sigma *= 1.25  # increase step
                success_count = 0
                window_size = 5
            elif success_count == 0 and (es_budget - calls_made) % window_size == 0:
                sigma *= 0.8  # decrease step
                window_size = 5

            sigma = np.clip(sigma, min_sigma, max_sigma)

            # Jump if stuck for too long
            if sigma < 0.01 and (es_budget - calls_made) > 20:
                # Random restart near best
                x = best_x + rng.standard_normal(dim) * 0.5
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                sigma = 0.5

        # Phase 2: Broyden's Method (quasi-Newton) refinement
        # Build approximate inverse Hessian from finite-difference gradients
        if calls_made < budget - 5 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f

            # Compute initial gradient using central differences
            def compute_gradient(x, step=1e-4):
                grad = np.zeros(dim)
                for i in range(dim):
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] = min(5.0, xp[i] + step)
                    xm[i] = max(-5.0, xm[i] - step)
                    fp = func(xp)
                    fm = func(xm)
                    calls_made += 2
                    grad[i] = (fp - fm) / (xp[i] - xm[i]) if xp[i] != xm[i] else 0.0
                return grad

            if calls_made + 2 * dim <= budget:
                g = compute_gradient(x)
                if time.time() < deadline and calls_made < budget:
                    # Broyden's method with inverse Hessian approximation
                    H_inv = np.eye(dim)  # inverse Hessian approximation
                    prev_x = x.copy()
                    prev_g = g.copy()

                    max_broyden_iters = 30
                    for _ in range(max_broyden_iters):
                        if calls_made >= budget - 5 or time.time() >= deadline:
                            break

                        # Compute search direction
                        d = -H_inv @ g

                        # Line search with backtracking
                        alpha = 1.0
                        improved = False
                        for _ls in range(20):
                            if calls_made >= budget - 2 or time.time() >= deadline:
                                break
                            new_x = x + alpha * d
                            new_x = np.clip(new_x, -5, 5)
                            new_f = func(new_x)
                            calls_made += 1

                            if new_f < fx:
                                # Success
                                new_g = compute_gradient(new_x) if calls_made + 2 * dim <= budget else None
                                if new_g is not None and time.time() < deadline:
                                    s = new_x - x
                                    y = new_g - g
                                    # Broyden update (BFGS-like)
                                    Hy = H_inv @ y
                                    if np.dot(y, Hy) > 1e-10:
                                        H_inv = H_inv + np.outer(s, s) / np.dot(y, s) - \
                                               np.outer(Hy, Hy) / np.dot(y, Hy)
                                x = new_x
                                fx = new_f
                                g = new_g if new_g is not None else g
                                improved = True
                                break
                            else:
                                alpha *= 0.5

                        if not improved:
                            # Try smaller step in gradient direction
                            alpha = 0.1
                            new_x = x - alpha * g
                            new_x = np.clip(new_x, -5, 5)
                            if calls_made < budget - 1:
                                new_f = func(new_x)
                                calls_made += 1
                                if new_f < fx:
                                    x = new_x
                                    fx = new_f
                                    if calls_made + 2 * dim <= budget and time.time() < deadline:
                                        g = compute_gradient(x)
                                    else:
                                        break
                                else:
                                    break
                            else:
                                break

                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

                        # Check convergence
                        if np.linalg.norm(g) < 1e-6:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
