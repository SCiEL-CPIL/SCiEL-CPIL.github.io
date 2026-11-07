"""Black-box continuous optimizer for BBOB-style f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start gradient-based local search with central-difference gradients."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline and calls < budget

        def compute_gradient(x, h=1e-4):
            """Compute gradient using central differences, tracking calls."""
            grad = np.zeros(n)
            nonlocal calls
            for i in range(n):
                if not time_ok():
                    return grad
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                xp = np.clip(xp, -5, 5)
                xm = np.clip(xm, -5, 5)
                fp = func(xp)
                calls += 1
                fm = func(xm)
                calls += 1
                grad[i] = (fp - fm) / (2 * h)
            return grad

        def gradient_descent(x0, f0, max_iter=80):
            """Gradient descent with backtracking line search."""
            nonlocal calls, best_x, best_f
            x = x0.copy()
            fx = f0
            lr = 0.5  # initial learning rate
            min_lr = 1e-6
            max_lr = 2.0
            c1 = 1e-4  # sufficient decrease
            rho = 0.5  # shrink factor for line search
            h = 1e-4   # gradient step size

            for it in range(max_iter):
                if not time_ok():
                    break

                # Compute gradient
                grad = compute_gradient(x, h)
                if not time_ok():
                    break

                # Check for convergence
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-8:
                    break

                # Backtracking line search
                step = -lr * grad
                t = 1.0
                accepted = False
                x_trial = None
                f_trial = None
                for _ in range(10):
                    if not time_ok():
                        break
                    x_trial = np.clip(x + t * step, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1
                    if f_trial <= fx - c1 * t * np.dot(grad, step):
                        accepted = True
                        break
                    t *= rho

                if accepted:
                    # Success: move to trial point
                    x = x_trial
                    fx = f_trial
                    lr = min(lr * 1.5, max_lr)  # increase step size
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure: decrease learning rate
                    lr *= 0.5
                    if lr < min_lr:
                        # Try with smaller gradient step
                        h = max(h * 0.5, 1e-6)
                        if h <= 1e-6:
                            break

            return x, fx

        # Multi-start strategy: origin + several random starts
        rng = np.random.default_rng(42)

        # Start 1: origin
        x0 = np.zeros(n)
        f0 = func(x0)
        calls += 1
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        if time_ok():
            gradient_descent(x0, f0, max_iter=80)

        # Additional random starts
        num_starts = 5
        for s in range(num_starts):
            if not time_ok():
                break

            # Generate random start point
            x0 = rng.uniform(-4, 4, n)
            f0 = func(x0)
            calls += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Only do gradient descent if this is a promising start
            if np.isfinite(best_f) and f0 < best_f + 50:
                gradient_descent(x0, f0, max_iter=50)

        # Final local refinement around the best point found
        if time_ok() and np.isfinite(best_f):
            # Try a few small perturbations around best
            for _ in range(3):
                if not time_ok():
                    break
                x0 = np.clip(best_x + 0.1 * rng.standard_normal(n), -5, 5)
                f0 = func(x0)
                calls += 1
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()
                    gradient_descent(x0, f0, max_iter=30)

        return best_x, best_f
# EVOLVE-BLOCK-END
