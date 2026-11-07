"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """L-BFGS with random restarts for Bent Cigar optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        # Count function calls
        call_count = [0]
        max_calls = budget

        def clipped_func(x):
            if call_count[0] >= max_calls:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            val = float(func(x_clipped))
            call_count[0] += 1
            return val

        # L-BFGS implementation with m=10 history
        def lbfgs_minimize(func, x0, m=10, max_iter=100, tol=1e-8):
            """Simple L-BFGS with finite difference gradients."""
            dim = len(x0)
            x = x0.copy()
            fx = func(x)

            # Finite difference gradient
            def gradient(x):
                g = np.zeros(dim)
                h = 1e-5
                for i in range(dim):
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += h
                    xm[i] -= h
                    g[i] = (func(xp) - func(xm)) / (2 * h)
                return g

            # L-BFGS history
            s_hist = []  # step vectors
            y_hist = []  # gradient differences
            m_hist = []  # inner products s'y

            g = gradient(x)
            num_grad_calls = 1

            for it in range(max_iter):
                # Check time
                if time.time() > deadline:
                    break

                # Check convergence
                if np.linalg.norm(g, ord=np.inf) < tol:
                    break

                # Compute search direction using L-BFGS two-loop recursion
                r = -g.copy()

                # Two-loop recursion
                alphas = []
                for i in range(len(s_hist) - 1, -1, -1):
                    s_i = s_hist[i]
                    y_i = y_hist[i]
                    sy = m_hist[i]
                    alpha = np.dot(s_i, r) / sy
                    alphas.append(alpha)
                    r = r - alpha * y_i
                alphas.reverse()

                # Initial Hessian approximation
                if s_hist:
                    sy = m_hist[-1]
                    yy = np.dot(y_hist[-1], y_hist[-1])
                    gamma = sy / yy if yy > 0 else 1.0
                else:
                    gamma = 1.0

                p = gamma * r
                for i in range(len(s_hist)):
                    s_i = s_hist[i]
                    y_i = y_hist[i]
                    sy = m_hist[i]
                    beta = np.dot(y_i, p) / sy
                    p = p + (alphas[i] - beta) * s_i

                # Line search (backtracking)
                step_size = 1.0
                g_norm = np.linalg.norm(g)
                for ls_iter in range(30):
                    x_new = np.clip(x + step_size * p, -5, 5)
                    fx_new = func(x_new)

                    if fx_new < fx - 1e-4 * step_size * np.dot(g, p):
                        break

                    step_size *= 0.5

                else:
                    # No improvement, try smaller steps
                    step_size = 1e-4
                    x_new = np.clip(x + step_size * p, -5, 5)
                    fx_new = func(x_new)
                    if fx_new >= fx:
                        break

                # Update history
                s_new = x_new - x
                g_new = gradient(x_new)
                y_new = g_new - g
                sy_new = np.dot(s_new, y_new)

                if sy_new > 1e-10:
                    s_hist.append(s_new)
                    y_hist.append(y_new)
                    m_hist.append(sy_new)
                    if len(s_hist) > m:
                        s_hist.pop(0)
                        y_hist.pop(0)
                        m_hist.pop(0)

                x = x_new
                fx = fx_new
                g = g_new

            return x, fx

        # Strategy: multiple random restarts with L-BFGS local optimization
        # Allocate budget across restarts
        # Each L-BFGS iteration uses ~2*dim+1 function calls for gradient
        # Plus line search calls

        rng = np.random.default_rng(42)

        # Number of restarts - we'll use a fraction of budget for exploration
        # and the rest for local optimization
        # With dim=5, each gradient eval costs ~11 calls (2*5+1)
        # Each L-BFGS iteration: 1 gradient (11 calls) + line search (~5-10 calls)
        # So roughly 15-20 calls per iteration
        # Budget 500 -> ~25-30 iterations total

        # Use 5 restarts, each with limited iterations
        num_restarts = 5
        calls_per_restart = budget // num_restarts

        best_x = None
        best_f = float('inf')

        for restart in range(num_restarts):
            if time.time() > deadline or call_count[0] >= max_calls:
                break

            # Generate starting point
            if restart == 0:
                # Start at origin or random
                x0 = rng.uniform(-3, 3, dim)
            else:
                # Subsequent restarts: random points, possibly biased toward previous best
                if best_x is not None and restart < num_restarts - 1:
                    # Mix of random and perturbation of best
                    if rng.random() < 0.5:
                        x0 = rng.uniform(-5, 5, dim)
                    else:
                        x0 = np.clip(best_x + rng.normal(0, 2, dim), -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)

            # Run L-BFGS with limited iterations based on remaining budget
            remaining_calls = max_calls - call_count[0]
            # Estimate iterations possible: each iter ~20 calls
            max_iters = min(50, max(5, remaining_calls // 25))

            x_opt, f_opt = lbfgs_minimize(clipped_func, x0, m=10, max_iter=max_iters)

            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt

            # Check if we've used most of the budget
            if call_count[0] >= max_calls - 50:
                break

        # If we have budget left, do a final polish with L-BFGS from best point
        if best_x is not None and call_count[0] < max_calls - 20 and time.time() < deadline:
            remaining = max_calls - call_count[0]
            max_iters = min(30, remaining // 25)
            if max_iters > 3:
                x_final, f_final = lbfgs_minimize(clipped_func, best_x, m=10, max_iter=max_iters)
                if f_final < best_f:
                    best_f = f_final
                    best_x = x_final

        return best_x, best_f
# EVOLVE-BLOCK-END
