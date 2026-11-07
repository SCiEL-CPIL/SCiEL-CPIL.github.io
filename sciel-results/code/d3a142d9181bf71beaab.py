"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free BFGS optimizer with finite differences."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin

        # State
        x = np.random.uniform(-5, 5, self.dim)
        f_x = func(x)
        calls = 1
        best_x = x.copy()
        best_f = f_x

        # BFGS State
        H = np.eye(self.dim)  # Inverse Hessian approximation
        eps = 1e-5

        def estimate_gradient(x, f_x):
            """Estimate gradient using forward differences."""
            g = np.zeros(self.dim)
            for i in range(self.dim):
                x_pert = x.copy()
                x_pert[i] += eps
                # Clip to bounds
                x_pert = np.clip(x_pert, -5, 5)
                # If clipped, we might need a backward diff or smaller eps, 
                # but for now assume interior or handle edge case
                if np.all(x_pert == x):
                    # At boundary, use backward diff
                    x_pert = x.copy()
                    x_pert[i] -= eps
                    x_pert = np.clip(x_pert, -5, 5)
                    if np.all(x_pert == x):
                        g[i] = 0.0
                        continue
                    f_pert = func(x_pert)
                    nonlocal calls
                    calls += 1
                    g[i] = (f_x - f_pert) / eps
                else:
                    f_pert = func(x_pert)
                    nonlocal calls
                    calls += 1
                    g[i] = (f_pert - f_x) / eps
            return g

        # Main Loop
        while calls < self.budget and time.time() < deadline:
            # Check if we are running low on budget for a full step (2*dim + 1)
            if calls + 2 * self.dim + 1 > self.budget:
                break

            # 1. Estimate Gradient at x
            g = estimate_gradient(x, f_x)
            
            # If gradient is tiny, we are likely at the minimum
            if np.linalg.norm(g) < 1e-8:
                break

            # 2. Compute Direction
            d = -H @ g

            # 3. Line Search
            alpha = 1.0
            step_accepted = False
            f_new = None
            x_new = None
            
            for _ in range(20):  # Max 20 line search steps
                x_trial = x + alpha * d
                x_trial = np.clip(x_trial, -5, 5)
                
                # If no movement, break
                if np.allclose(x_trial, x):
                    break
                    
                f_trial = func(x_trial)
                calls += 1
                
                # Armijo condition: f(x + alpha*d) <= f(x) + c * alpha * g.T * d
                # c = 1e-4
                if f_trial <= f_x + 1e-4 * alpha * (g @ d):
                    step_accepted = True
                    f_new = f_trial
                    x_new = x_trial
                    break
                else:
                    alpha *= 0.5

            if not step_accepted:
                # If line search fails, shrink H and try a random step?
                # Or just break to avoid wasting budget
                # Let's try a small random step to escape local minima/stagnation
                x_trial = x + 0.1 * np.random.randn(self.dim)
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls += 1
                if f_trial < f_x:
                    x_new = x_trial
                    f_new = f_trial
                    step_accepted = True
                    # Reset H if we take a random step
                    H = np.eye(self.dim)
                else:
                    # Stagnation
                    break

            if not step_accepted:
                break

            # 4. Update Best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            # 5. BFGS Update
            # Need gradient at new point
            g_new = estimate_gradient(x_new, f_new)
            
            s = x_new - x
            y = g_new - g
            
            # Check if s and y are significant
            if np.linalg.norm(s) > 1e-10 and np.linalg.norm(y) > 1e-10:
                rho = 1.0 / (y @ s)
                # BFGS update for inverse Hessian
                # H_new = (I - rho * s * y.T) H (I - rho * y * s.T) + rho * s * s.T
                I = np.eye(self.dim)
                term1 = I - rho * np.outer(s, y)
                term2 = I - rho * np.outer(y, s)
                H = term1 @ H @ term2 + rho * np.outer(s, s)
                
                # Ensure H is symmetric and positive definite (roughly)
                H = (H + H.T) / 2
            else:
                # Reset H if update is invalid
                H = np.eye(self.dim)

            x = x_new
            f_x = f_new

        return best_x, best_f
# EVOLVE-BLOCK-END
