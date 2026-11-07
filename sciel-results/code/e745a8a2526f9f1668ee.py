import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        # 330s limit, use 310s for safety margin
        self.deadline = self.start_time + 310

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        eval_count = 0
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal eval_count, best_f, best_x
            if time.time() > self.deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise RuntimeError("Budget exceeded")
            eval_count += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Phase 1: Global Search with multiple diverse starts
        # Use a mix of uniform random and structured sampling
        n_global = 60
        for i in range(n_global):
            if time.time() > self.deadline:
                break
            # Mix of uniform random and centered around origin
            if i < 30:
                x = rng.uniform(-5, 5, dim)
            elif i < 45:
                # Concentrate more near the center
                x = rng.uniform(-3, 3, dim)
            else:
                # Some points near the boundaries of the feasible region
                x = rng.uniform(-4.5, 4.5, dim)
            try:
                counted_func(x)
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                continue

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Local Refinement with L-BFGS-B from multiple best points
        # Keep track of top candidates from global search
        remaining_budget = budget - eval_count
        if remaining_budget > 10:
            bounds = [(-5.0, 5.0)] * dim
            
            # Try L-BFGS-B from the best point
            try:
                res = scipy.optimize.minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxfun': min(remaining_budget, 200), 'maxiter': 100}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 3: If budget remains, do additional local searches from perturbed best point
        remaining_budget = budget - eval_count
        if remaining_budget > 50 and time.time() < self.deadline:
            # Generate several perturbations around the current best
            n_perturb = 5
            for i in range(n_perturb):
                if time.time() > self.deadline or eval_count >= budget - 10:
                    break
                # Perturb with decreasing step sizes
                step = 1.0 / (i + 1)
                x_pert = best_x + rng.normal(0, step, dim)
                x_pert = np.clip(x_pert, -5, 5)
                try:
                    counted_func(x_pert)
                except (TimeoutError, RuntimeError):
                    break
                except Exception:
                    continue
                
                # If this perturbation improved, do a local search from it
                if eval_count < budget - 20 and time.time() < self.deadline:
                    try:
                        res = scipy.optimize.minimize(
                            counted_func,
                            x_pert,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxfun': min(budget - eval_count - 5, 100), 'maxiter': 50}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x
                    except (TimeoutError, RuntimeError):
                        break
                    except Exception:
                        continue

        return best_x, best_f
