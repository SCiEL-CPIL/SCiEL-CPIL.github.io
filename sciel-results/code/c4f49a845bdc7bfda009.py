import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.eval_count = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.start_time = time.time()
        self.eval_count = 0
        self.best_x = None
        self.best_f = np.inf

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Safety margin: return well before 330s kill
        max_time = 310.0

        # --- Phase 1: broad random sampling to find a good starting basin ---
        # Spend ~10% of budget on diverse random sampling
        n_sample = max(10, budget // 10)
        for i in range(n_sample):
            if time.time() - self.start_time > max_time:
                break
            if self.eval_count >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            self.eval_count += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()

        if self.eval_count >= budget or time.time() - self.start_time > max_time:
            return self.best_x, self.best_f

        # --- Phase 2: Simulated Annealing from the best point found ---
        x = self.best_x.copy()
        fx = self.best_f

        # Temperature schedule: start high enough to cross Rastrigin ridges,
        # cool geometrically. Rastrigin has local minima ~4 per dimension,
        # so typical gap ~ 2*dim*4 = 40 for dim=5. Start T ~ 50.
        T0 = 50.0
        T_min = 1e-6
        # Number of SA iterations we expect to run
        remaining = budget - self.eval_count
        # Geometric cooling: T_k = T0 * alpha^k, reach T_min at k_max
        # alpha = (T_min / T0) ** (1 / k_max)
        k_max = max(1, remaining)
        alpha = (T_min / T0) ** (1.0 / k_max)
        T = T0

        # Step size: start moderately large, shrink as T cools
        base_step = 2.0

        for step in range(remaining):
            if time.time() - self.start_time > max_time:
                break
            if self.eval_count >= budget:
                break

            # Adaptive step: proportional to sqrt(T) to balance exploration/exploitation
            step_size = base_step * np.sqrt(T / T0) + 0.05
            # Add a small chance of a large jump to escape deep local minima
            if rng.random() < 0.02:
                step_size *= 3.0

            delta = rng.standard_normal(dim) * step_size
            x_new = np.clip(x + delta, -5, 5)
            fx_new = func(x_new)
            self.eval_count += 1

            df = fx_new - fx
            if df < 0:
                # Always accept improvements
                x = x_new
                fx = fx_new
                if fx < self.best_f:
                    self.best_f = fx
                    self.best_x = x.copy()
            else:
                # Metropolis acceptance: accept worse with prob exp(-df/T)
                if T > T_min:
                    # Avoid overflow in exp
                    if df / T < 700:
                        if rng.random() < np.exp(-df / T):
                            x = x_new
                            fx = fx_new
                # Otherwise reject

            # Geometric cooling
            T *= alpha
            if T < T_min:
                T = T_min

        return self.best_x, self.best_f
