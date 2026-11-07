# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES family: diverse random restarts with adaptive step-size,
       plus a final high-precision polish pass. Strict wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(0)

        def remaining_time():
            return deadline - time.time()

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # --- Phase 1: Multi-restart (1+1)-ES portfolio ---
        # Each restart gets a sub-budget; we track the global best.
        best_x = None
        best_f = np.inf

        # Number of restarts: use most of the budget on a few good restarts.
        # Strategy: first restart gets a large chunk, subsequent ones smaller.
        # We adaptively decide when to stop based on time and improvement.
        n_restarts = 6
        # Allocate budget: first restart gets ~50%, rest split among others
        base_budget = budget // (n_restarts + 2)
        # Ensure at least a few evaluations per restart
        sub_budgets = []
        remaining = budget
        for i in range(n_restarts):
            if i == 0:
                b = min(remaining, max(80, budget // 2))
            else:
                b = min(remaining, max(20, base_budget))
            sub_budgets.append(b)
            remaining -= b
        # If we have budget left, give it to the last restart
        if remaining > 0 and sub_budgets:
            sub_budgets[-1] += remaining

        for r in range(n_restarts):
            if remaining_time() < 10.0:  # need time for polish
                break
            sb = sub_budgets[r]
            # Start point: random for first, near best for later (with jitter)
            if r == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                if best_x is not None:
                    # Start near the best found so far with some spread
                    spread = 3.0 / (r + 1)
                    x = clip(best_x + spread * rng.standard_normal(dim))
                else:
                    x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0 / (r + 1)  # smaller sigma for later restarts
            success = 0
            total = 0
            for _ in range(sb):
                if remaining_time() < 5.0:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                total += 1
                if fy < fx:
                    x, fx = y, fy
                    success += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                # Adaptive sigma control (1/5 rule approximation)
                if total >= 10:
                    rate = success / total
                    if rate > 0.35:
                        sigma *= 1.1
                    elif rate < 0.15:
                        sigma *= 0.9
                    success = 0
                    total = 0
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 10.0)

        # --- Phase 2: Final high-precision polish pass ---
        if best_x is not None and remaining_time() > 5.0:
            # Use a very small sigma and many iterations for fine-tuning
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            polish_budget = min(budget, max(50, int(remaining_time() * 50)))
            for _ in range(polish_budget):
                if remaining_time() < 1.0:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
