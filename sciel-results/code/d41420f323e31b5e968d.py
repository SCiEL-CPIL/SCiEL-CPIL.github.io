"""Multi-restart (1+1)-ES with LHS exploration and success-rate sigma adaptation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with LHS exploration and success-rate sigma adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ---- Phase 1: LHS + random diverse exploration to locate a good basin ----
        explore_budget = max(10, min(40, budget // 10))
        n_lhs = min(explore_budget, 20)
        for i in range(n_lhs):
            x = np.zeros(dim)
            for d in range(dim):
                lo = -5.0 + d * 10.0 / dim
                hi = lo + 10.0 / dim
                x[d] = lo + rng.random() * (hi - lo)
            eval_point(x)
        for _ in range(explore_budget - n_lhs):
            x = rng.uniform(-5.0, 5.0, dim)
            eval_point(x)

        if best_x is None:
            eval_point(np.zeros(dim))

        # ---- Phase 2: Multi-restart (1+1)-ES with success-rate sigma adaptation ----
        # Build diverse restart points: best found, plus random and perturbed points.
        restart_points = [best_x.copy()]
        for _ in range(4):
            restart_points.append(rng.uniform(-4.0, 4.0, dim))
        for _ in range(2):
            restart_points.append(np.clip(best_x + rng.normal(0, 1.5, dim), -5.0, 5.0))
        restart_points = restart_points[:8]

        # Allocate budget: first restart gets the majority so it can converge fast
        # (early progress is rewarded by AOCC); later restarts are shorter fallbacks.
        n_restarts = len(restart_points)
        alloc = []
        remaining = budget - call_count
        for i in range(n_restarts):
            if i == 0:
                alloc.append(remaining)
            else:
                alloc.append(max(5, remaining // (2 * (n_restarts - i))))
        # Ensure total does not exceed budget
        total_alloc = sum(alloc)
        if total_alloc > remaining:
            scale = remaining / total_alloc
            alloc = [int(a * scale) for a in alloc]
            alloc[-1] = remaining - sum(alloc[:-1])

        for r, start_pt in enumerate(restart_points):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            x = start_pt.copy()
            if r == 0:
                fx = best_f
            else:
                fx = eval_point(x)

            sigma = 1.5
            successes = 0
            total_steps = 0
            window_size = 10

            for _ in range(alloc[r]):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_point(y)
                total_steps += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.3
                else:
                    sigma *= 0.85

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Success-rate based sigma adaptation (stable, windowed)
                if total_steps % window_size == 0:
                    success_rate = successes / window_size
                    if success_rate > 0.5:
                        sigma *= 1.2
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    successes = 0
                sigma = np.clip(sigma, 0.01, 5.0)

        # ---- Phase 3: Fine polish around the best point with small steps ----
        if call_count < budget and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            successes = 0
            total_steps = 0
            window_size = 8
            for _ in range(budget - call_count):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_point(y)
                total_steps += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                if total_steps % window_size == 0:
                    success_rate = successes / window_size
                    if success_rate > 0.5:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.8
                    successes = 0
                sigma = np.clip(sigma, 1e-4, 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
