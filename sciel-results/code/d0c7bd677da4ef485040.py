"""Portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: LHS exploration + (1+1)-ES exploitation with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 320.0  # 10s safety margin before 330s kill

        # ---------- Phase 1: Latin Hypercube exploration ----------
        n_lhs = 60
        rng = np.random.default_rng(12345)
        perm = [rng.permutation(n_lhs) for _ in range(dim)]
        lxs = np.array([(perm[d] + rng.random()) / n_lhs for d in range(dim)]).T
        lxs = lxs * 10.0 - 5.0  # map to [-5,5]
        lxs = np.clip(lxs, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        calls = 0
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            fx = func(lxs[i])
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = lxs[i].copy()

        # ---------- Phase 2: (1+1)-ES with periodic restarts ----------
        # 1/5-success adaptation, restart every 100 steps from best-so-far
        step = 0
        total_es = budget - calls
        restart_interval = 100
        sigma = 1.2
        x = best_x.copy()
        fx = best_f

        # Track best across restarts
        global_best_x = best_x.copy()
        global_best_f = best_f

        while calls < budget and step < total_es:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = func(y)
            calls += 1
            step += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                sigma = min(sigma, 4.0)
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()
            else:
                sigma *= 0.82
                sigma = max(sigma, 0.05)

            # Periodic restart: escape local minima
            if step % restart_interval == 0:
                # Restart from global best with a fresh sigma, or a random point
                if rng.random() < 0.75:
                    x = global_best_x.copy() + 0.8 * rng.standard_normal(dim)
                    x = np.clip(x, -5.0, 5.0)
                    fx = func(x)
                    calls += 1
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                else:
                    x = rng.uniform(-5.0, 5.0, dim)
                    fx = func(x)
                    calls += 1
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                sigma = 1.2

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
