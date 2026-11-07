"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid PSO + Nelder-Mead for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_val = np.inf
        self.best_x = None
        self.start_time = None
        self.deadline = None

    def _call_func(self, func, x):
        self.call_count += 1
        val = func(x)
        if val < self.best_val:
            self.best_val = val
            self.best_x = np.array(x, copy=True)
        return val

    def _nelder_mead(self, func, x0, max_calls):
        n = self.dim
        simplex = [np.array(x0, copy=True)]
        for i in range(n):
            xi = np.array(x0, copy=True)
            step = 0.1 if abs(xi[i]) < 0.1 else 0.05 * abs(xi[i])
            xi[i] += step
            simplex.append(np.clip(xi, -5, 5))

        values = []
        for sx in simplex:
            if self.call_count >= max_calls:
                return
            values.append(self._call_func(func, sx))

        if len(values) < n + 1:
            return

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        while self.call_count < max_calls:
            order = np.argsort(values)
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)

            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = self._call_func(func, xr)

            if fr < values[0]:
                if self.call_count >= max_calls:
                    simplex[-1] = xr
                    values[-1] = fr
                    break
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                fe = self._call_func(func, xe)
                if fe < fr:
                    simplex[-1] = xe
                    values[-1] = fe
                else:
                    simplex[-1] = xr
                    values[-1] = fr
            elif fr < values[-2]:
                simplex[-1] = xr
                values[-1] = fr
            else:
                if self.call_count >= max_calls:
                    break
                xc = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                fc = self._call_func(func, xc)
                if fc < values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    for i in range(1, n + 1):
                        if self.call_count >= max_calls:
                            return
                        simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                        values[i] = self._call_func(func, simplex[i])

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 280.0
        self.call_count = 0
        self.best_val = np.inf
        self.best_x = None

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Phase 1: PSO global search
        swarm_size = 10
        pso_iterations = (budget - 50) // swarm_size  # reserve 50 for local opt
        pso_iterations = max(10, min(pso_iterations, 45))

        # Initialize swarm
        positions = rng.uniform(-5, 5, (swarm_size, dim))
        velocities = rng.uniform(-1, 1, (swarm_size, dim))
        personal_best = [np.array(positions[i], copy=True) for i in range(swarm_size)]
        personal_best_val = [np.inf] * swarm_size
        global_best = None
        global_best_val = np.inf

        # Evaluate initial positions
        for i in range(swarm_size):
            if self.call_count >= budget or time.time() > self.deadline:
                break
            val = self._call_func(func, positions[i])
            personal_best_val[i] = val
            personal_best[i] = np.array(positions[i], copy=True)
            if val < global_best_val:
                global_best_val = val
                global_best = np.array(positions[i], copy=True)

        # PSO loop
        w = 0.7
        c1 = 1.5
        c2 = 1.5

        for it in range(pso_iterations):
            if self.call_count >= budget or time.time() > self.deadline:
                break
            for i in range(swarm_size):
                if self.call_count >= budget or time.time() > self.deadline:
                    break
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best[i] - positions[i]) +
                                 c2 * r2 * (global_best - positions[i]))
                velocities[i] = np.clip(velocities[i], -2, 2)
                positions[i] = np.clip(positions[i] + velocities[i], -5, 5)
                val = self._call_func(func, positions[i])
                if val < personal_best_val[i]:
                    personal_best_val[i] = val
                    personal_best[i] = np.array(positions[i], copy=True)
                if val < global_best_val:
                    global_best_val = val
                    global_best = np.array(positions[i], copy=True)

        # Phase 2: Local optimization from best point
        remaining = budget - self.call_count
        if remaining > 10 and self.best_x is not None and time.time() < self.deadline:
            self._nelder_mead(func, self.best_x, min(self.call_count + remaining, budget))

        return self.best_x, self.best_val
# EVOLVE-BLOCK-END
