"""BBOB f14 (Different powers) optimizer.

Multi-start (1+1)-ES with best-first selection, adaptive step size,
restarts from global best, and coordinate-descent polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf

        def t_ok():
            return time.time() < deadline

        def ok():
            return t_ok() and calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()

        def call(x):
            nonlocal calls
            calls += 1
            return func(np.asarray(x, dtype=float))

        # Number of concurrent ES agents
        n_agents = 5

        # Latin Hypercube Sampling for diverse initial points
        lhs_points = []
        for d in range(dim):
            strata = np.linspace(-5, 5, n_agents + 1)
            for i in range(n_agents):
                lhs_points.append(rng.uniform(strata[i], strata[i + 1]))

        lhs_points = np.array(lhs_points).reshape(n_agents, dim)

        # Initialize agents
        agents = []
        for i in range(n_agents):
            x0 = lhs_points[i].copy()
            f0 = call(x0)
            record(x0, f0)

            agents.append({
                'x': x0,
                'f': f0,
                'sigma': 1.0,
                'last_improve': 0,
                'id': i
            })

        # Main loop: Best-First selection
        global_iter = 0
        stagnation_threshold = 30

        while ok():
            global_iter += 1

            # Select the agent with the best current value
            min_fs = [a['f'] for a in agents]
            best_agent_idx = int(np.argmin(min_fs))
            agent = agents[best_agent_idx]

            x = agent['x']
            f = agent['f']
            sigma = agent['sigma']

            # Check for stagnation
            if (global_iter - agent['last_improve']) > stagnation_threshold:
                # Restart this agent
                if best_x is not None and rng.random() < 0.5:
                    scale = max(0.1, 1.0 / (1 + global_iter * 0.02))
                    x_new = best_x + rng.normal(0, scale, dim)
                else:
                    x_new = rng.uniform(-5, 5, dim)
                x_new = np.clip(x_new, -5, 5)

                f_new = call(x_new)
                record(x_new, f_new)

                agent['x'] = x_new
                agent['f'] = f_new
                agent['sigma'] = 1.0
                agent['last_improve'] = global_iter
                continue

            # Generate perturbation
            perturbation = rng.normal(0, sigma, dim)
            x_try = x + perturbation
            x_try = np.clip(x_try, -5, 5)

            if not ok():
                break
            f_try = call(x_try)
            record(x_try, f_try)

            if f_try < f:
                # Accept
                agent['x'] = x_try
                agent['f'] = f_try
                agent['last_improve'] = global_iter
                # Adapt step size: increase slightly on success
                agent['sigma'] = min(agent['sigma'] * 1.05, 5.0)
            else:
                # Reject: decrease step size
                agent['sigma'] = max(agent['sigma'] * 0.95, 0.01)

        # Final polish phase: coordinate descent on the best point
        if best_x is not None and ok():
            x_polish = best_x.copy()
            f_polish = best_f
            step_size = 0.1

            for _ in range(15):
                if not ok():
                    break
                improved_cd = False
                for d in range(dim):
                    if not ok():
                        break
                    for direction in [1, -1]:
                        x_try = x_polish.copy()
                        x_try[d] += direction * step_size
                        x_try = np.clip(x_try, -5, 5)
                        f_try = call(x_try)
                        if f_try < f_polish - 1e-12:
                            x_polish = x_try
                            f_polish = f_try
                            record(x_polish, f_polish)
                            improved_cd = True
                            break
                    if improved_cd:
                        break
                if not improved_cd:
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
