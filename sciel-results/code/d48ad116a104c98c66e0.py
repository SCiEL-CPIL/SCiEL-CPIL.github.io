"""Multi-objective optimizer for DTLZ7 exploiting known front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ7 structure: front determined by (x0, x1); drive g->0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ7 specific:
        # f1 = x0
        # f2 = (1-x0)*cos(x1*pi/2)
        # f3 = (1-x0)*sin(x1*pi/2)
        # g = 120 + 10*n + sum_{i=2}^{n}(x_i - 5*sin(2*pi*x_i))^2
        # F2 = (1+g)*f2, F3 = (1+g)*f3
        #
        # To minimize all objectives, we need:
        # 1. g -> 0 (set x_i to minimize g for i>=2)
        # 2. x0, x1 to spread along the Pareto front

        # The minimum of (x - 5*sin(2*pi*x))^2 for x in [0,1] is 0 at x=0, x=0.5, x=1
        # So set x_i = 0 for all i >= 2 to get g = 120 + 10*20 = 320? 
        # Wait: g = 120 + 10*n + sum...
        # Actually for DTLZ7, g = 120 + 10*n + sum_{i=2}^{n}(x_i - 5*sin(2*pi*x_i))^2
        # The minimum of (x - 5*sin(2*pi*x))^2 on [0,1] is 0 (at x=0, 0.5, 1)
        # So min g = 120 + 10*n where n = dim-2 = 20, so min g = 120 + 200 = 320?
        # Hmm, that seems high. Let me reconsider.
        # Actually in standard DTLZ7: g = 1 + 10*(n-2) + sum_{i=3}^{n}(x_i - 5*sin(2*pi*x_i))^2
        # with n being the number of variables. For 22 vars, n-2=20, so g_min = 1+200=201.
        # But the exact formula depends on the implementation. Let's just use the problem() function.

        # Strategy:
        # 1. First, find a good g-minimizing configuration for x[2:]
        # 2. Then explore (x0, x1) to cover the Pareto front

        # Phase 1: Minimize g by optimizing x[2:]
        # Use a simple approach: try several configurations

        # Known good values: x_i = 0 or 0.5 or 1 minimize the individual terms
        # Let's start with all zeros for x[2:]
        x_base = np.zeros(dim)
        f_base = problem(x_base)
        used = 1

        # Try to improve g by perturbing x[2:]
        # Actually, let's use a smarter approach:
        # We'll do a two-phase search:
        # Phase 1: Find low-g points (optimize x[2:])
        # Phase 2: For each low-g point, explore (x0, x1)

        # Let's use a population-based approach with emphasis on the structure

        # Initialize population
        pop_size = 40
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))

        # First few: try to minimize g
        for i in range(min(10, pop_size)):
            x = np.zeros(dim)
            # Random perturbation of x[2:]
            x[2:] = np.random.uniform(0, 1, dim-2)
            X[i] = x
            F[i] = problem(x)
            used += 1

        # Rest: random initialization
        for i in range(10, pop_size):
            if used >= budget:
                break
            x = np.random.uniform(lo, hi, dim)
            X[i] = x
            F[i] = problem(x)
            used += 1

        # Evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Find non-dominated points
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break

            non_dom_mask = ~dominated
            if np.any(non_dom_mask):
                parents = X[non_dom_mask]
                F_parents = F[non_dom_mask]
            else:
                parents = X
                F_parents = F

            # Select parent
            pi = np.random.randint(len(parents))
            parent = parents[pi].copy()

            # Mutation: focus on x[2:] to reduce g, and x[0], x[1] for diversity
            # With high probability, mutate x[2:] toward 0 (known good)
            if np.random.rand() < 0.7:
                # Mutate x[2:] toward 0
                parent[2:] = parent[2:] * 0.8 + 0.2 * np.random.uniform(0, 1, dim-2)
            else:
                # Random mutation
                parent += 0.1 * (hi - lo) * np.random.standard_normal(dim)

            # Also mutate x[0] and x[1] for front coverage
            if np.random.rand() < 0.5:
                parent[0] = np.random.uniform(0, 1)
            if np.random.rand() < 0.5:
                parent[1] = np.random.uniform(0, 1)

            parent = np.clip(parent, lo, hi)

            fc = problem(parent)
            used += 1

            # Replace worst point
            # Use a simple criterion: maximize sum of objectives
            worst = np.argmax(np.sum(F, axis=1))
            X[worst] = parent
            F[worst] = fc

            # Occasionally add a new random point for diversity
            if used < budget and np.random.rand() < 0.1:
                if time.time() > deadline:
                    break
                x_new = np.random.uniform(lo, hi, dim)
                # But bias x[2:] toward 0
                if np.random.rand() < 0.5:
                    x_new[2:] *= 0.5
                f_new = problem(x_new)
                used += 1
                worst2 = np.argmax(np.sum(F, axis=1))
                X[worst2] = x_new
                F[worst2] = f_new

        return X
# EVOLVE-BLOCK-END
