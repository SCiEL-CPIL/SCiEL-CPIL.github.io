"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        calls = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            x = np.clip(x, -self.bound, self.bound)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, step_size=0.5, max_iter=200):
            """Run Nelder-Mead from x0 with given initial step size."""
            nonlocal calls, best_x, best_f, deadline
            
            n = self.dim
            # Build initial simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += step_size
            
            # Evaluate initial simplex
            simplex_f = np.array([eval_func(simplex[i]) for i in range(n + 1)])
            
            if calls >= self.budget:
                return
            
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            it = 0
            while calls < self.budget and it < max_iter:
                if time.time() > deadline:
                    break
                it += 1
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-12:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:n], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = eval_func(x_r)
                if calls >= self.budget:
                    break
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = eval_func(x_e)
                    if calls >= self.budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    f_c = eval_func(x_c)
                    if calls >= self.budget:
                        break
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = eval_func(simplex[i])
                        if calls >= self.budget:
                            break
        
        # Phase 1: Broad exploration with diverse starting points
        rng = np.random.default_rng(42)
        
        # Start with origin and several random points
        init_points = [np.zeros(self.dim)]
        for _ in range(9):
            init_points.append(rng.uniform(-4.0, 4.0, self.dim))
        
        for p in init_points:
            eval_func(p)
        
        # Phase 2: Run Nelder-Mead from the best point found
        if calls < self.budget:
            nelder_mead(best_x.copy(), step_size=1.0, max_iter=150)
        
        # Phase 3: If budget remains, try additional starts from promising regions
        if calls < self.budget and time.time() < deadline:
            # Generate a few more starts near the current best
            for _ in range(3):
                if calls >= self.budget or time.time() > deadline:
                    break
                perturbation = rng.normal(0, 0.5, self.dim)
                x_start = np.clip(best_x + perturbation, -self.bound, self.bound)
                eval_func(x_start)
                if calls < self.budget:
                    nelder_mead(x_start, step_size=0.3, max_iter=50)
        
        # Phase 4: Final fine-tuning from the global best
        if calls < self.budget and time.time() < deadline:
            nelder_mead(best_x.copy(), step_size=0.1, max_iter=80)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
