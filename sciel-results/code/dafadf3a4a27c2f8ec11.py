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

        def nelder_mead(x0, step_size=0.5, max_iter=100):
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

        def coordinate_descent(x_start, max_iter=100):
            """Polish x_start using coordinate descent."""
            nonlocal calls, best_x, best_f, deadline
            x = x_start.copy()
            f = eval_func(x)
            step = 0.1
            for _ in range(max_iter):
                if calls >= self.budget or time.time() > deadline:
                    break
                improved = False
                for i in range(self.dim):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    # Try positive
                    x_pos = x.copy()
                    x_pos[i] += step
                    x_pos[i] = np.clip(x_pos[i], -self.bound, self.bound)
                    if not np.all(x_pos == x):
                        f_pos = eval_func(x_pos)
                        if f_pos < f:
                            x = x_pos
                            f = f_pos
                            improved = True
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                            continue
                    # Try negative
                    x_neg = x.copy()
                    x_neg[i] -= step
                    x_neg[i] = np.clip(x_neg[i], -self.bound, self.bound)
                    if not np.all(x_neg == x):
                        f_neg = eval_func(x_neg)
                        if f_neg < f:
                            x = x_neg
                            f = f_neg
                            improved = True
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                            continue
                if not improved:
                    step /= 2.0
                    if step < 1e-5:
                        break
            return x, f

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
            nelder_mead(best_x.copy(), step_size=1.0, max_iter=100)
            
        # Phase 3: Polish the best point using Coordinate Descent
        if calls < self.budget and time.time() < deadline:
            coordinate_descent(best_x.copy(), max_iter=100)
            
        # Phase 4: If budget remains, try a new diverse start
        if calls < self.budget and time.time() < deadline:
            x_new = rng.uniform(-4.0, 4.0, self.dim)
            eval_func(x_new)
            if calls < self.budget:
                nelder_mead(x_new, step_size=1.0, max_iter=50)
                if calls < self.budget:
                    coordinate_descent(best_x.copy(), max_iter=50)

        return best_x, best_f
# EVOLVE-BLOCK-END
