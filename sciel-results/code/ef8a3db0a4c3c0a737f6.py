"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead based optimizer for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # State tracking
        calls = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Initialization: Multi-start
        n_starts = 10
        initial_points = []
        for _ in range(n_starts):
            # Sample in [-5, 5]^n
            x0 = rng.uniform(-5, 5, size=n)
            fx0 = evaluate(x0)
            initial_points.append((fx0, x0))
        
        if calls >= budget:
            return best_x, best_f

        # Sort by function value
        initial_points.sort(key=lambda x: x[0])
        
        # Pick the best initial point for Nelder-Mead
        x_best_init = initial_points[0][1]
        
        # 2. Nelder-Mead Setup
        # Initial simplex
        alpha = 0.5 # Initial step size
        simplex = [x_best_init.copy()]
        for i in range(n):
            xi = x_best_init.copy()
            xi[i] += alpha
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            
        # Evaluate simplex vertices
        simplex_f = []
        for v in simplex:
            if np.all(v == x_best_init):
                simplex_f.append(initial_points[0][0])
            else:
                simplex_f.append(evaluate(v))
        
        if calls >= budget:
            return best_x, best_f

        # Nelder-Mead Parameters
        r = 1.0   # Reflection
        e = 2.0   # Expansion
        c = 0.5   # Contraction
        s = 0.5   # Shrinkage

        def nelder_mead_step(simplex, simplex_f):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return simplex, simplex_f, False
            
            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = simplex_f[-1]
            
            # Reflection
            x_r = centroid + r * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = evaluate(x_r)
            
            if f_r < simplex_f[0]:
                # Expansion
                if calls >= budget:
                    return simplex, simplex_f, False
                x_e = centroid + e * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = evaluate(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Reflection accepted
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if calls >= budget:
                    return simplex, simplex_f, False
                # Outside contraction
                x_c = centroid + c * (worst - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = evaluate(x_c)
                if f_c < worst_f:
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    if calls >= budget:
                        return simplex, simplex_f, False
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + s * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = evaluate(simplex[i])
            
            return simplex, simplex_f, True

        # Main Loop
        while calls < budget:
            simplex, simplex_f, continued = nelder_mead_step(simplex, simplex_f)
            if not continued:
                break
            
            # Check for convergence
            f_min = min(simplex_f)
            f_max = max(simplex_f)
            if f_max - f_min < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
