import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = float(fx)
                best_x = x.copy()
            return float(fx)

        # Phase 1: Quick random sampling to find a good starting region
        n_random = min(25, budget // 10)
        for _ in range(n_random):
            if eval_count >= budget - 30 or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, d)
            fx = evaluate(x)

        # Phase 2: Nelder-Mead simplex method for robust convergence
        # Initialize simplex around best_x
        step_size = 0.5
        simplex = [best_x.copy()]
        for i in range(d):
            v = best_x.copy()
            v[i] += step_size
            simplex.append(v)
        
        f_values = [evaluate(v) for v in simplex]
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma_nm = 0.5  # shrink
        
        max_nm_iter = 150
        for _iter in range(max_nm_iter):
            if eval_count >= budget - 20 or time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            # Check convergence
            if f_values[-1] - f_values[0] < 1e-10 * (1 + abs(f_values[0])):
                break
            
            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = f_values[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = evaluate(x_r)
            
            if f_r < f_values[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = evaluate(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = evaluate(simplex[i])
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - worst)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = evaluate(simplex[i])
        
        # Phase 3: Coordinate-wise refinement with adaptive steps
        refine_x = best_x.copy()
        refine_f = best_f
        for iteration in range(8):
            if eval_count >= budget - 5 or time.time() > deadline:
                break
            improved = False
            for i in range(d):
                if eval_count >= budget - 5 or time.time() > deadline:
                    break
                # Try decreasing step sizes in both directions
                for step in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002,
                            -0.1, -0.05, -0.02, -0.01, -0.005, -0.002]:
                    if eval_count >= budget - 5 or time.time() > deadline:
                        break
                    test_x = refine_x.copy()
                    test_x[i] += step
                    test_x = np.clip(test_x, -5, 5)
                    fx = evaluate(test_x)
                    if fx < refine_f:
                        refine_f = fx
                        refine_x = test_x.copy()
                        best_f = fx
                        best_x = refine_x.copy()
                        improved = True
                        break
            if not improved:
                break
        
        return best_x, best_f
