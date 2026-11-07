"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer for Discus-like functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        best_x = None
        best_f = float('inf')
        call_count = 0
        rng = np.random.default_rng(42)

        # Helper to evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Nelder-Mead implementation for polishing
        def nelder_mead(start_x, max_iter=30):
            nonlocal call_count, deadline
            dim = self.dim
            simplex = [start_x.copy()]
            f_simplex = [evaluate(start_x)]

            for i in range(dim):
                x_i = start_x.copy()
                step = 0.5 if start_x[i] < 4.0 else -0.5
                x_i[i] += step
                simplex.append(x_i)
                f_simplex.append(evaluate(x_i))

            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            it = 0
            while it < max_iter:
                if time.time() > deadline or call_count >= budget:
                    break

                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                centroid = np.sum(simplex[:-1], axis=0) / dim
                x_r = centroid + alpha * (centroid - x_worst)
                f_r = evaluate(x_r)

                if f_r < f_simplex[0]:
                    x_e = centroid + gamma * (centroid - x_worst)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1], f_simplex[-1] = x_e, f_e
                    else:
                        simplex[-1], f_simplex[-1] = x_r, f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1], f_simplex[-1] = x_r, f_r
                else:
                    if f_r < f_worst:
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = evaluate(x_c)
                        if f_c < f_r:
                            simplex[-1], f_simplex[-1] = x_c, f_c
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                f_simplex[i] = evaluate(simplex[i])
                            order = np.argsort(f_simplex)
                            simplex = [simplex[i] for i in order]
                            f_simplex = [f_simplex[i] for i in order]
                            break
                    else:
                        x_c = centroid - rho * (centroid - x_worst)
                        f_c = evaluate(x_c)
                        if f_c < f_worst:
                            simplex[-1], f_simplex[-1] = x_c, f_c
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                f_simplex[i] = evaluate(simplex[i])
                            order = np.argsort(f_simplex)
                            simplex = [simplex[i] for i in order]
                            f_simplex = [f_simplex[i] for i in order]
                            break

                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
                    break
                it += 1
            return simplex[0], f_simplex[0]

        # Multi-start Nelder-Mead: generate diverse starting points
        # Start with origin and several random points in the box
        start_points = [np.zeros(dim)]
        
        # Add random points spread across the domain
        for _ in range(15):
            pt = rng.uniform(-4, 4, size=dim)
            start_points.append(pt)
        
        # Also add some points near the boundaries
        for _ in range(5):
            pt = rng.uniform(-5, 5, size=dim)
            # Push one coordinate toward boundary
            idx = rng.integers(0, dim)
            if pt[idx] < 0:
                pt[idx] = -5 + rng.uniform(0, 1)
            else:
                pt[idx] = 5 - rng.uniform(0, 1)
            start_points.append(pt)

        # Sort start points by proximity to origin (prioritize center)
        start_points.sort(key=lambda p: np.linalg.norm(p))

        # Run Nelder-Mead from each start point, tracking best
        for sp in start_points:
            if call_count >= budget - 10 or time.time() > deadline:
                break
            
            # Reserve enough budget for a full NM run (dim+1 initial + ~2*dim iterations)
            est_cost = (dim + 1) + 2 * dim * 3  # conservative estimate
            if call_count + est_cost > budget:
                # Use fewer iterations if budget is tight
                remaining = budget - call_count
                if remaining < dim + 1:
                    break
                max_iter = max(1, (remaining - dim - 1) // (2 * dim))
            else:
                max_iter = 30
            
            nelder_mead(sp.copy(), max_iter=max_iter)

        # Final polish on the best point found
        if best_x is not None and call_count < budget - 5 and time.time() < deadline:
            nelder_mead(best_x.copy(), max_iter=10)

        return best_x, best_f

# EVOLVE-BLOCK-END
