"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                return 1e10
            if calls >= budget:
                return 1e10
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = float(func(x_clipped))
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start Nelder-Mead
        # Use larger initial simplex for faster convergence
        delta = 2.5

        # Generate multiple starting points
        n_starts = 8
        for s in range(n_starts):
            if not time_ok() or calls >= budget - 50:
                break

            # Start points: mix of random and structured
            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = np.ones(dim) * 2.5
            elif s == 2:
                x0 = -np.ones(dim) * 2.5
            elif s == 3:
                x0 = np.ones(dim) * -2.5
            else:
                x0 = rng.uniform(-4, 4, dim)

            x0 = np.clip(x0, -5, 5)
            f0 = wrapped_func(x0)

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # If we've already found a very good point, skip NM
            if best_f < -80:
                break

            # Nelder-Mead with limited function evaluations
            nm_budget = min(budget - calls - 50, 120)
            if nm_budget < 10:
                break

            # Create initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                point = x0.copy()
                step = delta if abs(point[i]) < 0.5 else delta * 0.5
                point[i] += step
                point = np.clip(point, -5, 5)
                simplex.append(point)

            # Evaluate simplex
            fvals = []
            for pt in simplex:
                f = wrapped_func(pt)
                fvals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = pt.copy()

            fvals = np.array(fvals)

            # Nelder-Mead iterations
            alpha, gamma, rho = 1.0, 2.0, 0.5
            for _ in range(min(nm_budget, 100)):
                if not time_ok() or calls >= budget - 50:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = wrapped_func(xr)

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = wrapped_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = wrapped_func(xc)
                    if fc < max(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f = wrapped_func(simplex[i])
                            fvals[i] = f

                        # Check for convergence
                        spread = np.max(fvals) - np.min(fvals)
                        if spread < 1e-8:
                            break

        # Phase 2: CMA-ES-inspired (1+1)-ES with momentum and adaptive sigma
        # This combines the refinement machinery from inspiration 1
        if time_ok() and calls < budget - 10:
            sigma = 0.8
            x = best_x.copy()
            fx = best_f
            velocity = np.zeros(dim)
            momentum = 0.6
            sigma_min = 1e-4
            sigma_max = 3.0
            
            # Track success rate for adaptive sigma
            success_window = []
            window_size = 15
            
            for _ in range(budget - calls - 5):
                if not time_ok():
                    break
                if calls >= budget:
                    break
                
                noise = rng.standard_normal(dim)
                
                # Blend random exploration with momentum direction
                if np.linalg.norm(velocity) > 1e-8:
                    dir_unit = velocity / np.linalg.norm(velocity)
                    noise = 0.75 * noise + 0.25 * dir_unit * np.linalg.norm(noise)
                
                y = x + sigma * noise
                y = np.clip(y, -5, 5)
                fy = wrapped_func(y)
                
                if fy < fx:
                    # Improvement
                    velocity = momentum * velocity + (1 - momentum) * (y - x)
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.15, sigma_max)
                    success_window.append(1)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # No improvement
                    velocity = momentum * velocity
                    sigma = max(sigma * 0.92, sigma_min)
                    success_window.append(0)
                
                # Periodically adjust sigma based on success rate
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window)
                    if success_rate > 0.7:
                        sigma = min(sigma * 1.1, sigma_max)
                    elif success_rate < 0.3:
                        sigma = max(sigma * 0.85, sigma_min)
                    success_window = []
                
                # Occasionally try a larger jump to escape local optima
                if _ % 40 == 39 and sigma < 2.0:
                    if calls < budget - 1:
                        big_noise = rng.standard_normal(dim) * 1.5
                        big_candidate = np.clip(x + big_noise, -5, 5)
                        big_fy = wrapped_func(big_candidate)
                        if big_fy < fx:
                            x = big_candidate
                            fx = big_fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            sigma = min(sigma * 1.3, sigma_max)
                            velocity = momentum * velocity + (1 - momentum) * (x - best_x)
                
                if sigma < 1e-5:
                    break

        # Phase 3: Fine-tune with very small steps
        if time_ok() and calls < budget:
            sigma = 0.05
            x = best_x.copy()
            fx = best_f
            velocity = np.zeros(dim)
            momentum = 0.5
            
            for _ in range(min(80, budget - calls)):
                if not time_ok():
                    break
                if calls >= budget:
                    break
                
                noise = rng.standard_normal(dim)
                if np.linalg.norm(velocity) > 1e-8:
                    dir_unit = velocity / np.linalg.norm(velocity)
                    noise = 0.8 * noise + 0.2 * dir_unit * np.linalg.norm(noise)
                
                y = x + sigma * noise
                y = np.clip(y, -5, 5)
                fy = wrapped_func(y)
                
                if fy < fx:
                    velocity = momentum * velocity + (1 - momentum) * (y - x)
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.1, 0.2)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    velocity = momentum * velocity
                    sigma = max(sigma * 0.95, 1e-4)
                
                if sigma < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
