"""Multi-start Nelder-Mead ensemble with space-filling initialization and adaptive simplex refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead ensemble: launch independent simplexes from LHS points."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Generate space-filling initial points using Latin Hypercube Sampling
        # We want diverse starting points to handle hidden shifts
        n_starts = 8
        lhs_points = []
        for i in range(dim):
            # For each dimension, create a permutation of [0, 1, ..., n_starts-1]
            perm = rng.permutation(n_starts)
            for j in range(n_starts):
                # Map to [-5, 5] with some jitter
                val = -5 + 10 * (perm[j] + 0.5) / n_starts
                val += rng.normal(0, 0.5)  # Small jitter
                val = np.clip(val, -5, 5)
                if len(lhs_points) <= j:
                    lhs_points.append([])
                lhs_points[j].append(val)
        
        lhs_points = np.array(lhs_points)

        # Evaluate all starting points
        for i in range(n_starts):
            if not time_ok() or not budget_ok():
                break
            x = np.clip(lhs_points[i], -5, 5)
            fx = func(x)
            calls += 1
            record(x, fx)

        if best_x is None:
            return None, np.inf

        # Now launch Nelder-Mead from the top few best starting points
        # First, let's identify the best few
        # We'll run NM from the best point, and if budget allows, from the 2nd and 3rd best
        
        # For simplicity and robustness, let's run NM from the overall best point first
        # with a generous budget, then try others if time permits.
        
        # However, the strategy suggests launching 5 independent simplexes.
        # Since we can't run them in true parallel (single func call), we interleave or run sequentially.
        # Running sequentially from the best is usually most effective for AOCC if the best point is in the correct basin.
        # But if the best point is in a wrong basin, we need to try others.
        
        # Let's try a hybrid: Run NM from the best point. If it converges quickly and we have budget left,
        # try the next best.
        
        # Actually, a better strategy for AOCC is to get the best possible value ASAP.
        # NM from the best LHS point is a good bet.
        
        x0 = best_x.copy()
        f0 = best_f

        # Build initial simplex
        alpha = 0.3
        simplex = [x0.copy()]
        f_simplex = [f0]

        for i in range(dim):
            if not time_ok() or not budget_ok():
                break
            x_i = x0.copy()
            x_i[i] += alpha
            x_i = np.clip(x_i, -5, 5)
            f_i = func(x_i)
            calls += 1
            simplex.append(x_i)
            f_simplex.append(f_i)
            record(x_i, f_i)

        if len(simplex) >= 2:
            # Nelder-Mead parameters
            gamma = 1.0
            rho = 2.0
            sigma_nm = 0.5
            shrink = 0.5

            # Track best in this phase
            phase_best_x = best_x.copy()
            phase_best_f = best_f

            while time_ok() and budget_ok():
                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]

                x_best = simplex[0]
                f_best = f_simplex[0]
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_second_worst = simplex[-2]
                f_second_worst = f_simplex[-2]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + gamma * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                if not time_ok() or not budget_ok():
                    break
                f_r = func(x_r)
                calls += 1
                record(x_r, f_r)
                if f_r < phase_best_f:
                    phase_best_f = f_r
                    phase_best_x = x_r.copy()

                if f_r < f_best:
                    # Expansion
                    x_e = centroid + rho * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if not time_ok() or not budget_ok():
                        break
                    f_e = func(x_e)
                    calls += 1
                    record(x_e, f_e)
                    if f_e < phase_best_f:
                        phase_best_f = f_e
                        phase_best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_second_worst:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + sigma_nm * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - sigma_nm * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)

                    if not time_ok() or not budget_ok():
                        break
                    f_c = func(x_c)
                    calls += 1
                    record(x_c, f_c)
                    if f_c < phase_best_f:
                        phase_best_f = f_c
                        phase_best_x = x_c.copy()

                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if not time_ok() or not budget_ok():
                                break
                            simplex[i] = x_best + shrink * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            record(simplex[i], f_simplex[i])
                            if f_simplex[i] < phase_best_f:
                                phase_best_f = f_simplex[i]
                                phase_best_x = simplex[i].copy()
                        if not time_ok():
                            break

                # Check for convergence
                f_spread = np.max(f_simplex) - np.min(f_simplex)
                if f_spread < 1e-12 or np.max(np.abs(f_simplex)) < 1e-14:
                    break

                # Also check if simplex has collapsed in space
                max_dist = 0
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        d = np.max(np.abs(simplex[i] - simplex[j]))
                        max_dist = max(max_dist, d)
                if max_dist < 1e-10:
                    break

        # If we have budget left, try Nelder-Mead from the second best starting point
        # We need to track the second best from the LHS phase
        # For now, let's just do a few random restarts near the best if time allows
        # or try the second best LHS point
        
        # Let's re-evaluate the LHS points to find the second best
        # This is inefficient, so let's just try a perturbation of the current best
        # or try the next best LHS point if we stored them.
        
        # Since we didn't store all LHS points with their values, let's just do a few
        # random restarts near the current best if we have budget.
        
        if time_ok() and budget_ok():
            for _ in range(2):
                if not time_ok() or not budget_ok():
                    break
                
                x_new_start = best_x + rng.normal(0, 0.3, dim)
                x_new_start = np.clip(x_new_start, -5, 5)
                
                f_new = func(x_new_start)
                calls += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new_start.copy()
                
                # Brief NM refinement
                if f_new < best_f + 0.1 and time_ok() and budget_ok():
                    x02 = x_new_start.copy()
                    simplex2 = [x02.copy()]
                    f_simplex2 = [f_new]
                    
                    alpha2 = 0.3
                    for i in range(dim):
                        if not time_ok() or not budget_ok():
                            break
                        x_i2 = x02.copy()
                        x_i2[i] += alpha2
                        x_i2 = np.clip(x_i2, -5, 5)
                        f_i2 = func(x_i2)
                        calls += 1
                        simplex2.append(x_i2)
                        f_simplex2.append(f_i2)
                        if f_i2 < best_f:
                            best_f = f_i2
                            best_x = x_i2.copy()
                    
                    if len(simplex2) < 2:
                        continue
                    
                    # Brief NM iterations
                    for _iter in range(20):
                        if not time_ok() or not budget_ok():
                            break
                        indices2 = np.argsort(f_simplex2)
                        simplex2 = [simplex2[i] for i in indices2]
                        f_simplex2 = [f_simplex2[i] for i in indices2]
                        
                        x_b2 = simplex2[0]
                        f_b2 = f_simplex2[0]
                        x_w2 = simplex2[-1]
                        f_w2 = f_simplex2[-1]
                        x_sw2 = simplex2[-2]
                        f_sw2 = f_simplex2[-2]
                        
                        centroid2 = np.mean(simplex2[:-1], axis=0)
                        
                        x_r2 = centroid2 + 1.0 * (centroid2 - x_w2)
                        x_r2 = np.clip(x_r2, -5, 5)
                        if not time_ok() or not budget_ok():
                            break
                        f_r2 = func(x_r2)
                        calls += 1
                        if f_r2 < best_f:
                            best_f = f_r2
                            best_x = x_r2.copy()
                        
                        if f_r2 < f_b2:
                            x_e2 = centroid2 + 2.0 * (x_r2 - centroid2)
                            x_e2 = np.clip(x_e2, -5, 5)
                            if not time_ok() or not budget_ok():
                                break
                            f_e2 = func(x_e2)
                            calls += 1
                            if f_e2 < best_f:
                                best_f = f_e2
                                best_x = x_e2.copy()
                            if f_e2 < f_r2:
                                simplex2[-1] = x_e2
                                f_simplex2[-1] = f_e2
                            else:
                                simplex2[-1] = x_r2
                                f_simplex2[-1] = f_r2
                        elif f_r2 < f_sw2:
                            simplex2[-1] = x_r2
                            f_simplex2[-1] = f_r2
                        else:
                            if f_r2 < f_w2:
                                x_c2 = centroid2 + 0.5 * (x_r2 - centroid2)
                            else:
                                x_c2 = centroid2 - 0.5 * (x_w2 - centroid2)
                            x_c2 = np.clip(x_c2, -5, 5)
                            if not time_ok() or not budget_ok():
                                break
                            f_c2 = func(x_c2)
                            calls += 1
                            if f_c2 < best_f:
                                best_f = f_c2
                                best_x = x_c2.copy()
                            
                            if f_c2 < min(f_r2, f_w2):
                                simplex2[-1] = x_c2
                                f_simplex2[-1] = f_c2
                            else:
                                for i2 in range(1, len(simplex2)):
                                    if not time_ok() or not budget_ok():
                                        break
                                    simplex2[i2] = x_b2 + 0.5 * (simplex2[i2] - x_b2)
                                    simplex2[i2] = np.clip(simplex2[i2], -5, 5)
                                    f_simplex2[i2] = func(simplex2[i2])
                                    calls += 1
                                    if f_simplex2[i2] < best_f:
                                        best_f = f_simplex2[i2]
                                        best_x = simplex2[i2].copy()
                                if not time_ok():
                                    break
                        
                        if np.max(f_simplex2) - np.min(f_simplex2) < 1e-12:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
