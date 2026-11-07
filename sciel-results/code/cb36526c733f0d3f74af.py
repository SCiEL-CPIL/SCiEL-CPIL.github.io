"""Multi-start Random Sampling + Nelder-Mead local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start search with Nelder-Mead refinement."""

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

        # Phase 1: Multi-start Random Sampling
        # Use 20% of the budget for global exploration by sampling random points
        sampling_budget = int(budget * 0.2)
        candidates = []

        # Always include origin as a candidate
        x_origin = np.zeros(dim)
        f_origin = func(x_origin)
        calls += 1
        record(x_origin, f_origin)
        candidates.append((f_origin, x_origin.copy()))

        # Sample additional random points
        for _ in range(sampling_budget - 1):
            if not time_ok() or not budget_ok():
                break
            x_rand = rng.uniform(-5, 5, dim)
            f_rand = func(x_rand)
            calls += 1
            record(x_rand, f_rand)
            candidates.append((f_rand, x_rand.copy()))

        # Sort candidates by function value and pick the top K
        candidates.sort(key=lambda item: item[0])
        K = 3  # Number of local refinements to run
        top_candidates = candidates[:K]

        # Phase 2: Nelder-Mead refinement from top K points
        remaining_budget = budget - calls
        budget_per_nm = max(10, int(remaining_budget / K))

        for f_init, x_init in top_candidates:
            if not time_ok() or not budget_ok():
                break
            
            # Limit this specific NM run to budget_per_nm or until time/budget runs out
            nm_start_calls = calls
            nm_time_start = time.time()
            
            x0 = x_init.copy()
            f0 = f_init

            # Build initial simplex
            # Use a small step size for local refinement
            alpha = 0.1 
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

                # Track best in this local phase
                phase_best_x = x0.copy()
                phase_best_f = f0

                while time_ok() and budget_ok() and (calls - nm_start_calls) < budget_per_nm:
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

        return best_x, best_f
# EVOLVE-BLOCK-END
