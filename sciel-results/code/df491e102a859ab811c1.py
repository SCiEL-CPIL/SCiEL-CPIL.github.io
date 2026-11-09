"""Multi-start Nelder-Mead with dynamic budget allocation for BBOB f20."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility within an instance
        rng = np.random.default_rng(42)

        # Multi-start: 10 independent Nelder-Mead runs
        n_starts = 10

        # Initialize diverse starting points
        # Mix of random points and some structured points
        starts = []
        for i in range(n_starts):
            if i == 0:
                # Center
                x0 = np.zeros(dim)
            elif i == 1:
                # Random uniform
                x0 = rng.uniform(-5, 5, dim)
            elif i == 2:
                # Random uniform, different range
                x0 = rng.uniform(-4, 4, dim)
            elif i == 3:
                # Sparse random
                x0 = np.zeros(dim)
                idx = rng.choice(dim, size=2, replace=False)
                x0[idx] = rng.uniform(-5, 5, 2)
            elif i == 4:
                # Another random
                x0 = rng.uniform(-5, 5, dim)
            elif i == 5:
                # Random with one coordinate at boundary
                x0 = rng.uniform(-5, 5, dim)
                x0[0] = 5.0
            elif i == 6:
                # Random with one coordinate at boundary
                x0 = rng.uniform(-5, 5, dim)
                x0[1] = -5.0
            elif i == 7:
                # Small perturbations around center
                x0 = rng.uniform(-1, 1, dim)
            elif i == 8:
                # Medium perturbations
                x0 = rng.uniform(-2.5, 2.5, dim)
            else:
                # Fully random
                x0 = rng.uniform(-5, 5, dim)
            starts.append(x0)

        # Each Nelder-Mead run maintains its own simplex
        # Simplex has dim+1 points
        # We'll track: simplex points, function values, current best
        # Budget allocation: start with equal budget, reallocate based on progress

        # Initialize simplexes
        simplexes = []
        for x0 in starts:
            simplex = [x0.copy()]
            for j in range(dim):
                pt = x0.copy()
                # Perturb one coordinate
                step = 0.5
                if pt[j] + step > 5:
                    pt[j] = pt[j] - step
                elif pt[j] - step < -5:
                    pt[j] = pt[j] + step
                else:
                    pt[j] = pt[j] + step
                simplex.append(pt)
            simplexes.append(simplex)

        # Evaluate initial simplexes
        # Each simplex has dim+1 points, total = n_starts * (dim+1)
        # For dim=5, that's 10 * 6 = 60 evaluations initially

        fvals = []  # fvals[i][j] = func value of simplexes[i][j]
        best_x = None
        best_f = np.inf
        calls_made = 0

        for i in range(n_starts):
            fvals.append([])
            for j in range(dim + 1):
                x = np.clip(simplexes[i][j], -5, 5)
                f = func(x)
                fvals[i].append(f)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        # Check deadline
        if time.time() > deadline:
            return best_x, best_f

        # Dynamic budget allocation
        # Track convergence rate for each start
        # Allocate remaining budget based on recent improvement

        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # We'll run Nelder-Mead iterations, allocating budget dynamically
        # Each iteration of Nelder-Mead can use up to 4 function evaluations
        # (reflection, expansion, contraction, shrinkage)

        # For simplicity, we'll do rounds: in each round, give each active start
        # a chance to do one Nelder-Mead step, prioritizing those with best progress

        # Track progress: best value per start, and how many evaluations used
        start_best_f = [min(fvals[i]) for i in range(n_starts)]
        start_calls = [dim + 1] * n_starts  # initial simplex evaluations
        active = [True] * n_starts

        # We'll process starts in order of their current best, giving more
        # budget to better starts
        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sort active starts by their best function value
            active_indices = [i for i in range(n_starts) if active[i]]
            if not active_indices:
                break

            # Prioritize starts with lower (better) best_f
            active_indices.sort(key=lambda i: start_best_f[i])

            # In this round, give the top few starts a chance
            # Let's give the best 3 starts one Nelder-Mead step each
            for i in active_indices[:3]:
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Perform one Nelder-Mead step for start i
                simplex = simplexes[i]
                fv = fvals[i]

                # Sort simplex by function value
                order = np.argsort(fv)
                simplex = [simplex[k] for k in order]
                fv = [fv[k] for k in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = fv[-1]

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_made += 1
                start_calls[i] += 1

                if f_ref < fv[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - x_worst)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_made += 1
                    start_calls[i] += 1

                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fv[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fv[-1] = f_ref
                elif f_ref < fv[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fv[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                        x_con = np.clip(x_con, -5, 5)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_worst - centroid)
                        x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_made += 1
                    start_calls[i] += 1

                    if f_con < min(f_ref, f_worst):
                        simplex[-1] = x_con
                        fv[-1] = f_con
                    else:
                        # Shrinkage
                        for k in range(1, dim + 1):
                            simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                            simplex[k] = np.clip(simplex[k], -5, 5)
                            f_shrink = func(simplex[k])
                            calls_made += 1
                            start_calls[i] += 1
                            fv[k] = f_shrink
                        # Re-sort after shrinkage
                        order = np.argsort(fv)
                        simplex = [simplex[k] for k in order]
                        fv = [fv[k] for k in order]

                # Update start's simplex and values
                simplexes[i] = simplex
                fvals[i] = fv

                # Update best tracking
                start_best_f[i] = min(fv)
                if start_best_f[i] < best_f:
                    best_f = start_best_f[i]
                    # Find the point with best value
                    best_idx = np.argmin(fv)
                    best_x = simplex[best_idx].copy()

                # Check if this start is converging (small spread)
                spread = max(fv) - min(fv)
                if spread < 1e-6:
                    active[i] = False

        return best_x, best_f
