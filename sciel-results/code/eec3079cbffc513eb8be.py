"""Lattice-aware DE + custom Nelder-Mead + coordinate descent for BBOB f4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-aware DE for global search + custom NM + coordinate descent for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s wall-clock limit
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def custom_nelder_mead(x0, max_calls):
            """Custom Nelder-Mead with strict call budget."""
            nonlocal calls_made, x_best, f_best
            if x0 is None or max_calls < 8 or time.time() > deadline:
                return

            alpha, beta, gamma = 1.0, 0.5, 2.0
            # Initialize simplex
            simplex = [np.array(x0, dtype=np.float64)]
            f0 = evaluate(simplex[0])
            if calls_made >= budget:
                return

            for i in range(dim):
                xi = simplex[0].copy()
                step = 0.05 if xi[i] != 0 else 0.05
                xi[i] += step
                simplex.append(xi)
                if calls_made >= budget:
                    return
                evaluate(simplex[i + 1])

            if len(simplex) < dim + 1:
                return

            # Main NM loop
            for _ in range(max_calls):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Sort simplex
                order = np.argsort([evaluate(s) if s is None else f for s, f in
                                    [(simplex[i], None) for i in range(len(simplex))]])
                # Re-evaluate to get current values (but we already have them from evaluate)
                # Actually, let's track values properly
                vals = []
                for i in range(len(simplex)):
                    # We need the function value; re-evaluate would waste calls
                    # Instead, let's store values alongside
                    pass

                # Better approach: store (x, f) pairs
                break  # This won't work well; let me restructure

        def nm_refine(x_center, max_calls):
            """Nelder-Mead refinement with proper value tracking."""
            nonlocal calls_made, x_best, f_best
            if x_center is None or max_calls < 8 or time.time() > deadline:
                return

            alpha, beta, gamma = 1.0, 0.5, 2.0
            x0 = np.array(x_center, dtype=np.float64)
            f0 = evaluate(x0)
            if calls_made >= budget:
                return

            # Build simplex
            simplex = [(x0.copy(), f0)]
            for i in range(dim):
                xi = x0.copy()
                step = 0.05 if xi[i] != 0 else 0.05
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = evaluate(xi)
                if calls_made >= budget:
                    return
                simplex.append((xi, fi))

            n = len(simplex)  # dim + 1
            for _ in range(max_calls):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                simplex.sort(key=lambda t: t[1])

                # Centroid of all but worst
                centroid = np.mean([simplex[i][0] for i in range(n - 1)], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1][0])
                xr = np.clip(xr, -5, 5)
                fr = evaluate(xr)
                if calls_made >= budget:
                    break

                if fr < simplex[0][1]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = evaluate(xe)
                    if calls_made >= budget:
                        break
                    if fe < fr:
                        simplex[-1] = (xe, fe)
                    else:
                        simplex[-1] = (xr, fr)
                elif fr < simplex[-2][1]:
                    simplex[-1] = (xr, fr)
                else:
                    # Contraction
                    if fr < simplex[-1][1]:
                        # Outside contraction
                        xc = centroid + beta * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - beta * (simplex[-1][0] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = evaluate(xc)
                    if calls_made >= budget:
                        break
                    if fc < min(fr, simplex[-1][1]):
                        simplex[-1] = (xc, fc)
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i][0] = simplex[0][0] + 0.5 * (simplex[i][0] - simplex[0][0])
                            simplex[i][1] = evaluate(simplex[i][0])
                            if calls_made >= budget:
                                break
                        if calls_made >= budget:
                            break

                # Update global best
                for x, f in simplex:
                    if f < f_best:
                        f_best = f
                        x_best = x.copy()

                # Check convergence
                if n > 1:
                    spread = np.max([np.linalg.norm(simplex[i][0] - simplex[0][0]) for i in range(1, n)])
                    if spread < 1e-10:
                        break

        def coordinate_descent(x_center, max_calls):
            """Multi-scale coordinate descent for deterministic refinement."""
            nonlocal calls_made, x_best, f_best
            if x_center is None or max_calls < 4 or time.time() > deadline:
                return

            x = np.array(x_center, dtype=np.float64)
            f = f_best  # Assume x_center is the current best

            scales = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            for scale in scales:
                if calls_made >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and calls_made < budget and time.time() < deadline:
                    improved = False
                    for i in range(dim):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[i] += direction * scale
                            x_trial = np.clip(x_trial, -5, 5)
                            ft = evaluate(x_trial)
                            if calls_made >= budget:
                                break
                            if ft < f - 1e-12:
                                f = ft
                                x = x_trial
                                improved = True
                                if f < f_best:
                                    f_best = f
                                    x_best = x.copy()
                        if calls_made >= budget:
                            break

        def lattice_init(max_calls):
            """Probe integer lattice points to find the correct basin for f4."""
            nonlocal calls_made, x_best, f_best
            if calls_made >= max_calls or time.time() > deadline:
                return

            rng = np.random.default_rng(42)
            lattice_budget = min(max_calls, int(budget * 0.1))

            # Evaluate random integer points in [-4, 4]^5
            for _ in range(lattice_budget):
                if calls_made >= budget or time.time() > deadline:
                    break
                x_int = rng.integers(-4, 5, size=dim).astype(np.float64)
                evaluate(x_int)

            # Quick refine on best found
            if x_best is not None and calls_made < budget - 10:
                nm_refine(x_best.copy(), min(10, budget - calls_made))

        def de_search(max_calls, pop_size=12, F_mut=0.8, CR=0.5):
            """Differential evolution for global exploration."""
            nonlocal calls_made, x_best, f_best
            rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
            pop_size = min(pop_size, max(8, max_calls // 15))

            pop = []
            # Seed with best known point
            if x_best is not None:
                pop.append((x_best.copy(), f_best))

            # Fill rest with random points
            while len(pop) < pop_size:
                if calls_made >= max_calls or time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                f = evaluate(x)
                pop.append((x, f))

            if not pop:
                return

            pop.sort(key=lambda item: item[1])

            while calls_made < max_calls and time.time() < deadline:
                x_best_pop = pop[0][0]
                new_pop = []

                for i in range(pop_size):
                    if calls_made >= max_calls or time.time() > deadline:
                        new_pop.append(pop[i])
                        continue

                    indices = [j for j in range(pop_size) if j != i]
                    if len(indices) < 3:
                        new_pop.append(pop[i])
                        continue
                    r1, r2, r3 = rng.choice(indices, 3, replace=False)

                    # Mutation: best/1/bin
                    v = x_best_pop + F_mut * (pop[r1][0] - pop[r2][0])

                    # Crossover
                    j_rand = rng.integers(0, dim)
                    trial = np.zeros(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop[i][0][j]

                    trial = np.clip(trial, -5, 5)
                    f_trial = evaluate(trial)

                    if f_trial <= pop[i][1]:
                        new_pop.append((trial, f_trial))
                    else:
                        new_pop.append(pop[i])

                pop = new_pop
                pop.sort(key=lambda item: item[1])

            if pop and pop[0][1] < f_best:
                f_best = pop[0][1]
                x_best = pop[0][0].copy()

        # === Main Strategy ===

        # Phase 1: Lattice Search (10% budget) - find correct basin
        lattice_calls = int(budget * 0.1)
        if time.time() < deadline:
            lattice_init(lattice_calls)

        # Phase 2: DE Global Search (50% budget)
        de_calls = int(budget * 0.5)
        if time.time() < deadline:
            de_search(de_calls, pop_size=12)

        # Phase 3: Coordinate Descent (15% budget) - deterministic multi-scale
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            cd_calls = min(remaining, int(budget * 0.15))
            coordinate_descent(x_best.copy(), cd_calls)

        # Phase 4: Nelder-Mead Refinement (15% budget) - converge to minimum
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            nm_calls = min(remaining, int(budget * 0.15))
            nm_refine(x_best.copy(), nm_calls)

        # Phase 5: Final Polish - alternate CD and NM with remaining budget
        remaining = budget - calls_made
        while remaining >= 10 and x_best is not None and time.time() < deadline:
            # Short coordinate descent
            cd_calls = min(remaining // 2, 15)
            if cd_calls >= 4:
                coordinate_descent(x_best.copy(), cd_calls)
                remaining = budget - calls_made
                if remaining < 10:
                    break

            # Short NM
            nm_calls = min(remaining // 2, 15)
            if nm_calls >= 8:
                nm_refine(x_best.copy(), nm_calls)
                remaining = budget - calls_made

        # Final safety
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
