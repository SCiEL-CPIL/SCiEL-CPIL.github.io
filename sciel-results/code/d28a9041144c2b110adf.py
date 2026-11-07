"""Direct-structured optimizer for BBOB f20 (Schwefel) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Differential evolution with periodicity-aware initialization for BBOB f20 (Schwefel).
    Uses a coarse grid to detect phase shifts, then DE for global search,
    followed by Nelder-Mead and coordinate descent for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def elapsed():
            return time.time() - start_time

        def time_left():
            return deadline - time.time()

        def try_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            calls_made += 1
            return fx

        def nelder_mead(x0, max_calls):
            """Nelder-Mead simplex method from x0."""
            nonlocal best_f, best_x, calls_made
            n = dim
            x = x0.copy().astype(float)
            fx = try_func(x)
            if fx is None:
                return
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Initialize simplex
            simplex = [x.copy()]
            simplex_f = [fx]
            for i in range(n):
                xi = x.copy()
                step = 0.5 if abs(xi[i]) < 4.5 else -0.5
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fxi = try_func(xi)
                if fxi is None:
                    return
                simplex.append(xi)
                simplex_f.append(fxi)
                if fxi < best_f:
                    best_f = fxi
                    best_x = xi.copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            min_step = 1e-10
            calls_in_nm = n + 1

            for _ in range(max_calls):
                if calls_made >= budget or time.time() > deadline:
                    return
                if calls_in_nm >= max_calls:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = try_func(xr)
                if fxr is None:
                    return
                calls_in_nm += 1

                if fxr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = try_func(xe)
                    if fxe is None:
                        return
                    calls_in_nm += 1
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    # Contraction
                    if fxr < simplex_f[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - xr)
                    xc = np.clip(xc, -5, 5)
                    fxc = try_func(xc)
                    if fxc is None:
                        return
                    calls_in_nm += 1
                    if fxc < min(fxr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fsi = try_func(simplex[i])
                            if fsi is None:
                                return
                            calls_in_nm += 1
                            simplex_f[i] = fsi

                # Update best
                for i in range(n + 1):
                    if simplex_f[i] < best_f:
                        best_f = simplex_f[i]
                        best_x = simplex[i].copy()

                # Check simplex size
                spread = np.max([np.linalg.norm(simplex[0] - simplex[i]) for i in range(1, n + 1)])
                if spread < min_step:
                    break

        def coordinate_descent(x0, max_calls):
            """Coordinate descent with adaptive step."""
            nonlocal best_f, best_x, calls_made
            x = x0.copy().astype(float)
            fx = try_func(x)
            if fx is None:
                return
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            step = 0.5
            for _ in range(max_calls):
                if calls_made >= budget or time.time() > deadline:
                    return

                improved = False
                for i in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        return

                    for sign in [1.0, -1.0]:
                        y = x.copy()
                        y[i] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = try_func(y)
                        if fy is None:
                            return
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break

                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break

        # Phase 1: Coarse grid search to detect phase shifts
        # Schwefel has period 4*pi ≈ 12.566. The domain is [-5,5] = width 10.
        # We'll sample a grid to find promising regions.
        grid_points = []
        # Known Schwefel minimum: x* = 4.209687...
        # With hidden shift, the minimum could be anywhere in [-4,4]^5.
        # Let's do a stratified sampling approach.
        
        # First, evaluate some key points
        key_points = [
            np.full(dim, 4.20968),
            np.full(dim, -4.20968),
            np.zeros(dim),
            np.full(dim, 4.20968 - 2 * np.pi),  # shifted by one period
            np.full(dim, 4.20968 + 2 * np.pi),
            np.full(dim, 4.20968 - 4 * np.pi),
            np.full(dim, 4.20968 + 4 * np.pi),
        ]
        
        for p in key_points:
            p = np.clip(p, -5, 5)
            fx = try_func(p)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = p.copy()

        # Stratified random sampling for better coverage
        n_stratified = min(20, budget - calls_made)
        for _ in range(n_stratified):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.8, 4.8, dim)
            fx = try_func(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return None, np.inf

        # Phase 2: Differential Evolution
        if calls_made < budget and time.time() < deadline:
            pop_size = 12
            de_budget = min(budget - calls_made, 200)
            
            # Initialize population around best point and with diversity
            population = [best_x.copy()]
            for _ in range(pop_size - 1):
                # Mix of local and global exploration
                if rng.random() < 0.5:
                    # Local perturbation around best
                    x = best_x + rng.normal(0, 1.0, dim)
                else:
                    # Global random
                    x = rng.uniform(-4.8, 4.8, dim)
                x = np.clip(x, -5, 5)
                population.append(x)
            
            # Evaluate initial population
            pop_f = []
            for x in population:
                fx = try_func(x)
                if fx is None:
                    break
                pop_f.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if len(pop_f) < pop_size:
                # Re-evaluate missing
                while len(pop_f) < pop_size:
                    x = population[len(pop_f)]
                    fx = try_func(x)
                    if fx is None:
                        break
                    pop_f.append(fx)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

            # DE parameters
            F = 0.8  # scaling factor
            CR = 0.9  # crossover rate
            de_calls = calls_made - (budget - de_budget) if de_budget < budget else 0
            max_de_iter = de_budget // pop_size
            
            for gen in range(max_de_iter):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                for i in range(pop_size):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    # Select three distinct random indices
                    a, b, c = rng.choice(pop_size, size=3, replace=False)
                    while a == b or a == c or b == c:
                        a, b, c = rng.choice(pop_size, size=3, replace=False)
                    
                    # Differential evolution
                    v = population[a] + F * (population[b] - population[c])
                    v = np.clip(v, -5, 5)
                    
                    # Crossover
                    trial = population[i].copy()
                    j_rand = rng.integers(0, dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                    
                    trial = np.clip(trial, -5, 5)
                    f_trial = try_func(trial)
                    if f_trial is None:
                        break
                    
                    # Selection
                    if f_trial <= pop_f[i]:
                        population[i] = trial
                        pop_f[i] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
                
                # Check for stagnation
                if gen > 0 and gen % 5 == 0:
                    # If no improvement in recent generations, boost exploration
                    if best_f == np.min(pop_f):
                        # Replace worst individual with random
                        worst_idx = np.argmax(pop_f)
                        population[worst_idx] = rng.uniform(-4.8, 4.8, dim)
                        f_new = try_func(population[worst_idx])
                        if f_new is not None:
                            pop_f[worst_idx] = f_new
                            if f_new < best_f:
                                best_f = f_new
                                best_x = population[worst_idx].copy()

        # Phase 3: Nelder-Mead from best point
        if calls_made < budget and time.time() < deadline and best_x is not None:
            remaining = budget - calls_made
            nm_budget = min(remaining, 120)
            nelder_mead(best_x, nm_budget)

        # Phase 4: Coordinate descent polish
        if calls_made < budget and time.time() < deadline and best_x is not None:
            remaining = budget - calls_made
            cd_budget = min(remaining, 80)
            coordinate_descent(best_x, cd_budget)

        # Phase 5: Final local search from multiple nearby points
        if calls_made < budget and time.time() < deadline and best_x is not None:
            for step in [0.1, 0.05, 0.01]:
                if calls_made >= budget or time.time() > deadline:
                    break
                for i in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    for sign in [1.0, -1.0]:
                        if calls_made >= budget or time.time() > deadline:
                            break
                        y = best_x.copy()
                        y[i] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = try_func(y)
                        if fy is None:
                            break
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
