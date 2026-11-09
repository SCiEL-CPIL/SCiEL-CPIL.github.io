"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: diverse sampling + (mu+lambda)-ES + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial diverse sampling to handle multimodality
        n_init = min(15, budget)
        population = []
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            population.append((fx, x))

        population.sort(key=lambda t: t[0])
        best_fx, best_x = population[0]
        calls_made = n_init

        # Phase 2: (mu+lambda)-ES with periodic restarts
        mu = 4
        lam = 12
        sigma = 1.5
        sigma_min = 1e-4
        sigma_max = 5.0
        restart_interval = 12
        gen = 0

        # Keep a list of good candidates for local refinement
        good_candidates = [(best_fx, best_x.copy())]

        while calls_made < budget:
            if time.time() > deadline:
                break

            gen += 1

            # Get parents (best mu)
            parents = population[:mu]

            # Generate offspring
            offspring = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                # Pick a parent based on rank
                pi = min(i // (lam // mu), mu - 1) if lam >= mu else 0
                parent_fx, parent_x = parents[pi]
                step = sigma * rng.standard_normal(dim)
                y = np.clip(parent_x + step, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))

            # Combine parents and offspring
            population = population + offspring
            population.sort(key=lambda t: t[0])
            population = population[:mu + lam]

            # Update best
            if population[0][0] < best_fx:
                best_fx, best_x = population[0]
                good_candidates.append((best_fx, best_x.copy()))

            # Track good candidates for local refinement
            if population[1][0] < best_fx + 0.1:
                good_candidates.append((population[1][0], population[1][1].copy()))

            # Adaptive sigma
            parent_best = parents[0][0]
            offspring_best = min(f for f, _ in offspring) if offspring else parent_best
            if offspring_best < parent_best:
                sigma *= 1.3
            else:
                sigma *= 0.85
            sigma = np.clip(sigma, sigma_min, sigma_max)

            # Periodic restarts to escape local minima
            if gen % restart_interval == 0 and calls_made < budget:
                n_restarts = min(3, budget - calls_made)
                for _ in range(n_restarts):
                    if calls_made >= budget:
                        break
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls_made += 1
                    population.append((fx, x))
                population.sort(key=lambda t: t[0])
                population = population[:mu + lam]
                sigma = 2.0

        # Phase 3: Nelder-Mead local refinement on best candidates
        # Use remaining budget for local search
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline:
            # Take the best few candidates and refine with Nelder-Mead
            unique_candidates = []
            seen = set()
            for fx, x in good_candidates:
                key = tuple(np.round(x, 4))
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append((fx, x))
            unique_candidates.sort(key=lambda t: t[0])
            
            # Refine top candidates
            n_refine = min(3, len(unique_candidates))
            for i in range(n_refine):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                fx0, x0 = unique_candidates[i]
                # Simple Nelder-Mead implementation
                x = x0.copy()
                f = fx0
                # Initial simplex
                simplex = [x.copy()]
                for j in range(dim):
                    xi = x.copy()
                    xi[j] += 0.1
                    simplex.append(xi)
                
                fvals = [f] + [func(np.clip(s, -5, 5)) for s in simplex[1:]]
                calls_made += dim
                
                # Nelder-Mead iterations
                max_nm_iter = min(20, (budget - calls_made) // (dim + 1))
                for _ in range(max_nm_iter):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    # Sort simplex
                    idx = np.argsort(fvals)
                    simplex = [simplex[i] for i in idx]
                    fvals = [fvals[i] for i in idx]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid + (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    fr = func(xr)
                    calls_made += 1
                    
                    if fvals[0] <= fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    elif fr < fvals[0]:
                        # Expansion
                        xe = centroid + 2 * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fe = func(xe)
                        calls_made += 1
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        # Contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        calls_made += 1
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for j in range(1, dim + 1):
                                simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                simplex[j] = np.clip(simplex[j], -5, 5)
                                fvals[j] = func(simplex[j])
                                calls_made += 1
                                if calls_made >= budget or time.time() > deadline:
                                    break
                            if calls_made >= budget or time.time() > deadline:
                                break
                    
                    # Update best
                    min_idx = np.argmin(fvals)
                    if fvals[min_idx] < best_fx:
                        best_fx = fvals[min_idx]
                        best_x = simplex[min_idx].copy()
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
