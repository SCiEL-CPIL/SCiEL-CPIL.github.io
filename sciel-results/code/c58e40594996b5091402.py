import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # --- Phase 1: Differential Evolution ---
        N = 10
        if budget < N + 20: # If budget is too small, use (1+1)
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            sigma = 1.0
            for _ in range(budget - 1):
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
            return x, fx

        # Initialize population
        pop = rng.uniform(-5, 5, (N, dim))
        fits = np.array([func(p) for p in pop])
        calls = N
        
        F = 0.8
        CR = 0.9
        
        # Leave 20% budget for local search
        local_budget = max(10, budget // 5)
        de_limit = budget - local_budget
        
        while calls < de_limit:
            # Sort to keep track of best, though DE doesn't strictly need it
            # But it helps if we want to use a variant like jDE
            # Standard DE:
            for i in range(N):
                if calls >= de_limit: break
                
                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(N, 3, replace=False)
                while i in [r1, r2, r3]:
                    r1, r2, r3 = rng.choice(N, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)
                
                # Crossover
                j_rand = rng.integers(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                # Evaluation
                ft = func(trial)
                calls += 1
                
                # Selection
                if ft < fits[i]:
                    pop[i] = trial
                    fits[i] = ft
                    
        # --- Phase 2: Nelder-Mead Local Search ---
        best_idx = np.argmin(fits)
        x = pop[best_idx].copy()
        fx = fits[best_idx]
        
        # Initialize Simplex
        # Start with best point and small perturbations
        simplex = [x.copy()]
        for i in range(dim):
            p = x.copy()
            # Small step in dimension i
            step = 0.1
            p[i] += step
            p = np.clip(p, -5, 5)
            simplex.append(p)
            
        # Evaluate initial simplex
        simplex_fits = [fx]
        for p in simplex[1:]:
            fp = func(p)
            calls += 1
            simplex_fits.append(fp)
            
        # Nelder-Mead Loop
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma_nm = 0.5 # Shrink
        
        while calls < budget:
            # Sort simplex
            idx = np.argsort(simplex_fits)
            simplex = [simplex[i] for i in idx]
            simplex_fits = [simplex_fits[i] for i in idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = simplex_fits[-1]
            
            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            
            if fr < simplex_fits[0]:
                # Try expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    simplex_fits[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_fits[-1] = fr
            elif fr < simplex_fits[-2]:
                # Accept reflection
                simplex[-1] = xr
                simplex_fits[-1] = fr
            else:
                # Contraction
                # Outside contraction
                xc = centroid + rho * (centroid - worst)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                
                if fc < worst_f:
                    simplex[-1] = xc
                    simplex_fits[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        p = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        p = np.clip(p, -5, 5)
                        fp = func(p)
                        calls += 1
                        simplex[i] = p
                        simplex_fits[i] = fp
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

            # Update best
            if simplex_fits[0] < fx:
                fx = simplex_fits[0]
                x = simplex[0].copy()

        return x, fx
