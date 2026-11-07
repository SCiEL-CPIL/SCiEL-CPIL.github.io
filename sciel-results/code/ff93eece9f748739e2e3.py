"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with Nelder-Mead polish and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return deadline - time.time() > 1.5

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Variance-effectiveness population size
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        # Step size control
        cs = (mueff + 1.5) / (dim + mueff + 5)
        # Covariance matrix adaptation
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) / (1 - cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial state
        mean = np.zeros(dim)
        sigma = 2.0  # Start with a reasonable spread
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Run CMA-ES loop until budget is exhausted or time runs out
        while calls < budget and time_left():
            # Generate population
            population = []
            for i in range(lam):
                if calls + lam > budget:
                    # Run (1+1)-ES for remaining budget
                    while calls < budget and time_left():
                        y = np.clip(mean + sigma * rng.standard_normal(dim), -5, 5)
                        fy = func(y)
                        calls += 1
                        update_best(y, fy)
                        if fy < best_f:
                            mean = y.copy()
                            sigma *= 1.05
                        else:
                            sigma *= 0.95
                    break
                # Sample from N(mean, sigma^2 * C)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                
                z = rng.standard_normal(dim)
                y = mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                update_best(y, fy)
                population.append((fy, y))

            if calls >= budget:
                break

            # Sort population by fitness
            population.sort(key=lambda x: x[0])
            
            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * population[i][1]
            
            # Update step size control vectors
            norm_L = np.linalg.norm(L)
            if norm_L > 1e-16:
                y_old = (new_mean - mean) / (sigma * norm_L)
            else:
                y_old = np.zeros(dim)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_old
            # hsig condition using standard approximation
            E_norm = np.sqrt(dim) * (1 - 1.0/(4*dim) + 3.0/(32*dim*dim))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls / (cs * (dim + 1))))) < (1.4 + 2 / (dim + 1)) * E_norm
            
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * y_old
            
            # Update covariance matrix
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                if norm_L > 1e-16:
                    z_i = (population[i][1] - mean) / (sigma * norm_L)
                else:
                    z_i = np.zeros(dim)
                rank_mu += weights[i] * np.outer(z_i, z_i)
            
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * rank_mu
            C = (C + C.T) / 2.0
            
            # Update sigma
            norm_ps = np.linalg.norm(ps)
            sigma = sigma * np.exp(cs * (norm_ps / E_norm - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            mean = new_mean
            
            if sigma < 1e-10:
                break

        # Nelder-Mead polish pass using remaining budget
        if calls < budget and time_left():
            # Start from best_x found so far
            nm_x = best_x.copy()
            nm_f = best_f
            # Initial simplex: slightly perturb best_x
            step = 0.1
            simplex = [nm_x.copy()]
            for i in range(dim):
                x_i = nm_x.copy()
                x_i[i] += step
                x_i = np.clip(x_i, -5, 5)
                simplex.append(x_i)
            
            # Evaluate initial simplex
            simplex_f = []
            for x in simplex:
                if calls >= budget or not time_left():
                    break
                f = func(x)
                calls += 1
                simplex_f.append(f)
                update_best(x, f)
            
            if len(simplex_f) == dim + 1:
                # Nelder-Mead iterations
                max_nm_iter = 100
                for _ in range(max_nm_iter):
                    if calls >= budget or not time_left():
                        break
                    
                    # Sort simplex by fitness
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    worst_x = simplex[-1]
                    worst_f = simplex_f[-1]
                    
                    # Reflection
                    alpha = 1.0
                    xr = centroid + alpha * (centroid - worst_x)
                    xr = np.clip(xr, -5, 5)
                    if calls >= budget or not time_left():
                        break
                    fr = func(xr)
                    calls += 1
                    update_best(xr, fr)
                    
                    if fr < simplex_f[0]:
                        # Expansion
                        gamma = 2.0
                        xe = centroid + gamma * (centroid - worst_x)
                        xe = np.clip(xe, -5, 5)
                        if calls >= budget or not time_left():
                            break
                        fe = func(xe)
                        calls += 1
                        update_best(xe, fe)
                        
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    elif fr < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    else:
                        # Contraction
                        rho = 0.5
                        xc = centroid + rho * (worst_x - centroid)
                        xc = np.clip(xc, -5, 5)
                        if calls >= budget or not time_left():
                            break
                        fc = func(xc)
                        calls += 1
                        update_best(xc, fc)
                        
                        if fc < worst_f:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                if calls >= budget or not time_left():
                                    break
                                fi = func(simplex[i])
                                calls += 1
                                simplex_f[i] = fi
                                update_best(simplex[i], fi)
                            if calls >= budget or not time_left():
                                break
                            # Re-sort
                            order = np.argsort(simplex_f)
                            simplex = [simplex[i] for i in order]
                            simplex_f = [simplex_f[i] for i in order]
                    
                    # Check for convergence
                    if max(simplex_f) - min(simplex_f) < 1e-12:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
