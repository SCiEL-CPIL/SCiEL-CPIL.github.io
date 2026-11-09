"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart CMA-ES + Nelder-Mead for f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def elapsed():
            return time.time() - start_time

        def remaining():
            return budget - calls_made

        def time_ok(margin=5.0):
            return elapsed() < deadline - margin

        # Phase 1: Stratified + random initial sampling
        n_strata = 3
        points_per_dim = np.linspace(-4, 4, n_strata)
        
        import itertools
        all_combos = list(itertools.product(points_per_dim, repeat=dim))
        rng.shuffle(all_combos)
        
        n_initial = min(30, budget // 10)
        
        for i in range(n_initial):
            if calls_made >= budget or not time_ok():
                break
            if i < len(all_combos):
                x = np.array(all_combos[i], dtype=float)
            else:
                x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Add random points for diversity
        for i in range(n_initial // 2):
            if calls_made >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: CMA-ES from the best starting point
        def run_cmaes(x0, sigma0, max_calls, deadline_check):
            """Run CMA-ES with call limit and deadline check."""
            nonlocal best_f, best_x, calls_made
            
            x = x0.copy()
            sigma = sigma0
            c1 = 1 / (dim + 3)
            cmu = 2 / ((dim + 2) ** 2 + 5)
            cc = (4 + dim / 4) / (dim + 4 + 2 / dim)
            cs = (dim + 1.7) / (dim + 9)
            damps = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            lambda_ = 4 + int(np.floor(3 * np.log(dim)))
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)
            
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            local_best_f = float('inf')
            local_best_x = x0.copy()
            
            for gen in range(200):
                if calls_made >= max_calls or not deadline_check():
                    break
                
                offspring = []
                try:
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                
                for i in range(lambda_):
                    if calls_made >= max_calls or not deadline_check():
                        break
                    z = rng.standard_normal(dim)
                    y = L @ z
                    xi = x + sigma * y
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls_made += 1
                    offspring.append((fi, xi))
                
                if len(offspring) == 0:
                    break
                    
                offspring.sort(key=lambda t: t[0])
                
                for fi, xi in offspring:
                    if fi < local_best_f:
                        local_best_f = fi
                        local_best_x = xi.copy()
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                
                mu_actual = min(mu, len(offspring))
                selected = [offspring[i][1] for i in range(mu_actual)]
                
                new_x = np.zeros(dim)
                for i in range(mu_actual):
                    new_x += weights[i] * selected[i]
                
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (C @ (new_x - x))
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (dim + 1)) * chi_n
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50)
                
                v = (new_x - x) / sigma
                C = (1 - c1 - cmu) * C + c1 * (v @ v.T + (1 - hsig) * cc * (2 - cc) * pc @ pc.T) + \
                    cmu * sum(weights[i] * np.outer(selected[i] - x, (selected[i] - x)) / sigma ** 2 for i in range(mu_actual))
                
                pc = (1 - cc) * pc + cc * hsig * v
                
                x = new_x
                
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                if sigma < 1e-8:
                    break

            return local_best_f, local_best_x

        def run_nm(x0, max_calls, deadline_check):
            """Run Nelder-Mead with adaptive step size."""
            nonlocal best_f, best_x, calls_made
            
            if calls_made >= max_calls or not deadline_check():
                return best_f, best_x
            
            # Adaptive initial step size based on best_f magnitude
            init_step = 0.5
            if best_f > 10:
                init_step = 1.0
            elif best_f > 1:
                init_step = 0.5
            else:
                init_step = 0.2
            
            # Create initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += init_step
                simplex.append(xi)
            
            # Evaluate simplex
            for i, xi in enumerate(simplex):
                if calls_made >= max_calls or not deadline_check():
                    break
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_made += 1
                simplex[i] = (fi, xi)
            
            if calls_made >= max_calls or not deadline_check():
                return best_f, best_x
            
            # Sort simplex
            simplex.sort(key=lambda t: t[0])
            
            # Track best from NM
            nm_best_f = simplex[0][0]
            nm_best_x = simplex[0][1].copy()
            
            if nm_best_f < best_f:
                best_f = nm_best_f
                best_x = nm_best_x.copy()
            
            # Nelder-Mead iterations
            alpha, gamma, rho = 1.0, 2.0, 0.5
            max_iters = 50
            
            for _ in range(max_iters):
                if calls_made >= max_calls or not deadline_check():
                    break
                
                # Centroid of all but worst
                centroid = np.mean([pt[1] for pt in simplex[:-1]], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1][1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1
                
                if fr < simplex[0][0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1][1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1
                    
                    if fe < fr:
                        simplex[-1] = (fe, xe)
                    else:
                        simplex[-1] = (fr, xr)
                elif fr < simplex[-2][0]:
                    simplex[-1] = (fr, xr)
                else:
                    # Contraction
                    if fr < simplex[-1][0]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1][1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1][1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    
                    if fc < min(fr, simplex[-1][0]):
                        simplex[-1] = (fc, xc)
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = (simplex[i][0], simplex[0][1] + 0.5 * (simplex[i][1] - simplex[0][1]))
                            if calls_made >= max_calls or not deadline_check():
                                break
                            simplex[i] = (func(np.clip(simplex[i][1], -5, 5)), simplex[i][1])
                            calls_made += 1
                
                # Sort and update best
                simplex.sort(key=lambda t: t[0])
                if simplex[0][0] < best_f:
                    best_f = simplex[0][0]
                    best_x = simplex[0][1].copy()
                
                # Convergence check
                if len(simplex) > 1:
                    spread = np.max([np.linalg.norm(simplex[i][1] - simplex[0][1]) for i in range(1, dim+1)])
                    if spread < 1e-6:
                        break
            
            return best_f, best_x

        # Run CMA-ES from the best point
        cma_budget = min(remaining() - 20, 250)
        run_cmaes(best_x, 1.5, cma_budget, lambda: time_ok(5.0))

        # Phase 3: Nelder-Mead refinement from best point
        if remaining() > 10 and time_ok(10.0):
            nm_budget = min(remaining() - 5, 80)
            if nm_budget > 10:
                run_nm(best_x, nm_budget, lambda: time_ok(5.0))

        # Phase 4: Additional restarts if budget remains
        if remaining() > 30 and time_ok(10.0):
            for restart_iter in range(4):
                if remaining() < 20 or not time_ok(10.0):
                    break
                if restart_iter == 0:
                    x_restart = best_x + rng.normal(0, 1.0, dim)
                    sigma_restart = 1.0
                elif restart_iter == 1:
                    x_restart = rng.uniform(-4, 4, dim)
                    sigma_restart = 2.0
                elif restart_iter == 2:
                    x_restart = best_x + rng.normal(0, 2.0, dim)
                    sigma_restart = 1.5
                else:
                    x_restart = rng.uniform(-4, 4, dim)
                    sigma_restart = 1.0
                
                x_restart = np.clip(x_restart, -5, 5)
                cma_budget2 = min(remaining() - 15, 60)
                if cma_budget2 > 10:
                    run_cmaes(x_restart, sigma_restart, cma_budget2, lambda: time_ok(5.0))
                
                # NM refinement after each restart
                if remaining() > 15 and time_ok(5.0):
                    nm_budget2 = min(remaining() - 5, 30)
                    if nm_budget2 > 10:
                        run_nm(best_x, nm_budget2, lambda: time_ok(5.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
