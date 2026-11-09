"""CMA-ES based optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    
    Uses a lightweight CMA-ES implementation with adaptive population size,
    restarts on stagnation, and multi-start initialization to handle the
    bi-modal nature of Lunacek's function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ---------- CMA-ES Implementation ----------
        
        def cmaes_run(center, sigma, max_evals, min_sigma=1e-8):
            """Run CMA-ES from given center and sigma. Returns (best_x, best_f)."""
            nonlocal best_x, best_f, calls_made, deadline
            
            # Population size
            lam = int(4 + 3 * np.log(n))  # standard CMA-ES population size
            mu = lam // 2  # number of recombination weights
            
            # CMA-ES parameters
            cc = (4 + n / (n + 4)) / ((n + 1.3) ** 2 + 4)
            cs = (n + 1.7) / (n + 9 + mu_eff)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            # Recombination weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Initialize
            x_mean = center.copy()
            sigma_local = sigma
            C = np.eye(n)  # covariance matrix
            B = np.eye(n)  # eigenvectors
            d = np.ones(n)  # eigenvalues
            pc = np.zeros(n)  # evolution path for covariance
            ps = np.zeros(n)  # evolution path for step size
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # Track best within this CMA-ES run
            run_best_x = x_mean.copy()
            run_best_f = np.inf
            
            # Evaluate initial point
            f0 = safe_func(x_mean)
            if f0 < run_best_f:
                run_best_f = f0
                run_best_x = x_mean.copy()
            
            while calls_made < total_budget and time.time() < deadline - 5:
                # Check if we should stop
                if sigma_local < min_sigma:
                    break
                
                # Generate population
                population = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    # Transform using Cholesky of C
                    x_new = x_mean + sigma_local * (B @ (d * z))
                    x_new = np.clip(x_new, -5, 5)
                    population.append(x_new)
                
                # Evaluate population
                fvals = []
                for x in population:
                    if calls_made >= total_budget or time.time() >= deadline - 5:
                        break
                    f = safe_func(x)
                    fvals.append(f)
                    if f < run_best_f:
                        run_best_f = f
                        run_best_x = x.copy()
                
                if len(fvals) < lam:
                    break
                
                # Sort by fitness
                indices = np.argsort(fvals)
                x_sorted = [population[i] for i in indices]
                f_sorted = [fvals[i] for i in indices]
                
                # Check convergence
                if np.std(f_sorted[:mu]) < 1e-10 * (1 + abs(f_sorted[0])):
                    break
                
                # Weighted mean
                x_new_mean = np.sum([weights[i] * x_sorted[i] for i in range(mu)], axis=0)
                
                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma_local * (B @ (d ** -1) @ (x_new_mean - x_mean))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls_made + 1))) < (1.4 + 2 / (n + 1)) * chi_n
                
                # Update step size
                sigma_local *= np.exp(cs / (n + 2) * (np.linalg.norm(ps) / chi_n - 1))
                sigma_local = max(sigma_local, min_sigma)
                
                # Update covariance matrix
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new_mean - x_mean) / sigma_local
                
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                    c1 * np.outer(pc, pc) + \
                    cmu * np.sum([weights[i] * np.outer(x_sorted[i] - x_mean, x_sorted[i] - x_mean) for i in range(mu)], axis=0)
                
                # Update eigen decomposition (simplified: periodic re-computation)
                if calls_made % (2 * n) == 0 or np.max(np.abs(C - C.T)) > 1e-10:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-12)
                        B = eigvecs
                        d = np.sqrt(eigvals)
                    except:
                        pass
                
                # Update mean
                x_mean = x_new_mean
                x_mean = np.clip(x_mean, -5, 5)
            
            return run_best_x, run_best_f

        # ---------- Phase 0: Multi-start initialization ----------
        # Initialize with several diverse starting points to find both Lunacek basins
        init_points = [
            np.zeros(n),
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            rng.uniform(-3, 3, n),
            rng.uniform(-3, 3, n),
        ]
        
        for xp in init_points:
            if calls_made >= total_budget:
                break
            f = safe_func(xp)

        # ---------- Phase 1: CMA-ES from best initial point ----------
        if best_x is not None and calls_made < total_budget:
            best_x, best_f = cmaes_run(best_x.copy(), 1.0, total_budget - calls_made)
        
        # ---------- Phase 2: Additional CMA-ES runs from other promising points ----------
        # Try from a few other points if budget remains
        if calls_made < total_budget and time.time() < deadline - 10:
            other_starts = [
                np.ones(n) * 1.5,
                np.ones(n) * -1.5,
                rng.uniform(-2, 2, n),
            ]
            
            for start in other_starts:
                if calls_made >= total_budget or time.time() >= deadline - 5:
                    break
                
                f_start = safe_func(start)
                if f_start < best_f * 1.5 + 10:  # Only continue if promising
                    _, _ = cmaes_run(start.copy(), 0.8, total_budget - calls_made)
        
        # ---------- Phase 3: Fine-tuning with small sigma CMA-ES ----------
        if calls_made < total_budget and time.time() < deadline - 5 and best_x is not None:
            _, _ = cmaes_run(best_x.copy(), 0.1, total_budget - calls_made)

        return best_x, best_f
# EVOLVE-BLOCK-END
