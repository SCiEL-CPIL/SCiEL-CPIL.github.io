"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: random exploration + CMA-ES + L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        calls = [0]
        best_x = None
        best_f = np.inf

        def wrapped(x):
            calls[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Phase 1: Random exploration to find a good starting point
        # Use 31 points including origin
        n_explore = min(31, budget)
        explore_points = []
        # Origin
        explore_points.append(np.zeros(d))
        # Random points
        while len(explore_points) < n_explore:
            explore_points.append(rng.uniform(-5, 5, d))

        for p in explore_points:
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break
            wrapped(p)

        # Sort by function value to find best
        # We need to track which points gave which values
        # Let's redo this more carefully
        # Actually, let me restructure: collect all exploration results

        # Reset and do exploration properly
        calls[0] = 0
        best_f = np.inf
        best_x = None
        explore_results = []

        def wrapped2(x):
            nonlocal best_f, best_x
            calls[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            explore_results.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        explore_points = [np.zeros(d)]
        while len(explore_points) < n_explore:
            explore_points.append(rng.uniform(-5, 5, d))

        for p in explore_points:
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break
            wrapped2(p)

        # Find the best exploration point
        if explore_results:
            explore_results.sort(key=lambda t: t[0])
            start_x = explore_results[0][1].copy()
        else:
            start_x = np.zeros(d)

        # Phase 2: CMA-ES on the remaining budget
        remaining_budget = budget - calls[0]
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f

        # CMA-ES implementation
        sigma = 1.0
        # Population size: 4 + floor(3 * ln(n))
        pop_size = max(4, int(4 + 3 * np.log(d + 1)))
        # Adjust to fit remaining budget
        max_iter = remaining_budget // pop_size
        if max_iter < 1:
            # Not enough budget for CMA-ES, just do a few more random steps
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                y = np.clip(start_x + sigma * rng.standard_normal(d), -5, 5)
                fy = wrapped2(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            return best_x, best_f

        # CMA-ES state
        mean = start_x.copy()
        C = np.eye(d)
        Cs = np.zeros(d)
        ps = np.zeros(d)
        pc = np.zeros(d)
        chi_n = np.sqrt(d) * (1 + 1/(4*d) - 1/(20*d*d))
        cc = (4 + d/2) / (d + 4 + 2*chi_n**2)
        cs = (1 + 2*max(0, np.sqrt((d-1)/(d+2)) - 1)) / (d + 4)
        c1 = 2 / ((d + 1.3)**2 + pop_size)
        cmu = min(1 - c1, 2 * ((pop_size - 2 + 1/pop_size) / (d + 2) - 1))
        damps = 1 + 2*max(0, np.sqrt((pop_size - 1)/(d + 1) - 1)) + cs
        inv_sigma = 1.0 / sigma

        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if calls[0] >= budget:
                break

            # Generate population
            D = np.linalg.cholesky(C)
            # Sample from N(0, C)
            z = rng.standard_normal((pop_size, d))
            # Project to get N(0, C)
            # Use Cholesky: z @ D gives N(0, C) if z ~ N(0, I)
            # Actually, if C = D @ D.T, then D.T @ z ~ N(0, C)
            # Let's use: w = D.T @ z.T -> shape (d, pop_size)
            w = D.T @ z.T  # (d, pop_size)
            # w[:, i] ~ N(0, C)
            pop = mean + sigma * w.T  # (pop_size, d)
            pop = np.clip(pop, -5, 5)

            # Evaluate
            fvals = []
            for i in range(pop_size):
                if calls[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                fvals.append(wrapped2(pop[i]))
            
            if len(fvals) < pop_size:
                break

            fvals = np.array(fvals)
            # Sort indices by fitness
            order = np.argsort(fvals)
            sorted_pop = pop[order]
            sorted_fvals = fvals[order]

            # Update mean
            weights = np.ones(pop_size)
            weights[0] += 0.5 * pop_size
            # Actually standard CMA-ES uses specific weights
            # Let's use simple: top half get weight 1, rest get 0
            # Or use the standard: w_i = log(pop_size/2 + 0.5) - log(i+1) for i < pop_size/2
            mu_eff = pop_size // 2
            w = np.zeros(pop_size)
            for i in range(mu_eff):
                w[i] = np.log(mu_eff + 0.5) - np.log(i + 1)
            w /= w.sum()
            mu_eff = (w[:mu_eff]**2).sum() ** -0.5

            mean_new = np.sum(w[:, None] * sorted_pop, axis=0)

            # Update covariance
            delta_mean = mean_new - mean
            Cs = (1 - cs) * Cs + cs * delta_mean / sigma
            pc = (1 - cc) * pc + cc * delta_mean / sigma

            # Update C
            C = (1 + c1 * mu_eff * cmu - c1) * C
            for i in range(mu_eff):
                y_i = (sorted_pop[i] - mean) / sigma
                C += cmu * w[i] * np.outer(y_i, y_i)
            C += c1 * (np.outer(Cs, Cs) - C)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Check if we need to rescale sigma
            if np.linalg.norm(pc) / np.sqrt(1 - (1 - cc)**(2 * (iteration + 1))) > (1.4 + 1/d) * chi_n:
                sigma *= np.exp(cs / damps * (np.linalg.norm(pc) / chi_n - 1))

            mean = mean_new

            # Clamp sigma
            sigma = max(1e-8, min(sigma, 10.0))

        # Phase 3: L-BFGS-B polish
        if calls[0] < budget and time.time() < deadline:
            try:
                # Use the best point found so far
                x0 = best_x.copy()
                # Limit the number of function evaluations for L-BFGS-B
                remaining = budget - calls[0]
                maxfun = min(remaining, 200)
                
                # We need to use a wrapper that counts calls
                lbg_calls = [0]
                def lbg_wrapper(x):
                    lbg_calls[0] += 1
                    x = np.clip(np.asarray(x, dtype=float), -5, 5)
                    f = float(func(x))
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    return f

                res = minimize(lbg_wrapper, x0, method='L-BFGS-B',
                              bounds=[(-5, 5)] * d,
                              options={'maxfun': maxfun, 'maxiter': maxfun,
                                       'ftol': 1e-12, 'gtol': 1e-12})
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
