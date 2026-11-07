import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, keep 280s to be safe
        deadline = start_time + 280
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        calls = 0
        fbest = np.inf
        x_best = np.zeros(n)

        def time_ok():
            return time.time() < deadline

        def eval(x):
            nonlocal calls, fbest, x_best
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < fbest:
                fbest = fx
                x_best = x.copy()
            return fx

        # Phase 1: Initial Sampling
        # Evaluate origin
        if calls < budget and time_ok():
            eval(np.zeros(n))
        
        # Evaluate a few random points to get a better initial mean
        n_init = 5
        for _ in range(n_init):
            if calls >= budget or not time_ok():
                break
            x_try = rng.uniform(-5, 5, n)
            eval(x_try)

        # Phase 2: CMA-ES with Adaptive Restart
        # CMA-ES Parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initialize CMA-ES state
        def init_cma():
            nonlocal sigma, x_mean, pc, ps, C, C_sqrt, inv_C_sqrt, gen, no_improve
            x_mean = x_best.copy()
            sigma = 1.0
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            C_sqrt = np.eye(n)
            inv_C_sqrt = np.eye(n)
            gen = 0
            no_improve = 0

        sigma = 1.0
        x_mean = x_best.copy()
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        C_sqrt = np.eye(n)
        inv_C_sqrt = np.eye(n)
        gen = 0
        no_improve = 0
        best_so_far = fbest

        # Main CMA-ES Loop
        while calls < budget - 20 and time_ok() and (deadline - time.time()) > 30:
            offspring = []
            for i in range(lam):
                if calls >= budget - 20 or not time_ok():
                    break
                z = rng.standard_normal(n)
                y = C_sqrt @ z
                xi = x_mean + sigma * y
                fi = eval(xi)
                offspring.append((fi, xi, z))

            if len(offspring) < 2:
                break

            offspring.sort(key=lambda t: t[0])

            # Check for improvement
            if offspring[0][0] < best_so_far - 1e-10:
                best_so_far = offspring[0][0]
                no_improve = 0
            else:
                no_improve += 1

            # Restart logic: If stuck for 20 generations, reset CMA-ES around best point
            if no_improve > 20:
                # Perturb the best point slightly
                x_mean = x_best + 0.5 * rng.standard_normal(n)
                x_mean = np.clip(x_mean, -5, 5)
                sigma = 1.0
                pc = np.zeros(n)
                ps = np.zeros(n)
                C = np.eye(n)
                C_sqrt = np.eye(n)
                inv_C_sqrt = np.eye(n)
                gen = 0
                no_improve = 0
                continue

            # Update mean
            x_old = x_mean.copy()
            x_mean = np.zeros(n)
            for i in range(mu):
                x_mean += weights[i] * offspring[i][1]

            # Update sigma and covariance
            if sigma > 1e-12:
                y_mean = (x_mean - x_old) / sigma
            else:
                y_mean = np.zeros(n)

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_C_sqrt @ y_mean)
            ps_norm = np.linalg.norm(ps)
            h_sigma = 1.0 if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            if h_sigma:
                sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))

            pc = (1 - c_c) * pc + c_c * y_mean
            delta = (1 + 2 * max(0, mu_eff / n - 1)) * c_c * (1 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(offspring[i][2], offspring[i][2])
            C = (1 + c_1 * delta - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                inv_C_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                C_sqrt = np.eye(n)
                inv_C_sqrt = np.eye(n)

            sigma = np.clip(sigma, 1e-5, 10.0)
            gen += 1

        # Phase 3: Local Refinement with Nelder-Mead
        # Use remaining budget for local search
        if calls < budget and time_ok() and (deadline - time.time()) > 10:
            try:
                def nm_func(x):
                    nonlocal calls, fbest, x_best
                    if calls >= budget or not time_ok():
                        return fbest
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    if fx < fbest:
                        fbest = fx
                        x_best = x.copy()
                    return fx

                # Nelder-Mead is good for final polishing
                result = scipy.optimize.minimize(
                    nm_func, x_best, method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

            # If we still have budget, try a second polish from a slightly different start
            if calls < budget and time_ok() and (deadline - time.time()) > 5:
                try:
                    x_perturb = x_best + 0.05 * rng.standard_normal(n)
                    x_perturb = np.clip(x_perturb, -5, 5)

                    def nm_func2(x):
                        nonlocal calls, fbest, x_best
                        if calls >= budget or not time_ok():
                            return fbest
                        x = np.clip(x, -5, 5)
                        fx = func(x)
                        calls += 1
                        if fx < fbest:
                            fbest = fx
                            x_best = x.copy()
                        return fx

                    result = scipy.optimize.minimize(
                        nm_func2, x_perturb, method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                except Exception:
                    pass

        # Phase 4: Final fine-tuning if budget remains
        if calls < budget and time_ok():
            for scale in [0.01, 0.001]:
                if calls >= budget or not time_ok():
                    break
                for _ in range(min(10, budget - calls)):
                    if calls >= budget or not time_ok():
                        break
                    perturbation = rng.standard_normal(n) * scale
                    x_try = x_best + perturbation
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < fbest:
                        fbest = f_try
                        x_best = x_try.copy()

        return x_best, fbest
