"""New optimizer using Covariance Matrix Adaptation Evolution Strategy (CMA-ES) with multi-start restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with restarts for robustness."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        used_budget = 0

        def evaluate(x):
            nonlocal used_budget
            if used_budget >= budget:
                return np.inf
            used_budget += 1
            return func(x)

        # Phase 1: Initial random search to find a good starting point
        n_init = 10
        init_xs = rng.uniform(-5, 5, (n_init, dim))
        init_fs = np.array([evaluate(x) for x in init_xs])
        
        best_idx = np.argmin(init_fs)
        best_x = init_xs[best_idx].copy()
        best_f = init_fs[best_idx]

        if used_budget >= budget:
            return best_x, best_f

        # Phase 2: CMA-ES with restarts
        # Parameters
        lambda_ = 10  # Population size
        mu = lambda_ // 2  # Number of best points to update
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        def run_cmaes(x_start, f_start, remaining_budget):
            # State
            m = x_start.copy()
            sigma = 1.0
            C = np.eye(dim)
            C_inv = np.eye(dim)
            # Precompute Cholesky for C
            try:
                C_eig = np.linalg.cholesky(C)
            except:
                C_eig = np.eye(dim)
            
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 + 1/(4*dim) + 1/(20*dim**2))
            
            gen = 0
            last_best_f = f_start
            stagnation_count = 0
            
            while used_budget < budget:
                if used_budget + lambda_ > budget:
                    # Adjust population size if remaining budget is small
                    lambda_ = max(4, budget - used_budget)
                    if lambda_ <= 2:
                        break
                    mu = max(1, lambda_ // 2)
                    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    w /= w.sum()
                    mueff = 1 / np.sum(w**2)
                    # Recalculate rates? For simplicity, keep old rates or just stop.
                    # Given small budget, just do random sampling if we can't fit a pop?
                    # No, let's just break and do random search in main loop if needed.
                    # But CMA-ES needs a pop. If remaining < 4, break.
                    break

                # Sample
                ys = []
                xs = []
                fs = []
                
                # Update Cholesky if C changed (it does every step)
                try:
                    C_eig = np.linalg.cholesky(C)
                except:
                    # Regularize C
                    C = C + 1e-10 * np.eye(dim)
                    C_eig = np.linalg.cholesky(C)

                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = C_eig @ z
                    x = m + sigma * y
                    x = np.clip(x, -5, 5)
                    fx = evaluate(x)
                    ys.append(y)
                    xs.append(x)
                    fs.append(fx)

                # Sort
                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                xs = [xs[i] for i in order]
                fs = [fs[i] for i in order]

                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = xs[0].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # Update mean
                y_w = np.sum(np.array(ys[:mu]) * w[:, None], axis=0)
                m = m + sigma * c1 * (y_w - pc) # This is not standard, standard is m += sigma * sum(w_i * (x_i - m)/sigma)
                # Correct update:
                # m = sum(w_i * x_i) for i in 1..mu
                m = np.sum(np.array(xs[:mu]) * w[:, None], axis=0)

                # Update step size
                y_sigma = C_eig @ y_w
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(y_sigma + 1e-10) * y_sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                
                # Clamp sigma
                if sigma > 10: sigma = 10
                if sigma < 1e-8: sigma = 1e-8

                # Update covariance
                pc = (1 - cc) * pc + cc * (1 / (1 - cc)) * y_w
                rank_mu = np.outer(y_w, y_w)
                rank_one = np.outer(ps, ps)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(gen + 1)) / chi_n) < 1.4 + 2 / (dim + 1)
                hsig = 1 if hsig else 0
                C = (1 + c1 * (hsig * (np.outer(ps, ps) - C) - c1 * C) + cmu * rank_mu) * C
                # Simplified update often used:
                # C = (1 - c1 - cmu) * C + c1 * (1 + hsig) * np.outer(ps, ps) + cmu * rank_mu
                C = (1 - c1 - cmu) * C + c1 * (1 + hsig) * np.outer(ps, ps) + cmu * rank_mu
                
                # Symmetrize
                C = (C + C.T) / 2
                
                gen += 1
                
                # Restart condition
                if stagnation_count > 5:
                    # Restart from best with larger sigma
                    break

            return best_x, best_f

        # Run CMA-ES
        while used_budget < budget:
            # Start CMA-ES from current best
            x_start = best_x.copy()
            f_start = best_f
            
            # Run CMA-ES until stagnation or budget
            # We need to pass the state or just let it run. 
            # The inner function `run_cmaes` above is a bit messy with closures. 
            # Let's implement a cleaner loop.
            
            # Actually, let's just run one main CMA-ES loop that handles restarts internally.
            
            pass # Placeholder for logic below

        # Re-implementing the loop cleanly in the main body to avoid nested function complexity issues with `nonlocal` in some Python versions or just for clarity.
        
        # Reset state for a clean multi-start CMA-ES
        best_x, best_f = best_x, best_f # Keep initial best
        
        # We already used 10 calls.
        # Let's run CMA-ES from best_x
        
        lambda_ = 10
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)
        
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        chi_n = np.sqrt(dim) * (1 + 1/(4*dim) + 1/(20*dim**2))
        
        while used_budget < budget:
            m = best_x.copy()
            sigma = 1.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            stagnation_count = 0
            gen = 0
            
            while used_budget < budget:
                if used_budget + lambda_ > budget:
                    break
                
                try:
                    C_eig = np.linalg.cholesky(C)
                except:
                    C = C + 1e-10 * np.eye(dim)
                    C_eig = np.linalg.cholesky(C)

                ys = []
                xs = []
                fs = []
                
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = C_eig @ z
                    x = m + sigma * y
                    x = np.clip(x, -5, 5)
                    fx = evaluate(x)
                    ys.append(y)
                    xs.append(x)
                    fs.append(fx)

                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                xs = [xs[i] for i in order]
                fs = [fs[i] for i in order]

                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = xs[0].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # Update mean
                m = np.sum(np.array(xs[:mu]) * w[:, None], axis=0)

                # Update step size
                y_w = np.sum(np.array(ys[:mu]) * w[:, None], axis=0)
                y_sigma = C_eig @ y_w
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / (np.linalg.norm(y_sigma) + 1e-10) * y_sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10)

                # Update covariance
                pc = (1 - cc) * pc + cc * (1 / (1 - cc)) * y_w
                rank_mu = np.outer(y_w, y_w)
                rank_one = np.outer(ps, ps)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(gen + 1)) / chi_n) < 1.4 + 2 / (dim + 1)
                hsig = 1 if hsig else 0
                C = (1 - c1 - cmu) * C + c1 * (1 + hsig) * rank_one + cmu * rank_mu
                C = (C + C.T) / 2
                
                gen += 1
                
                if stagnation_count > 5:
                    break
            
            # If we broke out of inner loop due to stagnation but still have budget,
            # we restart from best_x with a fresh CMA-ES (reset C, sigma).
            # This effectively implements a multi-start strategy driven by stagnation.
            if used_budget >= budget:
                break
            
            # Optional: If we are stuck, try a random jump?
            # For now, the reset of C and sigma in the next iteration of the outer while loop
            # (since m is reset to best_x) acts as a restart.
            # However, if best_x is a local minimum, restarting from it with C=I and sigma=1
            # might just get stuck again.
            # To escape, we can perturb best_x slightly before restarting.
            if stagnation_count > 5 and used_budget < budget:
                # Perturb best_x
                best_x = best_x + rng.standard_normal(dim) * 0.5
                best_x = np.clip(best_x, -5, 5)
                # Re-evaluate? No, just use it as start for next CMA-ES.
                # The next CMA-ES will evaluate points around it.

        return best_x, best_f
# EVOLVE-BLOCK-END
