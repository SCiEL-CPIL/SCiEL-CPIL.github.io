"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Nelder-Mead polish for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return np.inf
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        c1 = (2 / ((dim + 1.3)**2 + mueff))
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)))
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chi = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Initialize population
        pop = []
        for _ in range(lambda_):
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            pop.append((fx, x))
        pop.sort(key=lambda t: t[0])
        mean = pop[0][1].copy()
        sigma = 2.0

        # CMA-ES state
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-12))
        sqrtC = B @ np.diag(D) @ B.T
        inv_sqrtC = B @ np.diag(1.0/D) @ B.T

        # Main CMA-ES loop
        while calls_used < budget:
            if time.time() > deadline - 10:
                break

            # Generate offspring
            offspring = []
            for _ in range(lambda_):
                if calls_used >= budget:
                    break
                z = rng.standard_normal(dim)
                y = mean + sigma * (sqrtC @ z)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                offspring.append((fy, y))

            if len(offspring) < mu:
                break

            offspring.sort(key=lambda t: t[0])

            # Update mean
            new_mean = sum(w[i] * offspring[i][1] for i in range(mu))
            new_mean = np.clip(new_mean, -5, 5)

            # Update step size control
            diff = new_mean - mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrtC @ diff) / sigma

            # Update pc
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 1/dim) * chi:
                pc = (1 - c1) * pc + c1 * (new_mean - mean) / sigma

            # Update covariance
            for i in range(mu):
                y_i = (offspring[i][1] - mean) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - c1)**2) * np.eye(dim)) \
                    + cmu * sum(w[j] * np.outer(y_i, y_i) for j in [i])
            
            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs/ds * (np.linalg.norm(ps)/chi - 1))

            mean = new_mean

            # Check for convergence
            if sigma < 1e-10 or np.linalg.norm(diff) < 1e-10:
                break

        # Nelder-Mead polish pass
        if calls_used < budget and best_x is not None:
            # Build initial simplex around best_x
            simplex = [best_x.copy()]
            fvals = [best_f]
            for i in range(dim):
                xi = best_x.copy()
                step = 0.1 * max(1.0, abs(best_x[i]))
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fvals.append(evaluate(xi))
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            
            nm_calls = 0
            max_nm_iter = min(budget - calls_used, 100)
            
            for _ in range(max_nm_iter):
                if calls_used >= budget or time.time() > deadline - 2:
                    break
                
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Centroid of best d points
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = evaluate(xr)
                
                if fxr < fvals[0]:
                    # Expand
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = evaluate(xe)
                    if fxe < fxr:
                        simplex[-1] = xe
                        fvals[-1] = fxe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fxr
                elif fxr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fxr
                else:
                    # Contract
                    if fxr < fvals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = evaluate(xc)
                    if fxc <= fxr:
                        simplex[-1] = xc
                        fvals[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = evaluate(simplex[i])
                            if calls_used >= budget:
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
