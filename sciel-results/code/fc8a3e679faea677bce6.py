"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, restarts, and CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: (1+1)-ES with restarts
        x = np.zeros(dim)  # Start at origin (min is in [-4,4]^5)
        fx = func(x)
        sigma = 1.5
        no_improve = 0
        max_no_improve = 40
        calls_made = 1

        for i in range(budget - 1):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                sigma *= 1.3
            else:
                no_improve += 1
                sigma *= 0.9
                if no_improve >= max_no_improve:
                    # Restart: perturb from current best
                    no_improve = 0
                    sigma = max(sigma * 2.0, 0.5)
                    x = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fx = func(x)
                    calls_made += 1
                    if calls_made >= budget:
                        break

        # Phase 2: CMA-ES polish with remaining budget
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline:
            # Simple CMA-ES-like local search
            c_sigma = max(sigma * 0.5, 0.1)
            c_mu = 0.5
            c_chi = np.sqrt(dim) * (1 + 1/(4*dim) - 1/(2*dim*dim))
            c_c = (4 + dim/4) / (dim + 4 + 2*dim/4)
            c_1 = 2 / ((dim + 1.3)**2 + dim)
            c_mu = 2 / ((dim + 2)**2 + dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            inv_C = np.eye(dim)
            d_corr = 1 + 2 * max(0, (c_c + 1) / (c_c - 1) - dim)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim*dim))
            
            # Use remaining budget for CMA-ES steps
            n_evals = min(remaining - 5, 100)  # Cap CMA-ES evaluations
            lam = max(4, 2 * dim + 2)  # Population size
            mu = lam // 2
            
            for step in range(n_evals):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                # Generate offspring
                w = np.ones(mu) / mu
                w /= w.sum()
                weights = w
                mu_eff = 1.0 / np.sum(w**2)
                
                # Update sigma
                sigma *= np.exp((c_1 + c_mu) * (np.sum(np.linalg.norm(rng.standard_normal((lam, dim)), axis=1) / chi_n) / lam - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Generate and evaluate offspring
                offspring = []
                for j in range(lam):
                    z = rng.standard_normal(dim)
                    y = np.clip(x + c_sigma * sigma * z, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    offspring.append((fy, y, z))
                    if calls_made >= budget:
                        break
                
                if len(offspring) == 0:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Update mean
                for k in range(min(mu, len(offspring))):
                    x = x + weights[k] * (offspring[k][1] - x)
                
                # Update covariance (simplified)
                v = np.array([offspring[k][2] for k in range(min(mu, len(offspring)))])
                v = (v.T * weights[:, None]).T
                diff = np.array([offspring[k][1] - x for k in range(min(mu, len(offspring)))])
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(diff[0], diff[0]) + c_mu * np.outer(np.sum(diff * weights[:, None], axis=0), np.sum(diff * weights[:, None], axis=0)) + (1 - c_1 - c_mu + c_1 + c_mu) * np.eye(dim) * 0  # simplified
                # Keep C well-conditioned
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                fx = offspring[0][0]
            
            # Final best
            # Try a few more local steps
            for _ in range(min(5, budget - calls_made)):
                if time.time() > deadline or calls_made >= budget:
                    break
                y = np.clip(x + sigma * 0.1 * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy

        return x, fx
# EVOLVE-BLOCK-END
