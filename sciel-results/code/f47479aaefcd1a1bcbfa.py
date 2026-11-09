"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        d = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(d)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = (2 + 2 / (d + 1)) / ((d + 2)**2 + 2)
        c_mu = min(1 - c_1, (2 + (mu_eff - 2) / (d + 1)) / ((d + 2)**2 + mu_eff + 2))

        # E||N(0,I)|| approximation
        E_norm = np.sqrt(d) * (1 - 1/(4*d) + 1/(32*d**2))

        def run_cmaes(x_center, f_center, sigma_init, budget_remaining, rng):
            """Run CMA-ES from given center, return (best_x, best_f, calls_made)."""
            nonlocal n_calls
            x = x_center.copy()
            f_x = f_center
            sigma = sigma_init
            C = np.eye(d)
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            calls_in_this_run = 0
            no_improve_gen = 0
            best_f_in_run = f_x
            stagnation_threshold = 20  # generations without improvement

            while calls_in_this_run < budget_remaining:
                if time.time() > deadline:
                    break

                # Compute C_inv_sqrt via eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                # Generate offspring
                offspring = []
                for i in range(lam):
                    if calls_in_this_run >= budget_remaining or time.time() > deadline:
                        break
                    z = rng.standard_normal(d)
                    y = x + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    f_y = func(y)
                    n_calls += 1
                    calls_in_this_run += 1
                    offspring.append((f_y, y, z))

                if len(offspring) < 2:
                    break

                # Sort and select
                offspring.sort(key=lambda t: t[0])
                if offspring[0][0] < f_x:
                    if offspring[0][0] < best_f_in_run:
                        no_improve_gen = 0
                    else:
                        no_improve_gen += 1
                    f_x = offspring[0][0]
                    x = offspring[0][1].copy()
                    best_f_in_run = f_x
                else:
                    no_improve_gen += 1

                # Stagnation detection: restart with perturbed center
                if no_improve_gen >= stagnation_threshold and calls_in_this_run < budget_remaining - lam:
                    # Perturb the center
                    perturbation = 0.5 * sigma * rng.standard_normal(d)
                    x = np.clip(x + perturbation, -5, 5)
                    f_x = func(x)
                    n_calls += 1
                    calls_in_this_run += 1
                    best_f_in_run = f_x
                    no_improve_gen = 0
                    # Reset some parameters for fresh start
                    sigma = max(sigma, 0.5)
                    C = np.eye(d)
                    p_sigma = np.zeros(d)
                    p_c = np.zeros(d)
                    continue

                # Weighted mean of selected z's
                z_w = np.zeros(d)
                for i in range(mu):
                    z_w += w[i] * offspring[i][2]

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ z_w)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E_norm - 1))
                sigma = np.clip(sigma, 1e-5, 10.0)

                # Update p_c
                h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 1.0/d) * c_sigma * E_norm else 0.0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ z_w)

                # Update C
                C_temp = np.zeros((d, d))
                for i in range(mu):
                    z_i = offspring[i][2]
                    C_temp += w[i] * np.outer(z_i, z_i)
                
                C = ((1 - c_1 - c_mu + mu_eff * c_sigma * (1 - c_sigma)) * C + 
                     c_1 * (np.outer(p_c, p_c) + (1 - c_sigma * (1 - c_sigma)) * np.eye(d)) + 
                     c_mu * C_temp)
                C = (C + C.T) / 2

            return x, f_x, calls_in_this_run

        # Main execution
        rng = np.random.default_rng(0)
        x = rng.uniform(-4, 4, d)
        f_x = func(x)
        n_calls = 1

        # Phase 1: Main CMA-ES run (use ~70% of budget)
        main_budget = int(budget * 0.7)
        x, f_x, used = run_cmaes(x, f_x, 2.0, main_budget, rng)

        # Phase 2: Polish with smaller sigma (use remaining budget)
        remaining = budget - n_calls
        if remaining > lam:
            x, f_x, used2 = run_cmaes(x, f_x, 0.3, remaining, rng)

        return x, f_x
# EVOLVE-BLOCK-END
