"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with aggressive mutation and CMA-ES polish for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < deadline

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or not check_time():
                return None
            x_clip = np.clip(x, -5, 5)
            f_val = func(x_clip)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clip.copy()
            return f_val

        # --- Phase 1: Differential Evolution ---
        # Large population to maintain diversity in multimodal landscape
        pop_size = 20
        F = 0.8  # Aggressive mutation factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        pop = []
        for _ in range(pop_size):
            x = rng.uniform(-4, 4, dim)
            f = evaluate(x)
            if f is None:
                break
            pop.append((x, f))
        
        if not pop:
            return best_x, best_f if best_x is not None else np.zeros(dim), best_f

        # Sort initial population
        pop.sort(key=lambda item: item[1])
        
        # Run DE until budget is nearly exhausted or time runs out
        # We reserve some budget for CMA-ES polish if time allows
        # Let's use 80% of budget for DE, 20% for CMA-ES
        de_budget = int(budget * 0.8)
        
        while calls_made < de_budget and check_time():
            # Check if we have enough budget for a full generation
            if calls_made + pop_size > de_budget:
                break
                
            new_pop = []
            for i in range(pop_size):
                # Select three distinct random individuals
                a, b, c = i, i, i
                while a == i or a == b or a == c:
                    a = rng.integers(0, pop_size)
                    b = rng.integers(0, pop_size)
                    c = rng.integers(0, pop_size)
                
                # Differential Evolution mutation: v = x_a + F * (x_b - x_c)
                v = pop[a][0] + F * (pop[b][0] - pop[c][0])
                
                # Binomial crossover
                trial = pop[i][0].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                trial = np.clip(trial, -5, 5)
                f_trial = evaluate(trial)
                if f_trial is None:
                    return best_x, best_f
                
                # Selection
                if f_trial <= pop[i][1]:
                    new_pop.append((trial, f_trial))
                else:
                    new_pop.append(pop[i])
            
            pop = new_pop
            pop.sort(key=lambda item: item[1])

        # --- Phase 2: CMA-ES Polish ---
        # Use remaining budget to refine the best solution found by DE
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and check_time():
            # Initialize CMA-ES from the best DE solution
            x_mean = best_x.copy()
            sigma = 1.0  # Start with smaller step size for polishing
            
            # CMA-ES parameters for dim=5
            c_sigma = (5 + 2) / (dim + 4)
            c_c = (4 + 5 / dim) / (dim + 4 + 2 * 5 / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + 5)
            c_mu = min(1 - c_1, 2 / ((dim + 2) ** 2 + 5))
            d_sigma = 1 + 2 * max(0, np.sqrt((5 - 1) / (dim + 1)) - 1) + c_sigma
            
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Run CMA-ES with remaining budget
            while calls_made + 5 <= budget and check_time():
                # Generate population
                batch_xs = []
                batch_fs = []
                
                for _ in range(5):
                    z = rng.standard_normal(dim)
                    # Efficient sampling using eigen decomposition
                    w, v = np.linalg.eigh(C)
                    w = np.clip(w, 1e-20, None)
                    sqrt_w = np.sqrt(w)
                    x_new = x_mean + sigma * (v.T @ (sqrt_w * z))
                    x_new = np.clip(x_new, -5, 5)
                    
                    f_new = evaluate(x_new)
                    if f_new is None:
                        break
                    
                    batch_xs.append(x_new)
                    batch_fs.append(f_new)
                
                if len(batch_xs) < 5:
                    break
                
                # Sort by fitness
                sort_idx = np.argsort(batch_fs)
                sorted_xs = [batch_xs[i] for i in sort_idx]
                
                # Update mean
                mu = 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= np.sum(weights)
                x_old = x_mean
                x_mean = np.sum([weights[i] * sorted_xs[i] for i in range(mu)], axis=0)
                
                # Update sigma and C (simplified)
                # For polishing, we can use a simpler adaptation
                # Reduce sigma slowly to converge
                sigma *= 0.95
                
                # Update C with rank-one and rank-mu
                inv_sqrt_w = 1.0 / sqrt_w
                inv_sqrt_C = v.T @ np.diag(inv_sqrt_w) @ v
                
                if np.linalg.norm(p_sigma) < 1.0:
                    p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu) * inv_sqrt_C @ (x_mean - x_old) / sigma
                else:
                    p_sigma = (1 - c_sigma) * p_sigma
                
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu) - 1))
                sigma = np.clip(sigma, 0.01, 5.0)
                
                if np.linalg.norm(p_c) < 1.0:
                    p_c = (1 - c_c) * p_c + c_c * (x_mean - x_old) / sigma
                    rank_one = np.outer(p_c, p_c)
                else:
                    p_c = (1 - c_c) * p_c
                    rank_one = np.outer(p_c, p_c)
                
                rank_mu = np.sum([weights[i] * np.outer(sorted_xs[i] - x_old, sorted_xs[i] - x_old) for i in range(mu)], axis=0)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + rank_mu) + c_mu * (rank_mu - C)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Add small regularization to ensure PD
                C += 1e-8 * np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
