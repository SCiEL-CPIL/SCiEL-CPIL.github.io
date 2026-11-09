"""Hybrid Differential Evolution and CMA-ES optimizer for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid DE (global) + CMA-ES (local) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # Initialize DE
        pop_size = max(10, 4 * dim)
        # Use a fixed seed for reproducibility in the class, though instances vary
        rng = np.random.default_rng(42)
        
        # Initial population spread over [-5, 5]
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        calls_made = pop_size
        
        # Adaptive parameters for DE
        F = 0.8
        CR = 0.5
        
        # Phase 1: Differential Evolution
        # We allocate roughly 60% of the budget to DE for global exploration
        de_budget = int(budget * 0.6)
        
        for gen in range(de_budget):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
                
            # Update adaptive parameters (simple adaptation)
            # jDE-like adaptation based on success rate is complex, 
            # so we use a simple decay or fixed values tuned for BBOB
            F = 0.7 + 0.3 * np.exp(-gen / 50.0)
            CR = 0.5 + 0.2 * np.sin(gen / 10.0)
            
            for i in range(pop_size):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Select 3 distinct individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                # Ensure they are not i
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Create trial vector
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                j_rand = rng.integers(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluate
                ft = func(trial)
                calls_made += 1
                
                # Selection
                if ft <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                        best_idx = i
                else:
                    # If failed, maybe adjust F/CR slightly? 
                    # For simplicity, we rely on the global adaptation above.
                    pass
        
        # Phase 2: CMA-ES Refinement
        # Use remaining budget for CMA-ES starting from best_x
        remaining_budget = budget - calls_made
        if remaining_budget > 0 and time.time() < deadline:
            # CMA-ES Implementation
            x_mean = best_x.copy()
            sigma = 0.5 * (5.0 - (-5.0)) / np.sqrt(dim) # Heuristic initial sigma
            # Actually, let's estimate sigma from the DE population spread or use a small value
            # Since DE found a good basin, a smaller sigma is better for start.
            sigma = 0.2
            
            # CMA-ES parameters
            mu = pop_size // 2  # Selection pressure
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Update rules
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Generate initial population for CMA-ES
            # We will reuse the budget count
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                # Generate lambda = pop_size points
                # To save time, we can generate a smaller population for CMA-ES if needed,
                # but pop_size from DE is usually okay (10-20).
                # Let's use a smaller lambda for CMA-ES to be efficient, e.g., max(4, dim+3)
                lambda_cma = max(4, dim + 3)
                if lambda_cma > pop_size:
                     # If DE pop was small, we can just use what we have or generate new
                     # For simplicity, generate new standard normals
                     pass
                
                # Generate samples
                # We need to generate lambda_cma samples
                # To fit within the loop structure, we generate them one by one or in batch?
                # Batch is faster.
                
                # However, the loop above is per-generation. 
                # Let's restructure the CMA-ES part to be a separate loop.
                
                break # Exit the DE loop structure to start CMA-ES properly
                
            # Proper CMA-ES Loop
            lambda_cma = max(4, dim + 3)
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                # Generate population
                # Standard normal samples
                Z = rng.standard_normal((lambda_cma, dim))
                # Transform using C
                # C = B * diag(D^2) * B.T
                # y = x_mean + sigma * B * D * z
                # We can compute B @ (D * Z.T).T
                Y = np.einsum('ij,j,ij->ik', B, D, Z)
                P = x_mean + sigma * Y
                
                # Evaluate
                F_vals = np.array([func(x) for x in P])
                calls_made += lambda_cma
                
                # Sort
                order = np.argsort(F_vals)
                P = P[order]
                F_vals = F_vals[order]
                
                # Update best
                if F_vals[0] < best_f:
                    best_f = F_vals[0]
                    best_x = P[0].copy()
                
                # Update mean
                x_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * P[:mu], axis=0)
                
                # Update step size control
                # p_sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B.T @ (D * (x_mean - x_old)))
                chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim**2))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - de_budget - lambda_cma))) < (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update covariance matrix
                # p_c
                pc = (1 - c_c) * pc + c_c * (x_mean - x_old) / sigma
                # Delta correction
                delta_corr = (1 - np.exp(-2 * c_c * (calls_made - de_budget - lambda_cma) / dim)) if False else 1.0 # Simplified
                # Actually, standard formula:
                # delta_h = (1 - exp(-2 * c_c * age / dim)) * (1 - exp(-c_c * age))
                # We need age counter. Let's use calls_made relative to start of CMA.
                
                # Update C
                # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (np.linalg.norm(pc)**2 > (1 + 4/c_c) * chi_n**2) * (chi_n**2 - 1)) + c_mu * sum(weights_i * (y_i - pc/sigma) (y_i - pc/sigma).T)
                
                # To keep it simple and robust, we use the rank-one and rank-mu updates
                # y_i = (P[i] - x_old) / sigma
                Y_centered = (P[:mu] - x_old) / sigma
                # Rank-mu update
                rank_mu = np.sum(weights[:, None, None] * Y_centered[:, :, None] * Y_centered[:, None, :], axis=0)
                
                # Rank-one update
                # Check for divergence
                if np.linalg.norm(pc) / chi_n < (1.4 + 2 / (dim + 1)):
                    rank_one = pc * pc.T + (np.linalg.norm(pc)**2 > (1 + 4/c_c) * chi_n**2) * (chi_n**2 - 1)
                else:
                    rank_one = pc * pc.T
                
                C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Recompute B and D via Cholesky
                # Add jitter for numerical stability
                try:
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                    # B is the transpose of the inverse of L? 
                    # Standard CMA-ES: C = B diag(D^2) B.T
                    # If we use Cholesky L such that C = L L.T, then B = L.T, D = diag(L)
                    # Or simpler: use eigen decomposition? Cholesky is faster.
                    # Let's use the relation: if C = L L.T, then we can set B = L.T and D = diag(L)
                    # Wait, C = B D^2 B.T. If B is orthogonal, D^2 are eigenvalues.
                    # Cholesky gives C = L L.T. This is not directly B D^2 B.T with orthogonal B.
                    # Standard implementation uses eigen decomposition or specific Cholesky update.
                    # For simplicity and speed in Python, eigen decomposition is safer for small dim (5).
                    evals, evecs = np.linalg.eigh(C)
                    # Clip negative eigenvalues
                    evals = np.clip(evals, 1e-12, None)
                    D = np.sqrt(evals)
                    B = evecs
                except np.linalg.LinAlgError:
                    # Fallback: reset to identity
                    B = np.eye(dim)
                    D = np.ones(dim)
                    C = np.eye(dim)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    sigma = 0.5
                    x_mean = best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
