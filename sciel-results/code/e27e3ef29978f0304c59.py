"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for high-dimensional continuous problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        dim = self.dim
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # 10 for dim=5
        sigma = 2.0
        x_mean = np.zeros(dim)
        
        # Covariance matrix (identity initially)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Step size control
        cs = 1 / (np.sqrt(dim) + 3)
        ds = 1 + 2 * max(0, np.sqrt((pop_size - 2) / (dim + 2)) - 1) + cs
        c_sigma = (ds + 2) / ((dim + 1.3) * (ds + 5))
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - dim - 2) / ((dim + 2) ** 2 + pop_size))
        
        # Weights for recombination
        w = np.log(pop_size / 2) - np.log(np.arange(1, pop_size + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        evals = 1
        
        while evals < self.budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate population
            population = []
            for i in range(pop_size):
                # Transform standard normal using C_sqrt
                z = rng.standard_normal(dim)
                # Simple approximation for C_sqrt update if needed, but for dim=5 we can just use Cholesky occasionally
                # For simplicity and speed in dim=5, we can use a simplified rank-mu update or just stick to diagonal adaptation if full CMA is too heavy.
                # However, standard CMA-ES is robust. Let's implement a lightweight version.
                
                # To avoid expensive Cholesky decomposition every step in a tight loop, 
                # and since dim is small (5), we can actually afford it or use a simpler covariance adaptation.
                # Let's use a simplified (1+lambda)-ES with diagonal covariance (SEP-ES) which is often sufficient and much faster/stable.
                # But the prompt asks for a DIFFERENT strategy. CMA-ES is the gold standard here.
                # Let's implement a basic CMA-ES with full covariance.
                
                pass 

            # Actually, implementing full CMA-ES from scratch is error-prone. 
            # A very strong alternative for Rosenbrock is **Nelder-Mead** with multi-start, 
            # or **Powell's method**. 
            # But let's stick to the population-based idea with a simpler, robust implementation:
            # **Particle Swarm Optimization (PSO)** or **Differential Evolution (DE)**.
            
            # Let's switch to **Differential Evolution (DE)** which is extremely robust for black-box optimization 
            # and handles rotated valleys well with the right mutation strategy (DE/rand/1/bin).
            
            # Reset to DE strategy
            break # Exit CMA loop, restart with DE logic below
            
        # --- DROPPED CMA IMPLEMENTATION IN FAVOR OF DIFFERENTIAL EVOLUTION (DE) ---
        # DE is simpler, very effective for Rosenbrock, and fits the "different design family" (population-based stochastic search vs (1+1)-ES).
        
        # Re-initialize for DE
        start_time = time.time()
        deadline = start_time + 290
        
        dim = self.dim
        pop_size = 20  # Larger population for exploration
        F = 0.8       # Mutation scale factor
        CR = 0.9      # Crossover rate
        
        # Initialize population
        population = np.random.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(x) for x in population])
        evals = pop_size
        
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Track global best
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            for i in range(pop_size):
                if evals >= self.budget or time.time() > deadline:
                    break
                    
                # Select three distinct individuals
                a, b, c = rng.choice(pop_size, 3, replace=False)
                while a == b or a == c or b == c:
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = population[a] + F * (population[b] - population[c])
                
                # Crossover
                u = rng.uniform(size=dim)
                j_rand = rng.integers(dim)
                trial = np.where(u < CR, v, population[i])
                trial[j_rand] = v[j_rand]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                evals += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
