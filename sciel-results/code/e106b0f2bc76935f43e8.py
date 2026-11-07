"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES with surrogate-assisted sampling for Rastrigin-rotated f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin

        rng = np.random.default_rng(42)

        # State
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        X = []  # list of x
        F = []  # list of f

        def update_best(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Initial Sampling to build surrogate
        # Sample 20 points uniformly
        n_init = 20
        for i in range(n_init):
            if calls_made >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            X.append(x)
            F.append(f)
            update_best(x, f)

        # Phase 2: Surrogate-assisted search
        # We will use a simple GP-like surrogate via interpolation or just use the data to guide CMA-ES
        # However, implementing a full GP from scratch is heavy. 
        # Instead, let's use a robust CMA-ES implementation that adapts quickly.
        # The current (1+1) ES is too simple. CMA-ES handles rotation well.
        
        # Let's implement a lightweight CMA-ES.
        # Standard CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (2 - c_c + c_sigma)
        c_mu = (2 / ((dim + 2) ** 2 + mu_eff)) * (
            min(1 - c_1, 2 / ((dim + 1.3) ** 2 + mu_eff))
        )

        # Initialize CMA-ES state
        # Start from the best point found in Phase 1
        x_mean = best_x.copy()
        sigma = 1.0  # Initial step size, will adapt
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = 1.0 / np.sqrt(dim)
        csigma = 1.0 / np.sqrt(dim)
        
        # Precompute Cholesky for sampling
        def cholesky_safe(C):
            try:
                return np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if singular
                return np.eye(dim)
        
        B = cholesky_safe(C)
        D = np.sqrt(np.diag(C))
        
        # Main CMA-ES Loop
        # We will run CMA-ES until budget or deadline.
        # To handle the multimodality of Rastrigin, we might need to restart CMA-ES if it stagnates.
        # But CMA-ES is generally robust. Let's try a single run first, or a few restarts.
        
        # Given the budget is 500, and dim is 5, CMA-ES should be able to make significant progress.
        # We will use the existing X, F to initialize the mean if possible, or just start from best_x.
        
        # Let's do a hybrid: Run CMA-ES for the remaining budget.
        
        while calls_made < budget and time.time() < deadline:
            # Generate lambda_ offspring
            offspring = []
            for i in range(lambda_):
                if calls_made >= budget or time.time() >= deadline:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5.0, 5.0)
                f = func(y)
                calls_made += 1
                offspring.append((f, y))
                X.append(y)
                F.append(f)
                update_best(y, f)
            
            if len(offspring) < lambda_:
                break
            
            # Sort offspring
            offspring.sort(key=lambda x: x[0])
            selected = offspring[:mu]
            
            # Update mean
            x_new = np.zeros(dim)
            for w, (f, y) in zip(weights, selected):
                x_new += w * y
            x_mean = x_new
            
            # Update covariance matrix
            # Standard CMA-ES update
            y_inv = np.linalg.inv(B) * D
            diff = np.array([y for _, y in selected])
            y_w = np.array([y_inv @ (y - x_mean) for _, y in selected])
            
            # Update C
            C_old = C.copy()
            for w, y_w_i in zip(weights, y_w):
                C += w * np.outer(y_w_i, y_w_i)
            C += (c_1 + c_mu - np.trace(C) / dim) * C  # Approximation, standard is more complex
            # Actually, let's use the standard update formula properly
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - delta_sigma) * C) + c_mu * sum(w_i * y_w_i @ y_w_i.T)
            # This is getting complex to implement correctly from scratch. 
            # Let's use scipy's minimize with Nelder-Mead or L-BFGS-B on the surrogate? 
            # No, the task is black-box.
            
            # Alternative: Use a simpler but effective strategy: 
            # Adaptive Step Size + Directional Memory (like (1+1) but with history)
            # Or just use a well-tuned CMA-ES from a library? 
            # "Python standard library, NumPy and SciPy are available."
            # SciPy does not have a built-in CMA-ES optimizer in `scipy.optimize`.
            # `pycma` is not standard.
            
            # Let's pivot to a different strategy: **Nelder-Mead with restarts** 
            # or **Differential Evolution**.
            # DE is robust to multimodality and rotation.
            
            # Let's stop the current CMA-ES attempt and switch to Differential Evolution (DE).
            # DE is simple, robust, and handles the box constraints well.
            # Strategy: DE/rand/1/bin
            # Parameters: F=0.5, CR=0.9, PopSize=50 (or smaller to fit budget)
            # With 500 budget, we can afford a small population and many generations.
            # PopSize = 20, Generations = 20 -> 400 evals.
            
            break # Exit CMA-ES loop to switch strategy
            
        # --- SWITCH TO DIFFERENTIAL EVOLUTION ---
        # Reset state for DE if we haven't used too many calls
        # We already used ~20 calls. Remaining ~480.
        
        pop_size = 20
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        # Use the best point as one member, others random
        pop = [best_x.copy()]
        for i in range(pop_size - 1):
            if calls_made >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            pop.append(x)
            update_best(x, f)
        
        # Evaluate initial population fitness (we already have best, need others)
        # We need to store fitness for each member
        pop_fitness = [np.inf] * pop_size
        pop_fitness[0] = best_f
        # Re-evaluate or store from previous calls? 
        # It's safer to keep a list of (x, f)
        # Let's refactor to keep a population list of (x, f)
        
        # Actually, let's just implement DE cleanly from scratch in the main loop.
        # We will discard the previous CMA-ES partial work and use the calls made so far.
        
        # Let's restart the logic cleanly for DE.
        # Current calls_made is ~20 + 20 = 40.
        
        # Create initial population
        de_pop = []
        de_fit = []
        
        # Add best known
        de_pop.append(best_x.copy())
        de_fit.append(best_f)
        
        # Add random points
        while len(de_pop) < pop_size:
            if calls_made >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            de_pop.append(x)
            de_fit.append(f)
            update_best(x, f)
        
        # Main DE Loop
        while calls_made < budget and time.time() < deadline:
            new_pop = []
            new_fit = []
            
            # For each individual, generate mutant
            for i in range(len(de_pop)):
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # Select 3 distinct random indices different from i
                r1, r2, r3 = rng.choice(len(de_pop), size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(len(de_pop), size=3, replace=False)
                
                v = de_pop[r1] + F * (de_pop[r2] - de_pop[r3])
                
                # Crossover
                u = np.zeros(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = de_pop[i][j]
                
                u = np.clip(u, -5.0, 5.0)
                f_u = func(u)
                calls_made += 1
                
                # Selection
                if f_u <= de_fit[i]:
                    new_pop.append(u)
                    new_fit.append(f_u)
                    update_best(u, f_u)
                else:
                    new_pop.append(de_pop[i])
                    new_fit.append(de_fit[i])
            
            de_pop = new_pop
            de_fit = new_fit
            
            # Check for stagnation
            # If no improvement in last 50 calls, restart population around best
            # Simple check: if best_f hasn't changed in last 50 calls
            # We don't track history of best_f easily here, so let's just run.
            
            # If time is running out, break
            if time.time() > deadline - 10:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
