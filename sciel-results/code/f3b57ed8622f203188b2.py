"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Optimizer: Latin Hypercube initialization followed by CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin from 300s limit
        
        # 1. Initialization Phase: Latin Hypercube Sampling
        # We spend ~20% of the budget on LHS to find a good starting point.
        # LHS ensures coverage of the box [-5, 5]^5.
        n_lhs = max(10, self.budget // 5)
        
        # Generate LHS points
        # For each dimension, we divide the range into n_lhs strata and pick one point from each.
        # Since we want to minimize calls, we generate n_lhs points total.
        # Standard LHS: 
        # 1. Create an n_lhs x n_lhs matrix where each row is a permutation of 0..n_lhs-1
        # 2. Scale to [0, 1]
        # 3. Scale to [-5, 5]
        
        lhs_points = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            # Random permutation of indices
            perm = np.random.permutation(n_lhs)
            # Random offset within each stratum
            offsets = np.random.rand(n_lhs)
            # Calculate position in [0, 1]
            positions = (perm + offsets) / n_lhs
            # Scale to [-5, 5]
            lhs_points[:, d] = -5 + 10 * positions

        best_x = None
        best_f = np.inf
        
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x

        # If we haven't made any calls yet (shouldn't happen unless n_lhs=0), do a random one
        if best_x is None:
            best_x = np.random.uniform(-5, 5, self.dim)
            best_f = func(best_x)

        # 2. Optimization Phase: CMA-ES
        # We use the remaining budget for CMA-ES.
        # CMA-ES adapts the covariance matrix to align with the function's landscape (e.g., sharp ridges).
        # This is significantly faster than (1+1)-ES for ill-conditioned problems.
        
        remaining_budget = self.budget - n_lhs
        
        # CMA-ES parameters
        # Population size: standard is 4 + floor(3 * log(n))
        pop_size = max(4, 4 + int(3 * np.log(self.dim)))
        
        # Initial sigma: heuristic based on best_x and box size.
        # Since we don't know the scale of the minimum, start with a moderate step.
        # The LHS phase helps ensure we are in the basin of attraction.
        sigma = 0.5
        
        # CMA-ES State
        # mean: current mean of the population
        # C: covariance matrix
        # step_sizes: diagonal of C (sigma * sqrt(diag(C)))
        # We implement a simplified CMA-ES to avoid external dependencies if needed, 
        # but since SciPy is available, we could use it. However, a lightweight 
        # implementation is often more robust in these black-box constraints if 
        # the API is strict. Let's implement a robust (mu+lambda)-CMA-ES variant.
        
        # Actually, implementing full CMA-ES is complex. 
        # A simpler, very effective alternative for 5D is a well-tuned (1,1)-ES 
        # with adaptive step size, OR a simple Nelder-Mead.
        # However, the prompt mentions "CMA-ES adapts covariance... enabling faster convergence".
        # Let's implement a basic CMA-ES.
        
        # State variables
        mean = best_x.copy()
        C = np.eye(self.dim)
        sigma = 0.5
        
        # Weights
        # Standard CMA-ES uses weighted recombination.
        # For simplicity and speed in 5D, we can use a (mu, lambda) scheme.
        # Let's use a population size of 10 (small enough for 500 budget).
        # Actually, let's stick to a simpler, proven method: 
        # **Nelder-Mead** is very effective for 5D and doesn't require many parameters.
        # But Nelder-Mead can struggle with sharp ridges if not tuned.
        
        # Let's try a **Simplex-based** approach with restarts or just robust CMA-ES.
        # Given the "sharp ridge" description, CMA-ES is theoretically superior.
        # I will implement a minimal CMA-ES.
        
        # CMA-ES Implementation Details:
        # 1. Sample lambda points: y_i = mean + sigma * (z_i @ C)
        # 2. Evaluate f(y_i)
        # 3. Sort by fitness
        # 4. Update mean, C, sigma
        
        lambda_ = 10  # Population size
        mu = lambda_ // 2  # Number of parents
        # Weights: linear weights, normalized
        w = np.arange(1, mu + 1)
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # Evolution paths
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        # Damping for pc update
        chi_n = np.sqrt(self.dim) * (1 - 1/(2*self.dim**2) + 1/(12*self.dim**4))
        
        # Pre-compute random matrix for sampling? No, sample each time.
        
        calls_made = n_lhs
        best_x_es = best_x
        best_f_es = best_f
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            pop = np.zeros((lambda_, self.dim))
            # Standard normal samples
            Z = np.random.randn(lambda_, self.dim)
            # Transform using Cholesky of C
            L = np.linalg.cholesky(C)
            for i in range(lambda_):
                y = Z[i] @ L
                pop[i] = mean + sigma * y
            
            # Evaluate
            fitness = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                calls_made += 1
                fitness[i] = func(pop[i])
                if fitness[i] < best_f_es:
                    best_f_es = fitness[i]
                    best_x_es = pop[i].copy()
            
            if calls_made >= self.budget:
                break
                
            # Sort indices by fitness
            idx = np.argsort(fitness)
            y = pop[idx]
            
            # Update mean
            mean = np.sum(w[:, np.newaxis] * y[:mu], axis=0)
            
            # Update sigma (CSA)
            # y_sigma = (1/sigma) * (mean - mean_old)
            # We need the old mean. Let's store it.
            # This implementation is getting complex. 
            # Let's use a simpler, very robust alternative: **Nelder-Mead** with a good start.
            # Nelder-Mead is often sufficient for 5D BBOB functions and is much less prone to implementation bugs.
            
            break # Exit CMA-ES loop to switch to Nelder-Mead for reliability
            
        # Fallback/Alternative: Use Nelder-Mead for the remaining budget
        # We have best_x from LHS. Let's run Nelder-Mead.
        # Nelder-Mead requires 2*dim initial points.
        # We already have best_x. We can generate neighbors.
        
        # Actually, let's restart the logic. The CMA-ES implementation above is too risky for a one-shot script without testing.
        # A **well-tuned (1+1)-ES with covariance adaptation** is harder.
        # **Nelder-Mead** is the safest bet for a "structural change" that is robust.
        # However, the inspiration says CMA-ES is better.
        # Let's implement a **Simplified CMA-ES** that only updates the step size and direction, 
        # or use **PyCMA** if available? No, standard library + NumPy/SciPy only.
        
        # Let's use **Nelder-Mead** from SciPy for the second phase. It's robust and fast.
        # We have used n_lhs calls. We have self.budget - n_lhs calls left.
        # Nelder-Mead can use up to ~100-200 calls for 5D.
        
        # Construct initial simplex
        x0 = best_x
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = x0
        for i in range(1, self.dim + 1):
            simplex[i] = x0.copy()
            simplex[i][i-1] += 0.1 * (x0[i-1] - (-5)) + 0.1 # Small step
            
        # Ensure simplex is within bounds? Nelder-Mead doesn't handle bounds well.
        # We must clip.
        
        # Let's just use a **Robust (1+1)-ES with larger initial sigma and adaptive bounds**.
        # Wait, the current program IS a (1+1)-ES. The instruction says "MATERIALLY DIFFERENT".
        # LHS + CMA-ES is materially different.
        # I will implement a **Minimal CMA-ES** correctly.
        
        # Reset state for CMA-ES
        mean = best_x.copy()
        C = np.eye(self.dim)
        sigma = 0.5
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        # Re-calculate constants
        lambda_ = 10
        mu = 5
        w = np.arange(1, mu + 1)
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(self.dim) * (1 - 1/(2*self.dim**2) + 1/(12*self.dim**4))
        
        calls_made = n_lhs
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Sample
            Z = np.random.randn(lambda_, self.dim)
            L = np.linalg.cholesky(C)
            pop = mean[None, :] + sigma * (Z @ L)
            
            # Evaluate
            fitness = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                calls_made += 1
                fitness[i] = func(pop[i])
                if fitness[i] < best_f_es:
                    best_f_es = fitness[i]
                    best_x_es = pop[i].copy()
            
            if calls_made >= self.budget:
                break
            
            # Sort
            idx = np.argsort(fitness)
            y = pop[idx]
            
            # Old mean
            mean_old = mean.copy()
            
            # Update mean
            mean = np.sum(w[:, np.newaxis] * y[:mu], axis=0)
            
            # Update C
            # y_sigma = (mean - mean_old) / sigma
            y_sigma = (mean - mean_old) / sigma
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (mean - mean_old)
            
            # Update pc
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - n_lhs + 1)))) < (1 + 2 / self.dim) * chi_n
            if h_sigma:
                pc = (1 - c_c) * pc + c_c * y_sigma
            else:
                pc = (1 - c_c) * pc
            
            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 - delta_h) * C + \
                c_1 * (y_sigma[:, np.newaxis] * y_sigma[np.newaxis, :] - C) + \
                c_mu * np.sum(w[:, np.newaxis] * (y[:mu] - mean_old)[:, np.newaxis] * (y[:mu] - mean_old)[np.newaxis, :], axis=0)
            
            # Ensure C is symmetric and positive definite (numerical stability)
            C = (C + C.T) / 2
            # Add small identity to prevent singularities
            C += 1e-6 * np.eye(self.dim)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Check for stagnation or convergence
            if np.linalg.norm(mean - mean_old) < 1e-8 and sigma < 1e-8:
                break

        return best_x_es, best_f_es

# EVOLVE-BLOCK-END
