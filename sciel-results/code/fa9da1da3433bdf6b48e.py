"""Differential Evolution black-box optimizer for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution (DE) optimizer with adaptive parameters.
    
    Strategy:
    - Use a population-based search (Differential Evolution) to handle the multimodal
      nature of Rastrigin.
    - Use 'best/2/bin' mutation to leverage the current best solution while exploring.
    - Adaptive scaling factor (F) and crossover rate (CR) based on success rates.
    - Strict budget management to ensure all 500 calls are used effectively.
    - Early termination if convergence is detected.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._func = None
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        self._func = func
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = np.zeros(self.dim)
        
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        # Population size typically 5*dim to 10*dim. 
        # With budget 500 and dim 5, we can afford a decent population.
        # Let's try pop_size = 10 + dim (15) or maybe larger for better coverage.
        # 500 calls / 15 pop = ~33 generations.
        # 500 calls / 25 pop = ~20 generations.
        # Let's start with a larger population for exploration.
        pop_size = 20 
        if dim > 10:
            pop_size = max(10, 5 * dim)
            
        # Ensure we have enough budget for at least 2 generations
        while pop_size * 2 > budget:
            pop_size = max(5, pop_size // 2)

        # Initialize population uniformly in [-5, 5]
        # Using a fixed seed for reproducibility within the instance? 
        # The task says "hidden seeded shift", implying the function is deterministic 
        # for a given instance, but we want robustness. 
        # However, using a fixed seed for the optimizer itself is safer to avoid 
        # random bad luck in initialization.
        rng = np.random.default_rng(42)
        
        population = rng.uniform(-5.0, 5.0, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        
        # Evaluate initial population
        for i in range(pop_size):
            if self._call_count >= budget:
                break
            if time.time() > deadline:
                break
            x = population[i]
            # Clip to bounds (though init is within bounds)
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            self._call_count += 1
            
            if not np.isfinite(f_val):
                f_val = 1e300
                
            fitness[i] = f_val
            if f_val < self._best_f:
                self._best_f = f_val
                self._best_x = x_clipped.copy()

        # DE Parameters
        # F: Scaling factor. Start with 0.8, adapt downwards.
        F = 0.8
        CR = 0.5 # Crossover rate
        
        # Track convergence
        no_improve_count = 0
        convergence_threshold = 1e-6
        best_prev = self._best_f

        # Main DE Loop
        while self._call_count < budget and time.time() < deadline:
            # Find best individual
            best_idx = np.argmin(fitness)
            best_x = population[best_idx]
            best_f = fitness[best_idx]
            
            # Update global best
            if best_f < self._best_f:
                self._best_f = best_f
                self._best_x = best_x.copy()
                no_improve_count = 0
            else:
                # Check for convergence
                if abs(self._best_f - best_prev) < convergence_threshold:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                best_prev = self._best_f
                
                # If stuck for too long, perturb F or restart?
                # For simplicity, just continue. DE usually handles this well.
                
            # Generate new population
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            
            for i in range(pop_size):
                if self._call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select two distinct random indices different from i
                r1, r2 = rng.choice(pop_size, size=2, replace=False)
                while r1 == i or r2 == i or r1 == r2:
                    r1, r2 = rng.choice(pop_size, size=2, replace=False)
                
                # Mutation: v = x_best + F * (x_r1 - x_r2)
                # Using best/2/bin
                v = best_x + F * (population[r1] - population[r2])
                
                # Clip to bounds
                v = np.clip(v, -5.0, 5.0)
                
                # Crossover: binomial crossover with CR
                # j_rand: random position for crossover
                j_rand = rng.integers(0, dim)
                trial_x = np.copy(population[i])
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial_x[j] = v[j]
                    
                # Evaluate trial
                trial_x_clipped = np.clip(trial_x, -5.0, 5.0)
                f_trial = func(trial_x_clipped)
                self._call_count += 1
                
                if not np.isfinite(f_trial):
                    f_trial = 1e300
                    
                # Selection
                if f_trial <= fitness[i]:
                    new_population[i] = trial_x
                    new_fitness[i] = f_trial
                    # Success: increase F slightly or keep? 
                    # Standard DE keeps F. Adaptive DE might adjust.
                    # Let's try a simple adaptation: if success, F *= 1.05 (max 1.0)
                    # if fail, F *= 0.95 (min 0.1)
                    # But this is per-individual. Global F adaptation is complex.
                    # Let's stick to fixed F=0.8, CR=0.5 which is robust.
                    # Or maybe adapt CR?
                    # Let's try simple adaptive F based on generation success rate.
                
            # Replace population
            population = new_population
            fitness = new_fitness
            
            # Simple adaptation: If best fitness didn't improve this generation,
            # reduce F to exploit more, or increase CR?
            # Actually, if we are stuck, we might want to explore more (increase F or CR).
            # If we are improving, we might want to exploit (decrease F).
            # Let's monitor global improvement.
            current_best = np.min(fitness)
            if current_best < self._best_f - 1e-10:
                # Improving: exploit a bit more
                F = max(0.5, F * 0.98)
                CR = min(0.9, CR * 1.02)
            else:
                # Stagnating: explore more
                F = min(1.0, F * 1.05)
                CR = max(0.1, CR * 0.95)

            # Early termination check
            if no_improve_count > 10 and F < 0.6:
                # We are stuck in a local minimum. 
                # Option: Random restart of the whole population?
                # Or just break and use remaining budget for a local polish?
                # Given the multimodal nature, a random restart might be risky if we lose the global best.
                # But if we are stuck, we are likely in a local min.
                # Let's do a partial restart: replace worst half with random points.
                worst_half = pop_size // 2
                indices_to_replace = np.argsort(fitness)[-worst_half:]
                for idx in indices_to_replace:
                    if self._call_count >= budget:
                        break
                    new_x = rng.uniform(-5.0, 5.0, size=dim)
                    f_val = func(new_x)
                    self._call_count += 1
                    if not np.isfinite(f_val):
                        f_val = 1e300
                    population[idx] = new_x
                    fitness[idx] = f_val
                    if f_val < self._best_f:
                        self._best_f = f_val
                        self._best_x = new_x.copy()
                no_improve_count = 0 # Reset counter

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
