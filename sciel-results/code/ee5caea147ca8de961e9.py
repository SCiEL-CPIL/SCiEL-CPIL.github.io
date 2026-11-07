"""Optimizer for BBOB f9 (Rosenbrock rotated) using multi-fidelity evolutionary strategy with adaptive step size."""
import numpy as np
import time


class Optimizer:
    """Multi-fidelity population-based evolutionary optimizer with adaptive step sizing."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_made = 0

        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def estimate_gradient(x, eps=1e-5):
            """Central finite difference gradient."""
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                xp = np.clip(xp, -5, 5)
                xm = np.clip(xm, -5, 5)
                fp = safe_func(xp)
                fm = safe_func(xm)
                grad[i] = (fp - fm) / (2 * eps)
            return grad

        def gradient_descent(x0, fx0, max_iters, eps=1e-5, lr_init=0.05):
            """Gradient descent with backtracking line search."""
            nonlocal calls_made
            x = x0.copy()
            fx = fx0
            iters = 0

            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if calls_made >= budget:
                    break

                # Estimate gradient (2*dim function calls)
                if calls_made + 2 * dim > budget:
                    break

                grad = estimate_gradient(x, eps)

                # Check if we're at a minimum
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-8:
                    break

                # Backtracking line search
                step_dir = -grad / gnorm  # normalize direction
                alpha = lr_init
                improved = False

                for _ in range(25):
                    if calls_made >= budget:
                        break
                    x_new = np.clip(x + alpha * step_dir, -5, 5)
                    fx_new = safe_func(x_new)

                    if fx_new < fx:
                        x, fx = x_new, fx_new
                        improved = True
                        alpha *= 1.15
                        break
                    else:
                        alpha *= 0.5

                if not improved:
                    if alpha < 1e-10:
                        break

                iters += 1

            return x, fx

        # Phase 1: Multi-fidelity evolutionary strategy
        # Population size, number of generations, and adaptive step sizing
        pop_size = 20
        n_generations = 25  # 20 * 25 = 500 evaluations max
        
        # Initialize population
        population = []
        fitnesses = []
        
        # First point at origin
        x0 = np.zeros(dim)
        f0 = safe_func(x0)
        population.append(x0.copy())
        fitnesses.append(f0)
        
        # Initialize rest of population with diverse points
        for i in range(1, pop_size):
            if calls_made >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            f = safe_func(x)
            population.append(x.copy())
            fitnesses.append(f)
        
        # Sort by fitness
        indices = np.argsort(fitnesses)
        population = [population[i] for i in indices]
        fitnesses = [fitnesses[i] for i in indices]
        
        # Track best and improvement rate
        global_best_f = fitnesses[0]
        global_best_x = population[0].copy()
        
        # Adaptive step size: starts large, shrinks based on improvement
        step_size = 1.5  # initial step size
        min_step_size = 0.01
        step_decay = 0.95  # multiplicative decay when no improvement
        
        for gen in range(n_generations):
            if time.time() > deadline - 10:
                break
            if calls_made >= budget:
                break
            
            # Select parents: top half of population
            n_parents = max(5, pop_size // 2)
            parents = population[:n_parents]
            
            # Generate offspring
            offspring = []
            offspring_fits = []
            
            n_offspring = pop_size - len(offspring)  # fill up to pop_size
            
            for _ in range(n_offspring):
                if calls_made >= budget:
                    break
                
                # Pick two parents (tournament selection)
                p1_idx = rng.integers(0, n_parents)
                p2_idx = rng.integers(0, n_parents)
                p1 = parents[p1_idx]
                p2 = parents[p2_idx]
                
                # Crossover: blend with random weights
                w = rng.uniform(0.3, 0.7)
                x_child = w * p1 + (1 - w) * p2
                
                # Mutation: add Gaussian noise scaled by adaptive step size
                noise = rng.normal(0, step_size, dim)
                x_child = x_child + noise
                
                # Clip to bounds
                x_child = np.clip(x_child, -5, 5)
                
                # Evaluate
                f_child = safe_func(x_child)
                offspring.append(x_child.copy())
                offspring_fits.append(f_child)
            
            if not offspring:
                break
            
            # Combine parents and offspring
            combined_pop = population + offspring
            combined_fit = fitnesses + offspring_fits
            
            # Sort and keep best pop_size
            all_indices = np.argsort(combined_fit)
            population = [combined_pop[i] for i in all_indices[:pop_size]]
            fitnesses = [combined_fit[i] for i in all_indices[:pop_size]]
            
            # Update global best
            if fitnesses[0] < global_best_f:
                global_best_f = fitnesses[0]
                global_best_x = population[0].copy()
                # Reset step decay on improvement
                step_size = max(step_size, 0.5)
            else:
                # No improvement, decay step size
                step_size *= step_decay
                step_size = max(step_size, min_step_size)
            
            # Early termination if step size too small
            if step_size < 0.02:
                break
        
        # Phase 2: Gradient descent refinement from best solution
        if best_x is not None and calls_made + 2 * dim < budget and time.time() < deadline - 15:
            # Coarse gradient descent
            x, fx = gradient_descent(best_x, best_f, max_iters=30, eps=1e-4, lr_init=0.1)
        
        # Phase 3: Fine gradient descent
        if best_x is not None and calls_made + 2 * dim < budget and time.time() < deadline - 10:
            x, fx = gradient_descent(best_x, best_f, max_iters=30, eps=1e-6, lr_init=0.01)
        
        # Phase 4: Ultra-fine gradient descent
        if best_x is not None and calls_made + 2 * dim < budget and time.time() < deadline - 5:
            x, fx = gradient_descent(best_x, best_f, max_iters=20, eps=1e-7, lr_init=0.005)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f
