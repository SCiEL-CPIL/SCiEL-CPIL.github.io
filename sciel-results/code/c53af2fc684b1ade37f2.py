"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """(1+lambda)-ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(123)
        n = self.dim
        lam = 20  # population size per generation
        
        # Initial point: try center first, then random
        x = np.zeros(n)
        fx = func(x)
        
        # Adaptive step size
        sigma = 2.0
        success_count = 0
        total_count = 0
        no_improve = 0
        best_x = x.copy()
        best_fx = fx
        
        calls_left = self.budget - 1
        generation = 0
        
        while calls_left > 0:
            # Generate offspring
            offspring = []
            for i in range(lam):
                if calls_left <= 0:
                    break
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                offspring.append((fy, y))
                calls_left -= 1
            
            if not offspring:
                break
            
            # Find best offspring
            offspring.sort(key=lambda t: t[0])
            best_fy, best_y = offspring[0]
            
            # Update best
            if best_fy < best_fx:
                best_fx = best_fy
                best_x = best_y.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Update mean: move toward best offspring
            x = best_y
            
            # Adapt sigma
            total_count += 1
            if best_fy < fx:
                success_count += 1
                sigma *= 1.1
            else:
                sigma *= 0.95
            
            sigma = np.clip(sigma, 1e-5, 5.0)
            fx = best_fy
            
            # Restart if stuck
            if no_improve > 50 and calls_left > 20:
                # Try a random restart
                x = rng.uniform(-5, 5, n)
                fx = func(x)
                calls_left -= 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                sigma = 2.0
                no_improve = 0
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
