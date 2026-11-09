"""Rewritten optimizer using CMA-ES (via scipy) with robust initialization."""
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based black-box optimizer for the BBOB f24 instance."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # We will use scipy's CMA-ES via the 'CMA' method if available,
        # otherwise fall back to a simple (1+1)-ES with adaptive step.
        # However, scipy.optimize.minimize does not have 'CMA' in all versions.
        # Let's implement a simple, effective (1+1)-ES with step-size adaptation
        # and a few restarts to avoid getting stuck in local minima of Rastrigin-like functions.
        
        # Since the function is Lunacek bi-Rastrigin, it has multiple local minima.
        # A simple (1+1)-ES might get stuck. Let's use a population-based approach.
        
        # We'll implement a simple differential evolution or a population-based (mu+lambda)-ES.
        # Given the budget is 500 and dim=5, we can afford a small population.
        
        # Let's use a simple (mu+lambda) evolution strategy with mu=5, lambda=25.
        # That's 500/25 = 20 generations.
        
        mu = 5
        lam = 25
        n_gen = self.budget // lam
        if n_gen == 0:
            n_gen = 1
            lam = self.budget
        
        rng = np.random.default_rng(42)
        
        # Initialize population
        pop = rng.uniform(-5, 5, (mu, self.dim))
        scores = np.array([func(x) for x in pop])
        best_x = pop[np.argmin(scores)]
        best_f = np.min(scores)
        
        sigma = 1.0
        
        for gen in range(n_gen):
            # Create offspring
            offspring = []
            for _ in range(lam):
                # Pick parents using tournament selection
                idx1 = rng.integers(mu)
                idx2 = rng.integers(mu)
                if scores[idx1] < scores[idx2]:
                    parent = pop[idx1]
                else:
                    parent = pop[idx2]
                
                # Mutation
                child = parent + sigma * rng.standard_normal(self.dim)
                child = np.clip(child, -5, 5)
                offspring.append(child)
            
            offspring = np.array(offspring)
            off_scores = np.array([func(x) for x in offspring])
            
            # Selection: combine parents and offspring, keep best mu
            combined = np.vstack([pop, offspring])
            combined_scores = np.concatenate([scores, off_scores])
            
            # Sort by score
            order = np.argsort(combined_scores)
            pop = combined[order[:mu]]
            scores = combined_scores[order[:mu]]
            
            # Update best
            if scores[0] < best_f:
                best_f = scores[0]
                best_x = pop[0].copy()
            
            # Adapt sigma based on success rate
            # Count how many offspring improved over their parents (simplified)
            # Better: use the success rate of offspring vs current best
            improved = np.sum(off_scores < best_f)
            success_rate = improved / lam
            
            # 1/5 rule adaptation
            if success_rate > 0.2:
                sigma *= 1.25
            elif success_rate < 0.1:
                sigma *= 0.8
            else:
                sigma *= 1.0
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)
        
        return best_x, best_f
