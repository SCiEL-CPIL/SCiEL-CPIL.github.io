"""Rewritten program using Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 300s, we stop at 290s
        deadline = start_time + 290.0
        
        # Use a fixed seed for deterministic search trajectory per instance
        rng = np.random.default_rng(12345)
        
        # Initialize at the center of the box
        x = np.zeros(self.dim)
        fx = func(x)
        
        x_best = x.copy()
        fx_best = fx
        
        # SA Parameters
        # High initial temperature to allow large exploratory moves
        T = 10.0
        # Geometric cooling rate
        alpha = 0.99
        # Large initial step size to rapidly explore the [-5, 5]^5 space
        sigma = 5.0
        
        for _ in range(self.budget):
            if time.time() > deadline:
                break
            
            # Generate a neighbor with Gaussian perturbation
            y = x + sigma * rng.standard_normal(self.dim)
            # Clip to the valid domain [-5, 5]
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            
            # Update global best
            if fy < fx_best:
                fx_best = fy
                x_best = y.copy()
            
            # Acceptance criterion
            if fy < fx:
                # Always accept improvements
                x = y
                fx = fy
            else:
                # Metropolis criterion for worse states
                delta_e = fy - fx
                # Avoid division by zero if T becomes very small
                if T > 1e-10:
                    prob = np.exp(-delta_e / T)
                else:
                    prob = 0.0
                
                if rng.random() < prob:
                    x = y
                    fx = fy
            
            # Cool down the temperature
            T *= alpha
            
            # Decay the step size to allow for fine-grained convergence
            # 0.99 decay over 500 steps: 5.0 * 0.99^500 ~ 0.03
            sigma *= 0.99

        return x_best, fx_best
# EVOLVE-BLOCK-END
