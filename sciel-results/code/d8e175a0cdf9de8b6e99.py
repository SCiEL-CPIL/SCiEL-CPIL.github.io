"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical multi-scale stochastic tunneling with (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        # Global best tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            return float(func(x))

        # Hierarchical multi-scale stochastic tunneling
        # Coarse-to-fine random walks with geometric step size reduction
        # This exploits the self-similar structure of Weierstrass functions
        
        # Phase 1: Coarse global search (large steps)
        # Use ~20% of budget for broad exploration
        coarse_budget = max(20, total_budget // 5)
        coarse_steps = 5  # Number of scale levels in coarse phase
        
        current_scale = 4.0  # Initial large step size
        for scale_level in range(coarse_steps):
            scale_step = current_scale * (0.5 ** scale_level)
            # Number of evaluations at this scale
            n_evals = max(3, coarse_budget // coarse_steps)
            
            for _ in range(n_evals):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                # Random walk with current step size
                if best_x is None:
                    x = rng.uniform(-5, 5, dim)
                else:
                    x = best_x + scale_step * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                
                fx = safe_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            current_scale *= 0.5
        
        # Phase 2: Mid-scale exploration
        mid_budget = max(20, total_budget // 4)
        mid_steps = 4
        current_scale = 1.0
        for scale_level in range(mid_steps):
            scale_step = current_scale * (0.5 ** scale_level)
            n_evals = max(3, mid_budget // mid_steps)
            
            for _ in range(n_evals):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                x = best_x + scale_step * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                
                fx = safe_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            current_scale *= 0.5
        
        # Phase 3: Fine (1+1)-ES refinement with 1/5 success rule
        # Use remaining budget for local exploitation
        remaining = total_budget - calls_made
        if remaining > 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with moderate step size
            
            # 1/5 success rule: target success rate of 1/5
            # This is known to be optimal for (1+1)-ES
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = safe_func(y)
                
                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase sigma (1/5 rule: multiply by 1.02 on success)
                    sigma *= 1.02
                else:
                    # Failure
                    # Decrease sigma (1/5 rule: multiply by 0.98 on failure)
                    sigma *= 0.98
                
                # Cap sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # If sigma becomes very small, we've likely converged
                if sigma < 1e-8:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
