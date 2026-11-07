"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned) on [-5,5]^5.
    
    Strategy: Global exploration with adaptive local search.
    - Phase 1: Broad global exploration using multiple diverse starting points
      with short (1+1)-ES runs to escape local minima.
    - Phase 2: Adaptive population-based search to identify promising basins.
    - Phase 3: Intensive local refinement around the best candidates.
    
    Key improvements:
    - Diverse initialization across the domain to cover multiple basins
    - Short exploration runs per start to avoid over-fitting to local minima
    - Adaptive step sizing that responds to landscape ruggedness
    - Final polish phase with very fine steps
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Phase 1: Global exploration with diverse starts
        # Use many short (1+1)-ES runs from different regions
        n_global_starts = 20
        budget_per_global = max(5, (budget // 3) // n_global_starts)
        
        # Generate diverse starting points using stratified sampling
        # This ensures coverage of the entire domain
        starts = []
        for i in range(n_global_starts):
            # Mix of uniform random and structured points
            if i < n_global_starts // 2:
                # Uniform random
                x = rng.uniform(-5, 5, dim)
            else:
                # Structured: grid-like with jitter
                idx = i - n_global_starts // 2
                # Create a pseudo-grid position
                base = -4 + (idx * 8) / max(1, n_global_starts // 2 - 1)
                x = np.full(dim, base) + rng.uniform(-0.5, 0.5, dim)
                x = np.clip(x, -5, 5)
            starts.append(x)
        
        # Evaluate each start with a short local search
        for x_start in starts:
            if calls_used >= budget - budget // 4:
                break
            
            x = x_start.copy()
            fx = func(x)
            calls_used += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Short (1+1)-ES with adaptive sigma
            sigma = 1.0
            for _ in range(budget_per_global):
                if calls_used >= budget - budget // 4:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, 3.0)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8
                    if sigma < 0.1:
                        # Escape mechanism: large jump to potentially new basin
                        sigma = 2.0
                        x = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                        fx = func(x)
                        calls_used += 1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
        
        # Phase 2: Adaptive refinement around best candidates
        # Use multiple restarts from the best point with different sigma values
        if best_x is not None and calls_used < budget - budget // 5:
            # Try several different step sizes from the best point
            sigma_candidates = [0.5, 1.0, 2.0]
            for sigma_init in sigma_candidates:
                if calls_used >= budget - budget // 5:
                    break
                
                x = best_x.copy()
                fx = best_f
                sigma = sigma_init
                remaining = min(30, budget - calls_used - budget // 5)
                
                for _ in range(remaining):
                    if calls_used >= budget - budget // 5:
                        break
                    
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_used += 1
                    
                    if fy < fx:
                        x, fx = y, fy
                        sigma = min(sigma * 1.15, 5.0)
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma *= 0.9
        
        # Phase 3: Final intensive polish
        if best_x is not None and calls_used < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            remaining = budget - calls_used
            
            for _ in range(remaining):
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.1, 1.0)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.95
                    if sigma < 0.05:
                        # Very small perturbations to fine-tune
                        sigma = 0.05
        
        return best_x, best_f
# EVOLVE-BLOCK-END
