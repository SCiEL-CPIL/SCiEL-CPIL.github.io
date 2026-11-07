"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned) on [-5,5]^5.
    
    Strategy: Multi-start with adaptive local search.
    - Uses a population-based approach with multiple restarts.
    - Each restart uses a (1+1)-ES with adaptive step size.
    - Global exploration via random restarts in different regions.
    - Exploits the fact that f18 can have many local minima, so we need
      to explore broadly before converging locally.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Reserve some budget for global exploration
        # Split budget into phases: exploration, exploitation, final polish
        # Phase 1: Broad exploration with multiple random starts
        # Phase 2: Local refinement around best points
        # Phase 3: Final polish with small steps
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Number of restarts in exploration phase
        n_restarts = 10
        budget_per_restart = max(10, (budget // 2) // n_restarts)
        
        # Phase 1: Multi-start exploration
        for r in range(n_restarts):
            if calls_used >= budget - 50:  # Keep some budget for final polish
                break
                
            # Random start point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Local (1+1)-ES from this point
            sigma = 1.5
            local_budget = min(budget_per_restart, budget - calls_used - 50)
            
            for _ in range(local_budget):
                if calls_used >= budget - 50:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
            
            # Early stopping if we've made good progress
            if best_f < -50:
                break
        
        # Phase 2: Local refinement around the best point found
        if best_x is not None and calls_used < budget - 20:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = budget - calls_used - 10
            
            for _ in range(remaining):
                if calls_used >= budget - 5:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    if sigma < 0.01:
                        # Try a larger jump to escape potential local min
                        sigma = 0.5
                        x = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                        fx = func(x)
                        calls_used += 1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
        
        # Phase 3: Final polish with very small steps
        if best_x is not None and calls_used < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = budget - calls_used
            
            for _ in range(remaining):
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.95
        
        return best_x, best_f
# EVOLVE-BLOCK-END
