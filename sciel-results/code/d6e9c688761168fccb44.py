"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES style optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Coarse exploration with multiple restarts
        # Use a population-based approach for better basin finding
        
        # First, do a quick random search to get a sense of the landscape
        n_init = 20
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            x = np.clip(x, -5, 5)
            f = func(x)
            update_best(x, f)

        # Phase 2: Local refinement from the best point using adaptive step size
        # Use a (1+lambda)-ES with population for better exploration
        
        if time.time() > deadline:
            return best_x, best_f

        # CMA-ES-like approach with population
        # Start from the best point found
        x_center = best_x.copy() if best_x is not None else rng.uniform(-4, 4, dim)
        
        # Adaptive population size
        pop_size = 8
        sigma = 1.5
        
        # Track success rate for step size adaptation
        successes = 0
        total_trials = 0
        sigma_history = []
        
        calls_made = n_init
        
        while calls_made < budget and time.time() < deadline:
            # Generate population
            improvements = 0
            for j in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Gaussian perturbation
                step = sigma * rng.standard_normal(dim)
                y = x_center + step
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < best_f:
                    improvements += 1
                    update_best(y, fy)
                
                # Track for step size adaptation
                total_trials += 1
                if fy < func(x_center) if hasattr(self, '_last_center_f') else True:
                    # Simplified: just track if it improved over center
                    pass

            # Adapt step size based on success rate
            # Use 1/5 rule approximation
            success_rate = improvements / pop_size if pop_size > 0 else 0
            
            if success_rate > 0.4:
                sigma *= 1.2
            elif success_rate < 0.2:
                sigma *= 0.8
            else:
                sigma *= 0.95
            
            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 5.0)
            
            # Update center: move toward best point
            if best_x is not None:
                x_center = 0.7 * x_center + 0.3 * best_x
                x_center = np.clip(x_center, -5, 5)
            
            # If we've made good progress, tighten the search
            if improvements == 0 and total_trials > 100:
                # Reduce population and sigma for fine-grained search
                pop_size = max(4, pop_size - 1)
                sigma *= 0.7
                total_trials = 0

        # Phase 3: Final fine-tuning from the best point
        if time.time() < deadline and calls_made < budget:
            x = best_x.copy() if best_x is not None else x_center
            fx = best_f if best_f < np.inf else func(x)
            fine_sigma = 0.1
            
            while calls_made < budget and time.time() < deadline:
                y = x + fine_sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    fine_sigma *= 1.1
                    update_best(x, fx)
                else:
                    fine_sigma *= 0.85
                
                if fine_sigma < 0.001:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
