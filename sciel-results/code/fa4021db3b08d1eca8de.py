"""Rewritten optimizer for BBOB-style f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based evolution strategy with restarts for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # Track global best
        best_x = None
        best_f = np.inf

        # Multi-restart with population-based ES
        # For multimodal f21, restarts are crucial to escape local optima
        # We'll use a (mu+lambda) ES with adaptive sigma
        
        # Phase 1: Broad exploration with multiple independent runs
        # Each run gets a portion of the budget
        
        n_restarts = 8
        budget_per_restart = max(10, budget // n_restarts)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with a different random point for each restart
            # First restart at center, others at random locations
            if restart == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)
            
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Population-based (1+1)-ES with larger initial step
            # For f21, peaks are separated, so we need good local search
            pop_size = 10
            sigma = 1.5
            mu = 5  # number of parents
            lam = 15  # number of offspring
            
            # Adaptive step size based on success rate
            success_count = 0
            total_trials = 0
            
            remaining = budget_per_restart - 1
            
            for step in range(remaining):
                if time.time() > deadline:
                    break
                
                # Generate offspring
                offspring = []
                for j in range(lam):
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    offspring.append((y, None))
                
                # Evaluate offspring
                for j, (y, _) in enumerate(offspring):
                    fy = func(y)
                    offspring[j] = (y, fy)
                    total_trials += 1
                    
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                
                # Sort offspring by fitness
                offspring.sort(key=lambda t: t[1])
                
                # Update parent: take the best offspring
                new_x, new_fx = offspring[0]
                
                # Adapt sigma based on success
                if new_fx < fx:
                    sigma *= 1.2
                    success_count += 1
                else:
                    sigma *= 0.85
                
                # Keep the best as new parent
                x, fx = new_x, new_fx
                
                # If we're stuck (no improvement for a while), try a bigger jump
                if step > 0 and step % 50 == 0:
                    # Check if we've been stuck
                    recent_improvements = 0
                    # Simple heuristic: if sigma is very small, try a restart-like jump
                    if sigma < 0.01:
                        # Try a few random exploratory steps
                        for _ in range(3):
                            y = x + rng.uniform(-2, 2, dim)
                            y = np.clip(y, -5, 5)
                            fy = func(y)
                            if fy < fx:
                                x, fx = y, fy
                                if fy < best_f:
                                    best_f = fy
                                    best_x = y.copy()
                                sigma = 0.5
                                break
                        sigma = max(sigma, 0.1)
        
        # Phase 2: Local refinement around the best point found
        if time.time() < deadline and best_x is not None:
            # Do a fine-grained search around best_x
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            
            # Use a smaller population for refinement
            lam = 10
            steps_left = min(50, max(10, int((deadline - time.time()) / 0.001) // 100))
            
            for _ in range(steps_left):
                if time.time() > deadline:
                    break
                
                offspring = []
                for j in range(lam):
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                
                best_y = None
                best_fy = np.inf
                for y in offspring:
                    fy = func(y)
                    if fy < best_fy:
                        best_fy = fy
                        best_y = y
                
                if best_fy < fx:
                    x = best_y
                    fx = best_fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
        
        return best_x, best_f
# EVOLVE-BLOCK-END
