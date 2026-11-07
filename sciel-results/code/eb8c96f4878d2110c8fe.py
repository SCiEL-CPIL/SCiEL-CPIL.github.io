import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        start_time = time.time()
        # Safety margin: 330s limit, use 320s
        deadline = start_time + 320.0

        def wrapper(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            try:
                fx = func(x)
            except Exception:
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # (1+1)-ES with adaptive step size
        # Weierstrass is multimodal and non-smooth, so simple ES with restarts is often robust.
        # However, the directive asks for a STRUCTURAL change from CMA-ES.
        # The "inspiration" suggests (1+1)-ES was effective. Let's implement a robust (1+1)-ES
        # with random restarts to handle multimodality.
        
        rng = np.random.default_rng(12345)
        
        # Strategy: 
        # 1. Use (1+1)-ES.
        # 2. If no improvement for a certain number of steps, restart from a random point or best point with larger step.
        # 3. Use a mix of global exploration (large steps) and local exploitation (small steps).
        
        def es_1plus1(start_x, sigma0, budget):
            nonlocal calls_made, best_f, best_x, deadline
            
            x = start_x.copy()
            fx = wrapper(x)
            sigma = sigma0
            no_improve = 0
            max_no_improve = 20  # Restart if stuck for 20 steps
            
            while calls_made < budget and time.time() < deadline:
                # Generate offspring
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = wrapper(y)
                
                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    no_improve = 0
                    # Slightly increase sigma to continue exploring? Or keep it?
                    # For (1+1)-ES, usually sigma is adapted. 
                    # Simple adaptation: increase if success, decrease if fail.
                    sigma *= 1.1
                else:
                    # Failure
                    no_improve += 1
                    sigma /= 1.1
                    
                    if no_improve > max_no_improve:
                        # Restart
                        # 50% chance to restart from best global, 50% from random
                        if rng.random() < 0.5 and best_f < fx:
                            x = best_x.copy()
                            fx = best_f
                        else:
                            x = rng.uniform(-4, 4, size=dim)
                            fx = wrapper(x)
                        sigma = 1.0  # Reset sigma
                        no_improve = 0
                        max_no_improve = 20 # Reset patience
                        
                # Safety check
                if time.time() > deadline - 5:
                    break

            return x, fx

        # Phase 1: Global search with larger initial sigma and frequent restarts
        # We will run (1+1)-ES for the entire budget, but the restart logic handles the global/local switch.
        
        # Initial point: 0 or random?
        # Since minimizer is in [-4, 4], 0 is a reasonable start.
        start_x = np.zeros(dim)
        sigma0 = 1.5
        
        try:
            es_1plus1(start_x, sigma0, total_budget)
        except TimeoutError:
            pass
        except Exception:
            pass

        return best_x, best_f
