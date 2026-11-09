"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import cma

class Optimizer:
    """CMA-ES based optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES configuration
        # Start at the center of the box, which is often a reasonable prior
        x0 = np.zeros(n)
        
        # Initial sigma: The box is [-5, 5]^5, so width is 10. 
        # A sigma of 2.0 covers a significant portion of the space initially.
        sigma0 = 2.0
        
        # CMA-ES parameters
        # Default population size is 4 + floor(3*log(n)) = 4 + 3 = 7 for n=5
        # We can increase population for better exploration if budget allows, 
        # but standard is often robust. Let's use standard or slightly larger.
        # With 500 budget, and pop size 7, we get ~70 generations.
        # Standard CMA-ES handles this well.
        
        opts = {
            'maxfevals': budget, 
            'seed': 42,
            'verbose': -1,
            'bounds': [-5.0, 5.0],
            'maxiter': 1000
        }
        
        # Create the problem
        problem = cma.FitFunctionEvaluation(func, x0, sigma0, opts)
        
        # Run CMA-ES
        # es.optimize handles the loop and termination criteria
        # It will stop when maxfevals is reached or convergence criteria are met
        try:
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            while not es.stop() and es.countiter < budget:
                # Get next solutions
                solutions = es.ask()
                # Evaluate solutions
                # Note: es.tell expects a list of (x, f) tuples or just f values if x is known
                # Here we evaluate func for each x
                values = [func(x) for x in solutions]
                
                # Check time deadline inside the loop to be safe, 
                # although cma is fast, we must respect the hard kill.
                if time.time() > deadline:
                    break
                    
                # Tell CMA-ES the results
                es.tell(solutions, values)
                
        except Exception:
            # If something goes wrong, we fall back to whatever best we have
            # but CMA-ES is very robust.
            pass
            
        # Retrieve the best result
        # es.result.xbest is the best x found
        # es.result.fbest is the best f found
        if hasattr(es, 'result') and es.result.xbest is not None:
            best_x = es.result.xbest
            best_f = es.result.fbest
        else:
            # Fallback if CMA-ES failed completely (unlikely)
            best_x = x0
            best_f = func(x0)

        return best_x, best_f
# EVOLVE-BLOCK-END
