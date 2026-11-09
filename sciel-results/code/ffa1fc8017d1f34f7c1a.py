"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES optimizer for BBOB f9."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initialize at origin
        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_used = 1

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used
            fx = func(x)
            calls_used += 1
            return fx

        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(n))
        # Step size sigma
        sigma = 2.0
        min_sigma = 1e-10
        max_sigma = 10.0
        
        # Restart logic
        max_iterations_per_run = 200  # Roughly budget/lam iterations
        min_improvement = 1e-5
        no_improve_count = 0
        threshold_no_improve = 50 # Iterations without improvement to trigger restart

        # Main loop
        while calls_used < budget and not check_deadline():
            # Start a new CMA-ES run
            # Initialize center
            if calls_used == 1:
                center = np.zeros(n)
                center_f = best_f
            else:
                # Perturb best found so far or random restart
                if rng.random() < 0.5:
                    center = best_x + 0.5 * rng.standard_normal(n)
                else:
                    center = 4.0 * (rng.random(n) - 0.5)
                center = np.clip(center, -5, 5)
                center_f = do_call(center)
                if center_f < best_f:
                    best_f = center_f
                    best_x = center.copy()

            sigma = 2.0
            no_improve_count = 0
            prev_best_f = best_f

            # CMA-ES inner loop
            while calls_used < budget and not check_deadline():
                # Generate population
                # Simple diagonal covariance for speed, relying on step size adaptation
                # This is effectively a (mu+lambda)-ES with adaptive sigma
                # We use a simple adaptation: if median improvement is good, increase sigma, else decrease
                
                # Sample population
                pop = []
                pop_f = []
                
                for i in range(lam):
                    # Standard normal perturbation
                    z = rng.standard_normal(n)
                    x = center + sigma * z
                    x = np.clip(x, -5, 5)
                    
                    if calls_used >= budget or check_deadline():
                        break
                    
                    fx = do_call(x)
                    pop.append(x)
                    pop_f.append(fx)

                if not pop:
                    break

                pop_f = np.array(pop_f)
                pop = np.array(pop)
                
                # Sort by fitness
                idx = np.argsort(pop_f)
                pop = pop[idx]
                pop_f = pop_f[idx]
                
                # Update best
                if pop_f[0] < best_f:
                    best_f = pop_f[0]
                    best_x = pop[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Update center (mean of best half)
                mu = max(1, lam // 2)
                center = np.mean(pop[:mu], axis=0)
                
                # Adapt sigma
                # Simple heuristic: if best improved, keep sigma or increase slightly
                # if many points improved, increase sigma
                # if no improvement, decrease sigma
                if no_improve_count == 0:
                    # Some improvement
                    sigma = min(sigma * 1.1, max_sigma)
                else:
                    # No improvement
                    sigma = max(sigma * 0.8, min_sigma)

                # Check for stagnation to restart
                if no_improve_count > threshold_no_improve:
                    break
                
                # Check for convergence (small sigma)
                if sigma < min_sigma * 10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
