"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        # Population size: small for high dimensionality? 5D is small.
        # N=10 is reasonable for 300 budget.
        N = 10
        
        # Initialize population
        # Spread initial points to cover the box
        pop = rng.uniform(-5, 5, (N, dim))
        fvals = np.array([func(x) for x in pop])
        calls = N
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # DE parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        # Adaptive step for final phase
        sigma = 1.0

        # Main loop
        while calls < budget:
            # Time check
            if time.time() > deadline - 5.0:
                break

            # Generate trial vectors for each member
            for i in range(N):
                if calls >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                # Select two distinct random indices different from i
                r1, r2 = np.random.choice([j for j in range(N) if j != i], 2, replace=False)
                
                # Mutation: x_r1 + F * (x_r2 - x_i)
                mutant = pop[r1] + F * (pop[r2] - pop[i])
                
                # Crossover
                trial = np.copy(pop[i])
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial < fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    
                    if f_trial < best_f:
                        best_x = trial.copy()
                        best_f = f_trial

            # Optional: If population stagnates (no improvement in last few iterations),
            # we could restart some members, but DE usually handles this.
            # However, for multimodal functions like Gallagher, we might get stuck.
            # A simple strategy: if no improvement in last N*N calls, reduce F or randomize a member.
            # Let's add a simple stagnation check: if best_f hasn't changed in last 50 calls, 
            # replace the worst member with a random one.
            
            # To keep it simple and robust within time, let's just run DE until budget.
            # DE is generally good for these 5D problems.

        # Final Phase: Fine-tuning the best solution
        # Use remaining budget for a local search around best_x
        if calls < budget:
            x = best_x
            fx = best_f
            sigma = 0.5
            
            # Estimate local scale based on how far we are from "good" values?
            # Hard to know. Just use adaptive sigma.
            
            # Track success rate to adapt sigma
            success_count = 0
            total_count = 0
            no_improve_streak = 0
            max_no_improve = 20
            
            while calls < budget:
                if time.time() > deadline - 1.0:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                total_count += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    no_improve_streak = 0
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx
                else:
                    no_improve_streak += 1

                # Adapt sigma
                # Simple adaptation: if success rate is around 0.2-0.5, keep sigma.
                # If too many successes, increase. If too few, decrease.
                if total_count % 10 == 0:
                    success_rate = success_count / total_count
                    if success_rate > 0.5:
                        sigma *= 1.2
                    elif success_rate < 0.2:
                        sigma *= 0.8
                    success_count = 0
                    total_count = 0

                if sigma < 1e-10:
                    break

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            fx = func(x)
            calls += 1
            best_x = x
            best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
