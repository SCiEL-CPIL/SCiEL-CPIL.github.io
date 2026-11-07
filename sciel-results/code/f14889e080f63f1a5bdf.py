"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Population-based Random Search with Adaptive Step Size (1+lambda)-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Use a larger population to better sample the space
        # Budget is 500. If pop_size is 10, we can do ~50 generations.
        # If pop_size is 50, we can do ~10 generations.
        # Random search benefits from diversity, but we need to converge.
        # Let's try a moderate population size.
        self.pop_size = 10 
        self.step_size = 2.0  # Initial step size, covering a good chunk of [-5, 5]
        self.min_step = 1e-6
        self.max_step = 10.0
        # Success rule parameters
        self.success_rate = 0.2  # Target success rate for step size adaptation

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        rng = np.random.default_rng(42)
        
        # Initialize mean in the center of the box
        mean = np.zeros(self.dim)
        
        # Evaluate the initial point
        # We must make at least 1 call.
        # Clip to box [-5, 5]
        x0 = np.clip(mean, -5, 5)
        f0 = func(x0)
        
        best_x = x0.copy()
        best_f = f0
        call_count = 1
        
        # We will run until budget is exhausted or time limit reached
        # Each generation uses pop_size calls.
        # Total calls = 1 (initial) + generations * pop_size <= budget
        
        while call_count + self.pop_size <= self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            # Sample from normal distribution centered at mean
            z = rng.standard_normal((self.pop_size, self.dim))
            pop = mean + self.step_size * z
            
            # Evaluate population
            fvals = []
            for i in range(self.pop_size):
                if time.time() > deadline:
                    break
                xi = np.clip(pop[i], -5, 5)
                fi = func(xi)
                fvals.append((fi, i))
                call_count += 1
            
            if not fvals:
                break
            
            # Sort by fitness
            fvals.sort(key=lambda x: x[0])
            
            # Update best
            if fvals[0][0] < best_f:
                best_f = fvals[0][0]
                best_x = np.clip(pop[fvals[0][1]], -5, 5)
            
            # Calculate success rate: fraction of offspring better than parent (mean)
            # In (1+lambda)-ES, we compare offspring to the current mean (which acts as parent)
            # However, standard (1+lambda)-ES updates mean to the best offspring.
            # Here we use a weighted mean update or simply move towards the best.
            
            # For step size control, we need to know if the step was too big or too small.
            # A simple heuristic: if the best offspring is significantly better, increase step? 
            # No, usually if many offspring are better, step might be too small? 
            # Actually, in (1+1)-ES with 1/5 rule:
            # If success rate > 0.1-0.2, increase step.
            # If success rate < 0.1-0.2, decrease step.
            
            # Let's use the standard 1/5 rule logic adapted for population.
            # Count how many offspring have f(x_i) < f(mean)
            # Note: f(mean) is not necessarily f0 anymore if mean moved. 
            # We should evaluate the current mean's fitness if it changed? 
            # No, that wastes budget. 
            # Standard approach: compare offspring to the previous best or previous mean.
            # Let's compare to the previous mean's fitness. 
            # We don't have f(mean) explicitly stored if mean != x0.
            # Let's store f_mean.
            
            # Correction: We need f_mean for the success rule.
            # If we move the mean, we don't evaluate it again.
            # We can approximate f(mean) by the previous best_f? No.
            # Let's stick to a simpler adaptive scheme or evaluate the mean occasionally?
            # Evaluating mean costs 1 call.
            # Alternative: Use the median of offspring to estimate local gradient/step size?
            
            # Let's use a different adaptation: 
            # If the best offspring is much better than the worst, the step size is good.
            # If all offspring are similar, step size might be too small or we are in a flat area.
            # If best is much better than mean, step might be too small?
            
            # Let's try the (1+lambda)-ES with a simple step size adaptation:
            # If the best offspring is better than the current best, we are making progress.
            # We will move the mean towards the best offspring.
            
            # Update Mean: Move towards the best offspring
            # A simple step: mean = best_offspring
            # This is aggressive. 
            # Or: mean = mean + step_size * (best_offspring - mean) / step_size = best_offspring
            # Basically, greedy hill climbing with random kicks.
            
            # Let's use a weighted mean of the top half offspring to be more robust.
            mu = self.pop_size // 2
            if mu > 0:
                top_indices = [fvals[i][1] for i in range(mu)]
                # Weights: linear
                weights = np.arange(mu, 0, -1, dtype=float)
                weights /= weights.sum()
                
                # Calculate weighted sum of positions
                sum_pos = np.zeros(self.dim)
                for w, idx in zip(weights, top_indices):
                    sum_pos += w * pop[idx]
                
                new_mean = sum_pos
                
                # Step size adaptation
                # Compare best offspring to the mean (approximated by previous mean or best_f?)
                # Let's use a heuristic: 
                # If the improvement (best_f - best_offspring_f) is large relative to step size, increase step?
                # Or use the standard 1/5 rule with a proxy.
                
                # Proxy: Fraction of offspring better than the *previous best*?
                # If many are better, we are in a promising area, maybe increase step to explore more?
                # Or if we are converging, decrease step.
                
                # Let's try: 
                # If best_offspring < best_f, we improved. 
                # Let's keep step size constant initially, or adapt based on variance of offspring.
                
                # Simple Adaptation:
                # If the spread of offspring is large, keep step.
                # If the spread is small, increase step?
                
                # Let's implement the 1/5 rule properly by tracking f_mean.
                # To save calls, we only evaluate the new mean if we decide to keep it?
                # No, we need f_mean to know if the step was good.
                # Let's evaluate the new mean once per generation? That costs 1 call.
                # Budget: 500. Pop: 10. Generations: 50.
                # Cost per gen: 10 (offspring) + 1 (mean) = 11.
                # 500 / 11 ~ 45 generations.
                # This is feasible.
                
                # But wait, the task says "Every call is recorded".
                # If we evaluate the mean, that's a call.
                
                # Let's refine: 
                # 1. Generate 10 offspring.
                # 2. Evaluate 10 offspring.
                # 3. Pick best 5, compute new mean.
                # 4. Evaluate new mean? 
                #    If we do, we know if the move was good.
                #    If f(new_mean) < f(old_mean), success.
                #    But we don't have f(old_mean) stored if we didn't evaluate it last time.
                
                # Let's store f_mean.
                # Init: f_mean = f0.
                # Loop:
                #   Generate offspring from mean.
                #   Evaluate offspring.
                #   Compute new_mean.
                #   Evaluate new_mean (1 call).
                #   If f(new_mean) < f(mean): success = True, else False.
                #   Adapt step size based on success rate over last N generations.
                #   Update mean = new_mean, f_mean = f(new_mean).
                #   Update best if f(new_mean) < best_f or any offspring < best_f.
                
                # This uses pop_size + 1 calls per generation.
                # With budget 500 and pop 10, we get ~45 generations.
                # This is a solid (1+lambda)-ES.
                
                # Let's implement this.
                
                # For now, in this loop iteration, we have already generated and evaluated offspring.
                # We need to integrate the mean evaluation.
                
                # Break out of the simple loop and rewrite the loop structure?
                # The current code structure evaluates offspring, then updates.
                # I will rewrite the loop inside __call__ to be cleaner.
                
                # Since I am in the middle of the thought process, I will just output the code with the cleaner structure.
                
                break # Break to rewrite loop structure in final code
                
            else:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
