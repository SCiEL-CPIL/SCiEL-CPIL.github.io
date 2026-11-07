"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer:
    Phase 1: (1+1)-ES with 1/5 success rule for robust global exploration.
    Phase 2: Nelder-Mead simplex method for rapid local convergence.
    
    This approach addresses the hidden shift/rotation by starting with a large 
    step size that covers the box, adapting it based on success rate, and then 
    switching to a local method once the step size shrinks sufficiently, 
    ensuring fast convergence in the final stage to maximize AOCC.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Reserve some budget for local refinement
        self.es_budget = int(budget * 0.7) 
        self.local_budget = budget - self.es_budget

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # --- Phase 1: (1+1)-ES ---
        
        # Initialize at center
        x = np.zeros(self.dim)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        calls = 1
        
        # Initial step size: large enough to explore [-5, 5]
        # The box is [-5, 5], so distance from center to corner is 5*sqrt(5) ~ 11.18
        # A step of 2.0 allows covering a good portion in a few steps if direction is right.
        step_size = 2.0
        min_step = 1e-4
        max_step = 10.0
        
        # 1/5 success rule parameters
        # If success rate > 0.2, increase step. If < 0.2, decrease.
        # We use a simple cumulative success counter over a window.
        window_size = 10
        successes = 0
        trials = 0
        
        while calls < self.es_budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            z = rng.standard_normal(self.dim)
            x_offspring = x + step_size * z
            
            # Clip to box
            x_offspring = np.clip(x_offspring, -5.0, 5.0)
            
            f_offspring = func(x_offspring)
            calls += 1
            
            # Update best
            if f_offspring < best_f:
                best_f = f_offspring
                best_x = x_offspring.copy()
            
            # Determine success
            # Success if offspring is better than parent
            if f_offspring < f_x:
                successes += 1
                # Accept offspring as new parent
                x = x_offspring
                f_x = f_offspring
            # Else, keep parent as is (x, f_x unchanged)
            
            trials += 1
            
            # Adapt step size every 'window_size' trials
            if trials >= window_size:
                success_rate = successes / trials
                # 1/5 rule: target success rate ~ 0.2
                # If success rate is high, step is too small -> increase
                # If success rate is low, step is too large -> decrease
                if success_rate > 0.25:
                    step_size *= 1.2
                elif success_rate < 0.15:
                    step_size /= 1.2
                
                # Clamp step size
                step_size = np.clip(step_size, min_step, max_step)
                
                # Reset counters
                successes = 0
                trials = 0
            
            # Check if we should switch to local refinement
            # If step size is small enough, we are likely in the basin of attraction
            if step_size < 0.1:
                break
        
        # --- Phase 2: Nelder-Mead Local Refinement ---
        
        # Use best_x from ES as the starting point for NM
        # NM uses a simplex of dim+1 points
        # We have 'local_budget' calls left.
        # Each NM iteration uses roughly dim+1 evaluations (worst case) or fewer if we reuse.
        # Standard NM evaluates one new point per iteration (replacing the worst).
        # So we can do up to local_budget iterations.
        
        # Initialize Simplex
        # Start with best_x and dim neighbors slightly perturbed
        simplex = [best_x.copy()]
        f_vals = [best_f]
        
        # Initial simplex size
        simplex_step = 0.1
        
        for i in range(1, self.dim + 1):
            x_new = best_x.copy()
            # Perturb in direction i
            x_new[i] += simplex_step
            x_new = np.clip(x_new, -5.0, 5.0)
            f_new = func(x_new)
            calls += 1
            simplex.append(x_new)
            f_vals.append(f_new)
            
            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
        
        # NM Iterations
        # We need to track the simplex and its fitness values
        # Sort simplex by fitness (ascending)
        # Index 0 is best, index dim is worst
        
        # To maintain order efficiently, we just keep lists and sort them when needed
        # Or we can just find min and max each time. Given dim=5, sorting is cheap.
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Sort simplex and fitness
            sorted_indices = np.argsort(f_vals)
            simplex = [simplex[i] for i in sorted_indices]
            f_vals = [f_vals[i] for i in sorted_indices]
            
            x_worst = simplex[-1]
            f_worst = f_vals[-1]
            x_best = simplex[0]
            f_best = f_vals[0]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - x_worst)
            x_ref = np.clip(x_ref, -5.0, 5.0)
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < f_best:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - x_worst)
                x_exp = np.clip(x_exp, -5.0, 5.0)
                f_exp = func(x_exp)
                calls += 1
                
                if f_exp < f_ref:
                    x_new = x_exp
                    f_new = f_exp
                else:
                    x_new = x_ref
                    f_new = f_ref
            elif f_ref < f_vals[-2]: # Better than 2nd worst
                x_new = x_ref
                f_new = f_ref
            else:
                # Contraction
                rho = 0.5
                x_con = centroid + rho * (x_worst - centroid)
                x_con = np.clip(x_con, -5.0, 5.0)
                f_con = func(x_con)
                calls += 1
                
                if f_con < f_worst:
                    x_new = x_con
                    f_new = f_con
                else:
                    # Shrink
                    # Move all points (except best) towards best
                    # This requires dim evaluations. 
                    # To save budget, we might just break or do a random kick?
                    # Standard NM shrinks. Let's do it if we have budget.
                    if calls + self.dim > self.budget:
                        break
                    
                    for i in range(1, self.dim + 1):
                        simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        f_vals[i] = func(simplex[i])
                        calls += 1
                        if f_vals[i] < best_f:
                            best_f = f_vals[i]
                            best_x = simplex[i].copy()
                    # After shrink, restart loop
                    continue
            
            # Replace worst with new point
            simplex[-1] = x_new
            f_vals[-1] = f_new
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
