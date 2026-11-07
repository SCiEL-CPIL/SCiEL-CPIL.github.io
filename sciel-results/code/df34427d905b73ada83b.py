import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        n = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0
        eval_count = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f, eval_count
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Global Scan (Halton Sequence)
        # The Step Ellipsoidal function has a flat plateau. 
        # A systematic scan ensures we cross the discontinuity early.
        scan_budget = int(budget * 0.2)  # 100 calls
        
        halton = qmc.Halton(d=n, scramble=True, seed=42)
        points_generated = 0
        
        while calls_made < scan_budget and (time.time() - start_time) < deadline:
            # Generate points in batches to strictly respect budget/time
            batch_size = min(10, scan_budget - calls_made)
            if batch_size <= 0:
                break
            
            # Generate points in [0,1]^d
            points_unit = halton.random(batch_size)
            
            # Map to [-5, 5]^d
            points_mapped = points_unit * 10.0 - 5.0
            
            for p in points_mapped:
                if calls_made >= scan_budget:
                    break
                if (time.time() - start_time) > deadline:
                    break
                evaluate(p)
            
            if (time.time() - start_time) > deadline:
                break

        # Phase 2: (1+10)-ES with Periodic Large Jumps
        # This strategy helps escape flat plateaus characteristic of the step-ellipsoidal function.
        # We use a population of 10 candidates per generation, keeping the best.
        
        # Remaining budget for ES
        es_budget = budget - calls_made
        
        if es_budget > 0:
            # Initialize current best as the starting point
            x_current = best_x.copy()
            f_current = best_f
            
            # Initial step size: roughly 1/5 of the domain width (10)
            step_size = 1.0
            
            # Adaptation parameters
            sigma = 1.0
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n*n))
            c_sigma = 1.0 / (5 * n + 5)
            c_c = 4 / (n + 4)
            c_1 = 2 / ((n + 1.3)**2 + n)
            c_mu = min(1 - c_1, 2 * ((n + 1) / (n + 3) - 1))
            mu = n // 2
            
            # Cumulation
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            d_amps = 1 + 2 * min(1, (mu - 1) / (n + 1)) * c_sigma / c_1
            
            # History of sigma for adaptation
            # We will use a simpler adaptation for (1+10)-ES to be robust
            # Actually, let's use a standard CMA-ES-like update but with lambda=10
            
            # Simpler (1+10)-ES implementation:
            # Keep a population of 10, evaluate them, select the best.
            # Adapt sigma based on improvement.
            
            pop_size = 10
            last_improvement = 0
            no_improve_count = 0
            
            # Use the best point from scan as initial center
            center = best_x.copy()
            f_center = best_f
            
            # Initial sigma
            sigma = 1.0
            
            # We will run generations until budget or time is up
            while calls_made < budget and (time.time() - start_time) < deadline:
                # Generate population
                # One individual is the current best (center)
                # Others are perturbations
                population = []
                fitness = []
                
                # Add center
                population.append(center)
                fitness.append(f_center)
                
                # Generate perturbations
                for _ in range(pop_size - 1):
                    # Sample from normal distribution
                    x_trial = center + sigma * rng.normal(size=n)
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    
                    # Evaluate
                    if calls_made >= budget:
                        break
                    if (time.time() - start_time) > deadline:
                        break
                    
                    f_trial = func(x_trial)
                    calls_made += 1
                    eval_count += 1
                    
                    population.append(x_trial)
                    fitness.append(f_trial)
                    
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()
                
                if calls_made >= budget or (time.time() - start_time) > deadline:
                    break
                
                # Selection: Find the best in the population
                idx_best = np.argmin(fitness)
                new_center = population[idx_best]
                new_f = fitness[idx_best]
                
                if new_f < f_center:
                    center = new_center
                    f_center = new_f
                    no_improve_count = 0
                    # Increase sigma slightly if we improved? 
                    # Or keep it? CMA-ES usually adapts based on success rate.
                    # For simplicity, let's use a simple rule:
                    # If improved, keep sigma. If not, decrease sigma.
                else:
                    no_improve_count += 1
                    if no_improve_count > 5:
                        # Reduce sigma to refine
                        sigma *= 0.8
                        no_improve_count = 0
                        # If sigma becomes too small, we might be stuck.
                        # But for step ellipsoidal, we want to find the edge.
                        # If we are on the plateau, we need to jump.
                        if sigma < 0.01:
                            # Large jump escape
                            jump_size = 2.0
                            x_jump = center + rng.normal(size=n) * jump_size
                            x_jump = np.clip(x_jump, -5.0, 5.0)
                            if calls_made < budget:
                                f_jump = func(x_jump)
                                calls_made += 1
                                if f_jump < best_f:
                                    best_f = f_jump
                                    best_x = x_jump.copy()
                                    center = x_jump
                                    f_center = f_jump
                            sigma = 1.0 # Reset sigma after jump
                            no_improve_count = 0

        # Phase 3: Local Refinement (Coordinate Descent)
        # Use the remaining budget for fine-tuning.
        if calls_made < budget:
            step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
            
            current_x = best_x.copy()
            current_f = best_f
            
            for step in step_sizes:
                if calls_made >= budget:
                    break
                if (time.time() - start_time) > deadline:
                    break
                    
                improved = True
                max_iter = 5
                iter_count = 0
                
                while improved and calls_made < budget and iter_count < max_iter:
                    improved = False
                    iter_count += 1
                    for d_idx in range(n):
                        if calls_made >= budget:
                            break
                        if (time.time() - start_time) > deadline:
                            break
                            
                        # Try positive direction
                        x_pos = current_x.copy()
                        x_pos[d_idx] += step
                        x_pos = np.clip(x_pos, -5, 5)
                        f_pos = func(x_pos)
                        calls_made += 1
                        
                        if f_pos < current_f:
                            current_x = x_pos
                            current_f = f_pos
                            if current_f < best_f:
                                best_f = current_f
                                best_x = current_x.copy()
                            improved = True
                            continue
                            
                        # Try negative direction
                        x_neg = current_x.copy()
                        x_neg[d_idx] -= step
                        x_neg = np.clip(x_neg, -5, 5)
                        f_neg = func(x_neg)
                        calls_made += 1
                        
                        if f_neg < current_f:
                            current_x = x_neg
                            current_f = f_neg
                            if current_f < best_f:
                                best_f = current_f
                                best_x = current_x.copy()
                            improved = True
                            continue

        return best_x, best_f
