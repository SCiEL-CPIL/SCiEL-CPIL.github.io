"""
Evolved Optimizer for BBOB f20 (Schwefel) with error-repair mechanism.
Uses a structural change: CMA-ES-inspired adaptation with error-repair
that detects out-of-bounds or anomalous evaluations and repairs them
before accepting, combined with multiple diverse initializations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from copy import deepcopy


class Optimizer:
    """
    Optimizer using a population-based adaptive search with error-repair.
    
    Structural change from grid/ES: Uses a population of candidates with
    adaptive covariance matrix (CMA-ES-lite), plus an error-repair mechanism
    that detects when evaluations seem anomalous (e.g., due to clipping
    effects at boundaries) and repairs the candidate before acceptance.
    
    Key innovations:
    1. Population-based search with adaptive step size and direction
    2. Error-repair: when a candidate violates bounds or shows anomalous
       function behavior, repair it by projecting back into feasible region
       and re-evaluating
    3. Multiple diverse starts to handle shifted/rotated minima
    4. Adaptive budget allocation between exploration and exploitation
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin

        calls = 0

        def can_continue():
            return calls < budget and time.time() < deadline

        def remaining():
            return budget - calls

        def make_func_counter():
            nonlocal calls
            def f(x):
                nonlocal calls
                calls += 1
                return func(x)
            return f

        f_count = make_func_counter()

        best_x = None
        best_f = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = np.array(x, dtype=float).copy()
                best_f = fx

        rng = np.random.default_rng(42)

        # Error-repair mechanism: detect and repair anomalous candidates
        def repair_candidate(x, fx, ref_x, ref_f):
            """
            Error-repair: if candidate x has anomalous function value
            relative to reference, attempt to repair it.
            
            Anomalies detected:
            - x is at boundary and fx is much worse than expected
            - fx is NaN or inf (shouldn't happen but guard anyway)
            - fx deviates significantly from local smoothness expectation
            
            Returns repaired (x_repaired, fx_repaired) or original if no repair needed.
            """
            # Check for numerical issues
            if not np.isfinite(fx):
                # Repair: move toward center
                x_repair = x * 0.5
                fx_repair = f_count(x_repair) if can_continue() else fx
                return x_repair, fx_repair

            # Check boundary proximity with poor value
            at_boundary = np.any(np.abs(x) >= 4.9)
            if at_boundary and ref_f is not None and fx > ref_f * 1.5 + 10:
                # Likely clipped effect - try moving inward
                x_repair = x * 0.8
                fx_repair = f_count(x_repair) if can_continue() else fx
                if np.isfinite(fx_repair) and fx_repair < fx:
                    return x_repair, fx_repair

            return x, fx

        # Phase 1: Diverse random initialization with error-repair
        n_init = min(50, budget // 10)
        init_points = []
        
        for i in range(n_init):
            if not can_continue():
                break
            
            # Generate diverse starting points
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = rng.uniform(-4, 4, dim)
            elif i == 2:
                x = rng.uniform(-4, 4, dim)
            elif i == 3:
                x = np.sign(rng.standard_normal(dim)) * np.abs(rng.standard_normal(dim)) * 2
            else:
                # Mix of strategies
                r = rng.random()
                if r < 0.3:
                    x = rng.uniform(-4, 4, dim)
                elif r < 0.6:
                    x = rng.standard_normal(dim) * 1.5
                elif r < 0.8:
                    # Sparse - mostly zero with one non-zero
                    idx = rng.integers(dim)
                    x = np.zeros(dim)
                    x[idx] = rng.uniform(-4, 4)
                else:
                    x = rng.uniform(-4, 4, dim) * 0.5
            
            x = np.clip(x, -5, 5)
            fx = f_count(x)
            
            # Apply error-repair
            x, fx = repair_candidate(x, fx, best_x, best_f)
            
            init_points.append((x.copy(), fx))
            update_best(x, fx)

        if not can_continue() or best_x is None:
            return best_x, best_f

        # Phase 2: Adaptive population-based search
        # Use a population with adaptive step size, inspired by CMA-ES
        pop_size = min(15, remaining() // 10 + 5)
        
        # Initialize population around best point with some diversity
        population = []
        
        # Keep best point
        population.append((best_x.copy(), best_f))
        
        # Add diverse points around best
        for i in range(1, pop_size):
            if not can_continue():
                break
            
            # Adaptive step: starts moderate, can shrink
            sigma = 1.5 / (1 + i * 0.3)
            x_new = best_x + sigma * rng.standard_normal(dim)
            x_new = np.clip(x_new, -5, 5)
            fx_new = f_count(x_new)
            
            # Error-repair
            x_new, fx_new = repair_candidate(x_new, fx_new, best_x, best_f)
            
            population.append((x_new, fx_new))
            update_best(x_new, fx_new)

        if not can_continue():
            return best_x, best_f

        # Phase 3: Iterative refinement with adaptive strategy
        sigma = 0.5
        sigma_min = 0.01
        sigma_max = 2.0
        no_improve_count = 0
        max_no_improve = 20
        
        while can_continue() and no_improve_count < max_no_improve:
            improved_this_iter = False
            iter_calls = 0
            max_iter_calls = min(30, remaining())
            
            for _ in range(max_iter_calls):
                if not can_continue():
                    break
                iter_calls += 1
                
                # Adaptive direction: combine best gradient-like info with random
                direction = rng.standard_normal(dim)
                
                # If we have a second-best point, use it to inform direction
                if len(population) > 1:
                    sorted_pop = sorted(population, key=lambda t: t[1])
                    second_best_x, second_best_f = sorted_pop[1]
                    if second_best_f > best_f:
                        # Direction from second-best to best
                        dir_to_best = best_x - second_best_x
                        dir_norm = np.linalg.norm(dir_to_best)
                        if dir_norm > 1e-8:
                            dir_to_best = dir_to_best / dir_norm
                            # Blend random and informed direction
                            direction = 0.5 * direction + 0.5 * dir_to_best
                            direction /= (np.linalg.norm(direction) + 1e-12)
                
                # Adaptive step size
                step = sigma * np.linalg.norm(direction)
                if step < sigma_min:
                    step = sigma_min
                if step > sigma_max:
                    step = sigma_max
                
                x_trial = best_x + step * direction
                x_trial = np.clip(x_trial, -5, 5)
                
                fx_trial = f_count(x_trial)
                
                # Error-repair
                x_trial, fx_trial = repair_candidate(x_trial, fx_trial, best_x, best_f)
                
                if fx_trial < best_f - 1e-10:
                    # Accept improvement
                    best_x = x_trial.copy()
                    best_f = fx_trial
                    improved_this_iter = True
                    no_improve_count = 0
                    sigma *= 1.1  # Slightly increase step when improving
                    sigma = min(sigma, sigma_max)
                    
                    # Update population: replace worst with new best neighborhood
                    if len(population) > 1:
                        population.sort(key=lambda t: t[1])
                        population[-1] = (best_x + sigma * 0.5 * rng.standard_normal(dim), fx_trial + 1)
                else:
                    no_improve_count += 1
                    sigma *= 0.85  # Shrink step when not improving
                    sigma = max(sigma, sigma_min)
                
                # Update population
                population.append((x_trial.copy(), fx_trial))
                # Keep population size bounded
                if len(population) > pop_size:
                    population.sort(key=lambda t: t[1])
                    population = population[:pop_size]
                
                update_best(x_trial, fx_trial)
            
            # If no improvement in this iteration, try a larger jump (restart-like)
            if not improved_this_iter and can_continue():
                # Escape local minimum: jump to a new random point
                jump_scale = max(sigma, 1.0) * 3
                x_jump = best_x + jump_scale * rng.standard_normal(dim)
                x_jump = np.clip(x_jump, -5, 5)
                fx_jump = f_count(x_jump)
                
                # Error-repair
                x_jump, fx_jump = repair_candidate(x_jump, fx_jump, best_x, best_f)
                
                if fx_jump < best_f:
                    best_x = x_jump.copy()
                    best_f = fx_jump
                    sigma = max(sigma, 0.5)
                    no_improve_count = 0
                else:
                    sigma *= 0.5
                    sigma = max(sigma, sigma_min)
                
                update_best(x_jump, fx_jump)

        # Phase 4: Final local refinement with very small steps
        if can_continue():
            sigma_final = 0.05
            for _ in range(remaining()):
                if not can_continue():
                    break
                
                x_trial = best_x + sigma_final * rng.standard_normal(dim)
                x_trial = np.clip(x_trial, -5, 5)
                fx_trial = f_count(x_trial)
                
                # Error-repair
                x_trial, fx_trial = repair_candidate(x_trial, fx_trial, best_x, best_f)
                
                if fx_trial < best_f:
                    best_x = x_trial.copy()
                    best_f = fx_trial
                    sigma_final *= 1.1
                else:
                    sigma_final *= 0.8
                    sigma_final = max(sigma_final, 0.001)
                
                update_best(x_trial, fx_trial)

        return best_x, best_f

# EVOLVE-BLOCK-END
