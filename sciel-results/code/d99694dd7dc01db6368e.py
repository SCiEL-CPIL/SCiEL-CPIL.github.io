"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer:
    1. Population-based Random Search (RS) for global exploration.
    2. Coordinate Descent (CD) with Golden Section Search for local exploitation.
    
    This approach is structurally different from (1+1)-ES:
    - RS maintains a population, allowing it to escape local minima or find the 
      correct basin of attraction more reliably in the presence of unknown shifts/rotations.
    - CD decouples dimensions, which is ideal for the Bent Cigar function (f12) where 
      one dimension is very narrow (high curvature) and others are broad. 
      Golden section search provides precise 1D convergence without requiring gradient 
      information or complex covariance updates.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # --- Phase 1: Population-based Random Search ---
        # Allocate ~30% of the budget to explore the landscape.
        # This helps identify the region containing the minimum, especially 
        # given the hidden shift/rotation.
        pop_size = 10
        n_rs_calls = int(budget * 0.3)
        
        # Initialize population
        x_best = rng.uniform(-5, 5, dim)
        f_best = func(x_best)
        calls_made = 1
        
        # Maintain a small population of candidates
        population = [x_best]
        
        for i in range(n_rs_calls - 1):
            if time.time() > deadline:
                break
            # Generate new candidate: mix of random restarts and local perturbations
            # 20% chance of global restart, 80% local perturbation around best
            if rng.random() < 0.2:
                x_new = rng.uniform(-5, 5, dim)
            else:
                # Perturb the current best
                x_new = np.clip(x_best + 1.0 * rng.standard_normal(dim), -5, 5)
            
            f_new = func(x_new)
            calls_made += 1
            
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()
                population.append(x_new)
            elif len(population) < pop_size:
                population.append(x_new)
            else:
                # Keep the best 'pop_size' candidates to maintain diversity
                # Simple heuristic: just keep the best one and some random ones?
                # For simplicity and speed, we just track the single best for CD start.
                pass
        
        # --- Phase 2: Coordinate Descent with Golden Section Search ---
        # Use the remaining budget to refine the best point found in Phase 1.
        # CD iterates over dimensions, performing a 1D optimization on each.
        
        # Golden Section Search parameters
        phi = (1 + np.sqrt(5)) / 2
        inv_phi = 1 / phi
        inv_phi2 = inv_phi * inv_phi
        
        def golden_section_1d(func, x, dim_idx, f_current, budget_limit, deadline, rng):
            """
            Perform 1D golden section search on dimension `dim_idx` of `x`.
            Returns updated x, f, and number of calls used.
            """
            calls = 0
            # Initial interval: expand from current point
            # Since we don't know the scale, start with a moderate step
            # and expand if we keep going in the same direction, or shrink if stuck.
            # For Bent Cigar, the narrow dimension needs precise search.
            
            # Determine initial search bounds
            # We try to find a bracket [a, b] where f(a) > f(x) and f(b) > f(x)
            # or just search in a local window.
            
            # Strategy: Adaptive 1D search
            # Start with a step size based on the remaining budget and dimension count.
            # A simple approach: try steps of size 1.0, 0.1, 0.01...
            
            # However, Golden Section requires a bracket. 
            # Let's use a "bracketing" phase first.
            
            # Try to expand from x[dim_idx] to find a valley.
            step = 1.0
            a = x[dim_idx]
            b = x[dim_idx]
            
            # We have a limited budget for this 1D search.
            # Let's allocate a small fixed number of calls for bracketing, 
            # then use the rest for refinement.
            max_bracket_calls = 5
            max_refine_calls = 20
            
            # Bracketing
            f_x = f_current
            # Try positive direction
            x_test = x.copy()
            x_test[dim_idx] = np.clip(x[dim_idx] + step, -5, 5)
            f_test = func(x_test)
            calls += 1
            
            if f_test < f_x:
                # Move in positive direction
                a = x[dim_idx]
                b = x[dim_idx] + step
                f_a = f_x
                f_b = f_test
                best_val = f_test
                best_pos = x[dim_idx] + step
                # Expand further if still improving
                while calls < max_bracket_calls:
                    step *= 1.5
                    x_test[dim_idx] = np.clip(x[dim_idx] + step, -5, 5)
                    f_test = func(x_test)
                    calls += 1
                    if f_test < best_val:
                        a = b
                        b = x[dim_idx] + step
                        f_a = f_b
                        f_b = f_test
                        best_val = f_test
                        best_pos = x[dim_idx] + step
                    else:
                        break
            else:
                # Try negative direction
                x_test[dim_idx] = np.clip(x[dim_idx] - step, -5, 5)
                f_test = func(x_test)
                calls += 1
                
                if f_test < f_x:
                    # Move in negative direction
                    b = x[dim_idx]
                    a = x[dim_idx] - step
                    f_b = f_x
                    f_a = f_test
                    best_val = f_test
                    best_pos = x[dim_idx] - step
                    while calls < max_bracket_calls:
                        step *= 1.5
                        x_test[dim_idx] = np.clip(x[dim_idx] - step, -5, 5)
                        f_test = func(x_test)
                        calls += 1
                        if f_test < best_val:
                            b = a
                            a = x[dim_idx] - step
                            f_b = f_a
                            f_a = f_test
                            best_val = f_test
                            best_pos = x[dim_idx] - step
                        else:
                            break
                else:
                    # No improvement in either direction with initial step.
                    # Shrink step and try again? Or just return current.
                    # For efficiency, if no improvement with step=1.0, 
                    # the minimum might be very close or we are in a flat region.
                    # Try a smaller step.
                    step = 0.1
                    x_test[dim_idx] = np.clip(x[dim_idx] + step, -5, 5)
                    f_test = func(x_test)
                    calls += 1
                    if f_test < f_x:
                        a = x[dim_idx]
                        b = x[dim_idx] + step
                        f_a = f_x
                        f_b = f_test
                        best_val = f_test
                        best_pos = x[dim_idx] + step
                    else:
                        x_test[dim_idx] = np.clip(x[dim_idx] - step, -5, 5)
                        f_test = func(x_test)
                        calls += 1
                        if f_test < f_x:
                            b = x[dim_idx]
                            a = x[dim_idx] - step
                            f_b = f_x
                            f_a = f_test
                            best_val = f_test
                            best_pos = x[dim_idx] - step
                        else:
                            # No improvement found. Return current.
                            return x, f_x, calls
            
            # Now we have a bracket [a, b] with a better point inside.
            # Perform Golden Section Search to refine.
            # Ensure a < b
            if a > b:
                a, b = b, a
                f_a, f_b = f_b, f_a
            
            # Golden Section Iterations
            c = b - inv_phi * (b - a)
            d = a + inv_phi * (b - a)
            
            # Evaluate c and d
            x_c = x.copy()
            x_c[dim_idx] = c
            f_c = func(x_c)
            calls += 1
            
            x_d = x.copy()
            x_d[dim_idx] = d
            f_d = func(x_d)
            calls += 1
            
            # Initialize best in bracket
            if f_c < f_d:
                # Min is in [a, d]
                b = d
                f_b = f_d
                best_val = f_c
                best_pos = c
            else:
                # Min is in [c, b]
                a = c
                f_a = f_c
                best_val = f_d
                best_pos = d
            
            # Iterate until budget or precision
            while calls < max_bracket_calls + max_refine_calls:
                if time.time() > deadline:
                    break
                
                # Calculate new points
                if f_c < f_d:
                    # Shrink from right
                    b = d
                    d = c
                    f_d = f_c
                    c = b - inv_phi * (b - a)
                    x_c[dim_idx] = c
                    f_c = func(x_c)
                    calls += 1
                    if f_c < best_val:
                        best_val = f_c
                        best_pos = c
                else:
                    # Shrink from left
                    a = c
                    c = d
                    f_c = f_d
                    d = a + inv_phi * (b - a)
                    x_d[dim_idx] = d
                    f_d = func(x_d)
                    calls += 1
                    if f_d < best_val:
                        best_val = f_d
                        best_pos = d
                
                # Convergence check
                if abs(b - a) < 1e-4:
                    break
            
            # Update x with the best position found in this dimension
            x[dim_idx] = best_pos
            return x, best_val, calls

        # Main CD Loop
        # Iterate through dimensions multiple times until budget is exhausted
        remaining_budget = budget - calls_made
        
        # Estimate calls per CD cycle: 5 dims * (5 bracket + 20 refine) = 125 calls per cycle
        # We can afford roughly remaining_budget / 125 cycles.
        
        while calls_made < budget and time.time() < deadline:
            improved = False
            for dim_idx in range(dim):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Perform 1D optimization on this dimension
                x_new, f_new, calls_used = golden_section_1d(
                    func, x_best, dim_idx, f_best, budget, deadline, rng
                )
                
                calls_made += calls_used
                
                if f_new < f_best:
                    x_best = x_new
                    f_best = f_new
                    improved = True
            
            # If no dimension improved, we might be stuck or converged.
            # To avoid infinite loops in flat regions, we can break or do a small random kick.
            if not improved:
                # Try a small random perturbation to escape potential local minima
                if calls_made < budget - 10:
                    x_kick = np.clip(x_best + 0.1 * rng.standard_normal(dim), -5, 5)
                    f_kick = func(x_kick)
                    calls_made += 1
                    if f_kick < f_best:
                        x_best = x_kick
                        f_best = f_kick
                    else:
                        # If kick doesn't help and we've done several cycles, break
                        # For simplicity, just break to save time for final verification
                        break

        return x_best, f_best
# EVOLVE-BLOCK-END
