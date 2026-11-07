"""Optimized search for BBOB f15 (rotated Rastrigin).
Combines (μ+λ)-ES population exploration with Rastrigin-aware periodic
coordinate refinement and multi-restart to exploit the known structure
of the objective while maintaining basin coverage.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based (μ+λ)-ES with Rastrigin-periodic refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x_clipped))
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return call_count < budget

        # --- Local refinement: coordinate descent with Rastrigin-periodic snapping ---
        def local_refine(x, fx, max_rounds=5):
            """Coordinate descent with Rastrigin-periodic candidates.
            
            Rastrigin minima occur at multiples of 2*pi in the unrotated frame.
            Even with rotation, snapping coordinates to 2*pi*k provides good
            local candidates since the function is periodic in each axis direction.
            """
            nonlocal call_count, best_x, best_f
            if not budget_ok() or not time_ok():
                return x, fx

            two_pi = 2.0 * np.pi

            for _ in range(max_rounds):
                if not budget_ok() or not time_ok():
                    break
                improved = False
                for axis in range(dim):
                    if not budget_ok() or not time_ok():
                        break
                    x0 = x[axis]
                    
                    # Generate candidates: current, nearby, and Rastrigin-aligned
                    candidates = set()
                    
                    # Small perturbations around current value
                    for d in [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]:
                        candidates.add(float(np.clip(x0 + d, -5.0, 5.0)))
                    
                    # Rastrigin: minima at multiples of 2*pi
                    k = np.round(x0 / two_pi)
                    for kk in [k - 1, k, k + 1]:
                        candidates.add(float(np.clip(kk * two_pi, -5.0, 5.0)))
                    
                    # Also try points halfway between minima (to avoid maxima)
                    for kk in [k - 1, k, k + 1]:
                        candidates.add(float(np.clip(kk * two_pi + two_pi / 2, -5.0, 5.0)))
                    
                    # Remove current value to avoid redundant evaluation
                    candidates.discard(float(np.clip(x0, -5.0, 5.0)))
                    
                    if not candidates:
                        continue
                    
                    best_cand = x0
                    best_cand_f = fx
                    
                    for c in candidates:
                        if not budget_ok() or not time_ok():
                            break
                        c = float(c)
                        if abs(c - x0) < 1e-12:
                            continue
                        y = x.copy()
                        y[axis] = c
                        fy = eval_func(y)
                        if fy < best_cand_f:
                            best_cand_f = fy
                            best_cand = c
                    
                    if best_cand != x0:
                        x[axis] = best_cand
                        fx = best_cand_f
                        improved = True
                
                if not improved:
                    break
            
            return x, fx

        # --- Phase 1: Initial diverse sampling ---
        init_count = 8
        for _ in range(init_count):
            if not budget_ok() or not time_ok():
                break
            x = np.clip(rng.uniform(-5, 5, dim), -5, 5)
            eval_func(x)
        
        if not budget_ok() or not time_ok() or best_x is None:
            return best_x, best_f

        # --- Phase 2: (μ+λ)-ES with periodic local refinement ---
        mu = 4
        lam = 10
        sigma = 2.0
        
        # Initialize population from best points found so far
        # We don't have explicit population tracking, so re-sample around best
        pop = [best_x.copy()]
        pop_f = [best_f]
        
        # Add some diverse points
        for _ in range(mu - 1):
            if not budget_ok() or not time_ok():
                break
            x = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
            fx = eval_func(x)
            pop.append(x)
            pop_f.append(fx)
        
        # Sort population
        idx = np.argsort(pop_f)
        pop = [pop[i] for i in idx]
        pop_f = [pop_f[i] for i in idx]

        generation = 0
        no_improve_count = 0
        best_f_prev = best_f
        stagnation_threshold = 5  # generations without improvement before restart

        while budget_ok() and time_ok():
            generation += 1

            # Adaptive sigma
            if best_f < best_f_prev - 1e-6:
                sigma = min(sigma * 1.2, 6.0)
                best_f_prev = best_f
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= stagnation_threshold:
                    sigma = max(sigma * 0.7, 0.15)
                    no_improve_count = 0

            # Generate offspring
            offspring = []
            offspring_f = []
            for _ in range(lam):
                if not budget_ok() or not time_ok():
                    break
                # Select parent (uniform from population)
                parent_idx = rng.integers(0, mu)
                parent = pop[parent_idx]
                child = np.clip(parent + sigma * rng.standard_normal(dim), -5, 5)
                child_f = eval_func(child)
                offspring.append(child)
                offspring_f.append(child_f)

            if not offspring:
                break

            # Combine and select
            all_x = pop + offspring
            all_f = pop_f + offspring_f
            idx = np.argsort(all_f)[:mu]
            pop = [all_x[i] for i in idx]
            pop_f = [all_f[i] for i in idx]

            # Periodic local refinement on the best point every few generations
            if generation % 8 == 0 and budget_ok() and time_ok():
                x_ref, f_ref = local_refine(pop[0].copy(), pop_f[0], max_rounds=3)
                if f_ref < pop_f[0]:
                    pop[0] = x_ref
                    pop_f[0] = f_ref

            # If stuck with very small sigma, inject diversity
            if sigma < 0.2 and generation % 20 == 0:
                if budget_ok() and time_ok():
                    new_x = np.clip(rng.uniform(-5, 5, dim), -5, 5)
                    new_f = eval_func(new_x)
                    worst_idx = np.argmax(pop_f)
                    if new_f < pop_f[worst_idx]:
                        pop[worst_idx] = new_x
                        pop_f[worst_idx] = new_f
                    sigma = 1.5
                    no_improve_count = 0

        # --- Phase 3: Final intensive local refinement ---
        if best_x is not None and budget_ok() and time_ok():
            x_final = best_x.copy()
            f_final = best_f
            for _ in range(8):
                if not budget_ok() or not time_ok():
                    break
                x_ref, f_ref = local_refine(x_final.copy(), f_final, max_rounds=5)
                if f_ref < f_final:
                    f_final = f_ref
                    x_final = x_ref.copy()
                else:
                    break

            if f_final < best_f:
                best_f = f_final
                best_x = x_final

        return best_x, best_f
# EVOLVE-BLOCK-END
