"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Structured Initialization + Adaptive Multi-Restart (1+1)-ES Optimizer.
    
    This optimizer combines deterministic structured sampling (lattice + Halton) 
    for robust early coverage with an adaptive (1+1)-Evolution Strategy for 
    local refinement. It uses a multi-restart strategy to escape local minima 
    on multimodal functions like Rastrigin and Gallagher.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Helper to update best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Helper to check time
        def time_exceeded():
            return time.time() > deadline

        # ---------------------------------------------------------
        # Phase 1: Structured Initialization
        # ---------------------------------------------------------
        # We generate a diverse set of candidate points to evaluate first.
        # This ensures we don't miss obvious minima (like integers for Rastrigin)
        # and provides a good starting point for the ES.

        candidates = []

        # 1. Lattice points (Integer grid)
        # Rastrigin has minima at integers. Sampling a subset of the integer grid
        # in [-4, 4] is cheap and high-yield for f3 and f15.
        lattice_vals = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        n_lattice_samples = 30
        for _ in range(n_lattice_samples):
            pt = np.array([rng.choice(lattice_vals) for _ in range(dim)])
            candidates.append(pt)

        # 2. Halton Sequence for quasi-random coverage
        def halton(idx, base):
            q = 1.0
            r = 0.0
            while idx > 0:
                q /= base
                r += q * (idx % base)
                idx //= base
            return r

        primes = [2, 3, 5, 7, 11]
        n_halton = 40
        for i in range(1, n_halton + 1):
            pt = np.zeros(dim)
            for j in range(dim):
                pt[j] = halton(i, primes[j])
            # Map [0, 1) to [-5, 5]
            pt = pt * 10.0 - 5.0
            candidates.append(np.clip(pt, -5, 5))

        # 3. Random Uniform Points
        n_random = 20
        for _ in range(n_random):
            pt = rng.uniform(-5, 5, dim)
            candidates.append(pt)

        # Prioritize high-value points for early AOCC score
        high_priority = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
        ]
        
        # Evaluate high priority first
        evaluated_points = []
        for hp in high_priority:
            if calls >= budget or time_exceeded():
                break
            pt = np.clip(hp, -5, 5)
            f = func(pt)
            calls += 1
            update_best(pt, f)
            evaluated_points.append(pt)

        # Filter out high priority points from candidates to avoid re-evaluation
        # Simple check: if candidate is close to any evaluated point, skip
        indices = rng.permutation(len(candidates))
        
        for idx in indices:
            if calls >= budget or time_exceeded():
                break
            pt = np.clip(candidates[idx], -5, 5)
            
            # Skip if we just evaluated a very similar point (optional optimization)
            # For now, just evaluate.
            f = func(pt)
            calls += 1
            update_best(pt, f)

        # ---------------------------------------------------------
        # Phase 2: Adaptive Multi-Restart (1+1)-ES
        # ---------------------------------------------------------
        # We now refine the best point found. We use a (1+1)-ES with adaptive sigma.
        # To avoid stagnation, we periodically restart from new points if progress
        # stalls or after a fixed number of steps.
        
        # Parameters for ES
        base_sigma = 1.0
        min_sigma = 0.01
        max_sigma = 4.0
        
        # We divide the remaining budget into "restarts".
        # Each restart runs for a certain number of steps.
        # If we find a new global best, we might continue. If we stagnate, we restart.
        
        # Let's use a dynamic restart strategy:
        # - Run ES for `steps_per_restart` steps.
        # - If no improvement in last `stagnation_limit` steps, restart.
        # - Restart point is either:
        #   a) A random point in the box (global exploration)
        #   b) A perturbation of the current best (local exploration)
        
        steps_per_restart = 20
        stagnation_limit = 15
        
        x = best_x.copy()
        fx = best_f
        sigma = base_sigma
        
        restart_count = 0
        stagnation_counter = 0
        last_improvement_f = fx
        
        while calls < budget and not time_exceeded():
            # Determine if we should restart
            if stagnation_counter >= stagnation_limit:
                # Restart
                restart_count += 1
                stagnation_counter = 0
                last_improvement_f = fx
                
                # Choose new starting point
                # Probability of global vs local restart decreases as we use more budget
                progress = calls / budget
                p_global = max(0.1, 0.5 - 0.4 * progress) # Early: 50% global, Late: 10% global
                
                if rng.random() < p_global:
                    # Global random restart
                    x = rng.uniform(-5, 5, dim)
                    sigma = base_sigma # Reset sigma to explore broadly
                else:
                    # Local restart: perturb current best
                    x = np.clip(best_x + rng.normal(0, sigma, dim), -5, 5)
                    sigma = max(sigma, 0.5)
                
                fx = func(x)
                calls += 1
                update_best(x, fx)
                continue

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                update_best(x, fx)
                sigma *= 1.2
                stagnation_counter = 0
            else:
                sigma *= 0.85
                stagnation_counter += 1
            
            # Clamp sigma
            sigma = np.clip(sigma, min_sigma, max_sigma)

        return best_x, best_f
# EVOLVE-BLOCK-END
