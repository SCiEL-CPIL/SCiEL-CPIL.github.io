"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Sobol sweep + Gaussian local refinement for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42) # Fixed seed for reproducibility within the call, though problem uses its own seed for evaluation
        
        all_x = []
        all_f = []
        eval_count = 0
        
        def is_dominated(f_a, f_b):
            # Returns True if f_b dominates f_a
            # f_b dominates f_a if f_b <= f_a in all objectives and < in at least one
            return np.all(f_b <= f_a) and np.any(f_b < f_a)

        def get_non_dominated_indices(f_list):
            if not f_list:
                return []
            f_arr = np.array(f_list)
            n = len(f_arr)
            nd_indices = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if is_dominated(f_arr[i], f_arr[j]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return nd_indices

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            
            x_clipped = np.clip(x, lo, hi)
            try:
                f = problem(x_clipped)
                f = np.array(f)
                if not np.all(np.isfinite(f)):
                    return None
            except Exception:
                return None
            
            eval_count += 1
            all_x.append(x_clipped)
            all_f.append(f)
            return f

        # Phase 1: Sobol Quasi-Random Sweep
        # Use a power of 2 for Sobol quality. 128 points is a good start.
        n_sobol = 128
        
        # Generate Sobol sequence
        # SciPy's qmc.Sobol is available but might be overkill or have import overhead?
        # Standard library doesn't have Sobol. Let's use a simple deterministic Halton-like or
        # implement a simple Sobol-like grid.
        # Actually, since we can't rely on scipy import being fast or present in all minimal envs (though it is available),
        # let's use a deterministic stratified sampling or a simple Halton sequence which is easy to implement.
        
        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    base = j + 2
                    f = 1.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        i_val = int(i_val / base)
                    points[i, j] = f
            return points

        sobol_points = halton_sequence(n_sobol, self.dim)
        sobol_points = lo + sobol_points * (hi - lo)
        
        for i in range(n_sobol):
            if time.time() > deadline:
                break
            evaluate(sobol_points[i])

        # Phase 2: Local Gaussian Refinement
        # We have used ~128 evals. Remaining: 200 - 128 = 72 evals.
        # We will identify non-dominated points and perturb them.
        
        while eval_count < self.budget and time.time() < deadline:
            if not all_f:
                break
                
            nd_indices = get_non_dominated_indices(all_f)
            if not nd_indices:
                # If no non-dominated points (shouldn't happen with valid points), break
                break
            
            # Pick a non-dominated point to refine
            # We can cycle through them or pick the one with best "spread" or just round-robin
            # To ensure we explore different parts of the front, let's pick the one with max crowding distance approximation
            # or just cycle. Cycling is simpler and robust.
            
            # Let's pick the index from the current list of all points that is non-dominated.
            # To avoid re-evaluating the exact same point, we perturb it.
            
            # Choose a base point. Let's pick the first one for now, or randomize among ND points.
            base_idx = nd_indices[rng.integers(len(nd_indices))]
            x_base = all_x[base_idx]
            
            # Determine perturbation size. 
            # Start with a moderate sigma, e.g., 0.1 of the range.
            # We can adapt sigma based on recent improvements, but let's keep it simple: fixed sigma.
            sigma = 0.05
            
            # Generate a perturbation
            perturbation = rng.normal(0, sigma, size=self.dim)
            x_new = x_base + perturbation
            
            # Evaluate
            f_new = evaluate(x_new)
            
            if f_new is None:
                continue

            # If we have very few evals left, we might want to try smaller steps?
            # No, let's just keep going until budget is exhausted.

        # Return the first point (ignored by evaluator, but required by interface)
        if all_x:
            return all_x[0]
        return lo

# EVOLVE-BLOCK-END
