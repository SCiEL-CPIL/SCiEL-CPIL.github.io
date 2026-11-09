# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        
        # Population size: moderate for diversity and computational cost
        # Current best used 25. Let's stick close to that.
        pop_size = min(25, self.budget // 8)
        
        # --- Initialization ---
        # Combine random sampling with "peak-biased" initialization from Inspiration 2
        # WFG7 is multimodal. Common peaks in normalized [0,1] space are around 0.35, 0.5, 0.65.
        # We will split the initial population: half random, half peak-biased.
        
        X_list = []
        F_list = []
        used = 0
        
        peak_vals = [0.35, 0.5, 0.65, 0.2, 0.8, 0.4, 0.6]
        
        for i in range(pop_size):
            if used >= self.budget:
                break
            if i < pop_size // 2:
                # Random initialization
                x = lo + (hi - lo) * rng.random(self.dim)
            else:
                # Peak-biased initialization
                # Pick a random peak value for each dimension + small noise
                x = np.array([
                    lo[d] + (hi[d] - lo[d]) * (peak_vals[rng.integers(len(peak_vals))] + 0.05 * rng.standard_normal())
                    for d in range(self.dim)
                ])
                x = np.clip(x, lo, hi)
            
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            X_list.append(x)
            F_list.append(f)
            used += 1
            
            if time.time() > deadline:
                break

        if not X_list:
            return np.array([])

        X = np.array(X_list)
        F = np.array(F_list)
        
        # --- Evolutionary Loop ---
        # Use adaptive step size. Start moderate, shrink slowly.
        # Current best used 0.12 start, 0.995 shrink.
        # Let's try a slightly larger initial step for exploration, then refine.
        step_scale = 0.15
        shrink_rate = 0.995
        
        # We will maintain a fixed size population.
        # If we generate a child, we replace the worst point.
        
        while used < self.budget and time.time() < deadline:
            # Find non-dominated points
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            
            # Vectorized domination check is tricky to do perfectly in one go for all i,
            # so we stick to the loop which is fast for N=25.
            for i in range(n):
                if dominated[i]:
                    continue
                # Check if any other point dominates i
                # F[j] <= F[i] for all obj AND F[j] < F[i] for at least one obj
                le = np.all(F <= F[i], axis=1)
                lt = np.any(F < F[i], axis=1)
                le[i] = False  # exclude self
                if np.any(le & lt):
                    dominated[i] = True
            
            # Select parent from non-dominated set
            nd_idx = np.where(~dominated)[0]
            if len(nd_idx) > 0:
                # Weighted selection? Or just random?
                # Random from non-dominated is simple and effective.
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
            else:
                # Fallback if all dominated (shouldn't happen if we keep non-dom)
                parent_idx = rng.integers(n)
            
            parent = X[parent_idx].copy()
            
            # Mutate with Gaussian noise
            # Adaptive step: shrink over time
            mutation = step_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = np.array(problem(child), dtype=float)
            used += 1
            
            # Replace the worst point
            # "Worst" is defined by highest sum of objectives (simple scalarization)
            # This is a heuristic but works well for maintaining spread.
            # To be more robust, we could check if the new child is non-dominated.
            # If it is, it's likely good. If it's dominated, it might still be better than the worst.
            
            sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(sums))
            
            # Update population
            X[worst_idx] = child
            F[worst_idx] = fc
            
            # Shrink step size
            step_scale *= shrink_rate
            
            # Optional: If we have time, do a local polish on the best point?
            # The directive suggests a "final high-precision polish pass".
            # Let's check if we are near the end of the budget.
            if used > self.budget * 0.9:
                # Polish phase: reduce step size significantly
                step_scale = min(step_scale, 0.01)
                # Maybe even switch to a finer search around the current best non-dominated points
                # But simply shrinking the step is a good approximation.
                if used > self.budget * 0.95:
                    step_scale = min(step_scale, 0.001)

        return X
# EVOLVE-BLOCK-END
