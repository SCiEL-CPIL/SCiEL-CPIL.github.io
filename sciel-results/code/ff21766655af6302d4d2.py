import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        
    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ranges = hi - lo
        
        # Phase 1: Global Exploration with Sobol Sequence
        # Allocate 50% of budget to global exploration
        n_init = self.budget // 2
        
        # Generate Sobol points
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol requires power of 2 for best performance, but we can use any number
        # We round up to next power of 2 for sampler generation, then slice
        n_sobol = 2 ** int(np.ceil(np.log2(n_init)))
        points = sampler.random(n_sobol)
        points = points[:n_init]
        
        # Scale to bounds
        X_init = lo + points * ranges
        
        # Evaluate
        F_init = np.array([problem(x) for x in X_init])
        
        # Phase 2: Local Refinement
        # Identify Pareto front from initial population
        pareto_indices = self._pareto_front_indices(F_init)
        
        if len(pareto_indices) == 0:
            return None

        # Select top candidates for local search
        # Use 8 candidates to allow more budget per candidate (approx 12-15 evals each)
        n_local_candidates = min(8, len(pareto_indices))
        
        # Sort Pareto front by first objective to select diverse points
        pareto_F = F_init[pareto_indices]
        sorted_pareto_idx = np.argsort(pareto_F[:, 0])
        pareto_indices_sorted = pareto_indices[sorted_pareto_idx]
        
        # Select n_local_candidates evenly
        if n_local_candidates > 1:
            step = len(pareto_indices_sorted) / n_local_candidates
            selected_indices = [pareto_indices_sorted[int(i * step)] for i in range(n_local_candidates)]
        else:
            selected_indices = [pareto_indices_sorted[0]]
            
        # Local Search Budget
        remaining_budget = self.budget - n_init
        n_evals_per_candidate = remaining_budget // len(selected_indices)
        
        # Adaptive Local Search
        for i, idx in enumerate(selected_indices):
            x_curr = X_init[idx].copy()
            f_curr = F_init[idx].copy()
            
            # Adaptive step size: start with 10% of range
            step_size = 0.1 * ranges
            no_improvement_count = 0
            
            evals_used_for_this = 0
            while evals_used_for_this < n_evals_per_candidate:
                # Generate one neighbor
                delta = np.random.randn(self.dim) * step_size
                x_new = x_curr + delta
                x_new = np.clip(x_new, lo, hi)
                
                f_new = np.array(problem(x_new))
                evals_used_for_this += 1
                
                # Check if new point strictly dominates current point
                if np.all(f_new <= f_curr) and np.any(f_new < f_curr):
                    x_curr = x_new
                    f_curr = f_new
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= 3:
                        step_size *= 0.7
                        no_improvement_count = 0
                        
        return None

    def _pareto_front_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]
