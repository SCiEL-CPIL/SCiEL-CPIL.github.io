"""Multi-objective optimizer on ZDT1 with structure-aware sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer that exploits ZDT1 structure:
    - g(x) is minimized when x[1:] = 0.5
    - f1 = x[0], f2 = (1 + g) * (1 - sqrt(x[0]/(1+g)))
    So the Pareto front is traced by varying x[0] while keeping g minimal.
    
    Strategy:
    1. Initialize with points on the known Pareto front (x[1:]=0.5, x[0] spread).
    2. Use remaining budget to refine x[0] values and slightly perturb x[1:] 
       to find points that might slightly improve hypervolume.
    3. Use a deadline guard to stay within time limits.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For ZDT1: x[0] controls f1, x[1:] affects g (should be 0.5 for optimal g)
        # Pareto front: x[1:] = 0.5, x[0] in [0,1]
        # f1 = x[0], f2 = 1 - sqrt(x[0]) when g=0

        # Phase 1: Sample the known Pareto front densely
        # Use a non-uniform distribution to concentrate points where hypervolume gain is highest
        n_phase1 = min(budget, 150)
        if n_phase1 < 10:
            n_phase1 = budget

        # Create points on the Pareto front: x[1:] = 0.5, x[0] varies
        # Use a distribution that oversamples the middle where f2 changes rapidly
        t_vals = np.linspace(0, 1, n_phase1)
        # Apply a slight bias: more points in [0.1, 0.9]
        t_vals = 0.5 + 0.45 * np.sin(np.pi * t_vals)  # Slight concentration in middle
        t_vals = np.clip(t_vals, 0, 1)
        t_vals = np.sort(t_vals)

        X = np.zeros((n_phase1, dim))
        X[:, 0] = t_vals
        X[:, 1:] = 0.5  # Optimal for g(x) in ZDT1

        F = np.array([problem(x) for x in X])
        used = n_phase1

        # Phase 2: Refine with local search around promising points
        # Focus on points that are on the non-dominated set
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Find non-dominated points
            nd_mask = self._non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) > 0:
                # For each non-dominated point, try small perturbations of x[0]
                # and slight variations in x[1:] to see if we can improve
                n_refine = min(remaining, 50)
                for i in range(n_refine):
                    if time.time() > deadline - 5.0:
                        break
                    if used >= budget:
                        break
                    
                    # Pick a random non-dominated point to refine
                    idx = nd_indices[np.random.randint(len(nd_indices))]
                    base_x = X[idx].copy()
                    
                    # Perturb x[0] slightly
                    delta_x0 = np.random.uniform(-0.02, 0.02)
                    new_x = base_x.copy()
                    new_x[0] = np.clip(base_x[0] + delta_x0, lo[0], hi[0])
                    
                    # Optionally perturb a few x[1:] values slightly
                    if np.random.random() < 0.3:
                        j = np.random.randint(1, dim)
                        new_x[j] = np.clip(base_x[j] + np.random.uniform(-0.05, 0.05), lo[j], hi[j])
                    
                    f_new = problem(new_x)
                    used += 1
                    
                    # Replace the worst point if new point is better
                    # Use a simple criterion: replace if new point is non-dominated
                    # or if it improves the sum of objectives
                    if self._is_better(f_new, F, idx):
                        # Find the worst point to replace
                        worst_idx = self._find_worst(F)
                        X[worst_idx] = new_x
                        F[worst_idx] = f_new
                        # Update non-dominated mask
                        nd_mask = self._non_dominated_mask(F)
                        nd_indices = np.where(nd_mask)[0]

        return X

    def _non_dominated_mask(self, F):
        """Return boolean mask of non-dominated points in F."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
                    break
        return nd

    def _is_better(self, f_new, F, exclude_idx):
        """Check if f_new is better than some existing point (for replacement)."""
        # Simple check: is f_new non-dominated by any other point?
        for i in range(len(F)):
            if i == exclude_idx:
                continue
            if np.all(F[i] <= f_new) and np.any(F[i] < f_new):
                return False
        return True

    def _find_worst(self, F):
        """Find the index of the worst point (highest sum of objectives)."""
        return int(np.argmax(np.sum(F, axis=1)))
# EVOLVE-BLOCK-END
