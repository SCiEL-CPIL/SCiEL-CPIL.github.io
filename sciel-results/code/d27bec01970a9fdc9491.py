"""Multi-objective EGO optimizer for RE21: Pareto-based search with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Pareto-based multi-objective search with adaptive mutation and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, f, F):
        """Check if f is dominated by any point in F."""
        if len(F) == 0:
            return False
        return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

    def _pareto_tournament(self, F, n_candidates=2):
        """Select a parent using Pareto tournament selection."""
        n = len(F)
        if n <= n_candidates:
            return np.random.randint(n)
        candidates = np.random.choice(n, size=n_candidates, replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if self._is_dominated(F[best], F[c]):
                best = c
        return best

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial population
        pop_size = min(30, self.budget // 3)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            dominated = np.array([self._is_dominated(F[i], F) for i in range(len(F))])
            parents = X[~dominated] if np.any(~dominated) else X
            
            # Select parent
            parent_idx = self._pareto_tournament(F, n_candidates=2)
            parent = X[parent_idx]
            
            # Adaptive mutation: step size based on spread
            spread = (hi - lo) * 0.1
            child = parent + spread * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point (by sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
            
            # Local refinement on non-dominated points occasionally
            if used % 10 == 0 and len(F) > 0:
                non_dom_idx = np.where(~np.array([self._is_dominated(F[i], F) for i in range(len(F))]))[0]
                if len(non_dom_idx) > 0:
                    refine_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
                    x0 = X[refine_idx]
                    try:
                        res = minimize(
                            lambda x: np.sum(problem(x)),
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4}
                        )
                        if res.success:
                            fx = problem(res.x)
                            if not self._is_dominated(fx, F):
                                worst = int(np.argmax(np.sum(F, axis=1)))
                                X[worst], F[worst] = res.x, fx
                                used += 1
                    except:
                        pass
        
        return X
# EVOLVE-BLOCK-END
