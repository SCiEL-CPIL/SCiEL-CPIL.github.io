"""Optimizer for BBOB f15 (Rastrigin rotated) using lattice-based candidate generation."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Lattice-based optimizer exploiting the 2π-periodicity of Rastrigin in the rotated frame.
    
    Strategy:
    1. Initial exploration: evaluate at a small set of random points to get an initial best estimate.
    2. Lattice search: generate candidate minima by shifting the best point by integer multiples
       of a step size in each coordinate direction. This approximates the lattice of minima
       in the rotated frame.
    3. Iteratively refine: from the best candidate found, generate new candidates and evaluate
       them until the budget is exhausted.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = np.inf
        self.evaluated = set()

    def _counted_func(self, x):
        """Wrapper to count function calls."""
        self.calls += 1
        return self.func(x)

    def _make_key(self, x):
        """Create a hashable key for a point."""
        return tuple(np.round(x, 8))

    def _evaluate(self, x):
        """Evaluate a point if not already evaluated and within budget."""
        if self.calls >= self.budget:
            return np.inf
        key = self._make_key(x)
        if key in self.evaluated:
            return np.inf
        self.evaluated.add(key)
        fx = self._counted_func(x)
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return fx

    def __call__(self, func):
        self.func = func
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.evaluated = set()

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial exploration - evaluate at random points and origin
        n_initial = min(20, budget // 2)
        for i in range(n_initial):
            if self.calls >= budget - 10:
                break
            if i == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            self._evaluate(x0)

        # Phase 2: Lattice-based search
        # Generate candidates by shifting the best point by integer steps in each coordinate
        # Step size: use 1.0 as a reasonable approximation for the lattice spacing
        step_sizes = [1.0, 2.0, 0.5]  # Try multiple step sizes
        
        for step in step_sizes:
            if self.calls >= budget - 5:
                break
            
            # Generate candidates from the current best point
            if self.best_x is None:
                break
                
            candidates = []
            for d in range(dim):
                for delta in [-2 * step, -1 * step, 1 * step, 2 * step]:
                    x_cand = self.best_x.copy()
                    x_cand[d] += delta
                    x_cand = np.clip(x_cand, -5, 5)
                    candidates.append(x_cand)
            
            # Also try combinations of shifts in multiple dimensions
            if self.calls < budget - 20:
                for d1 in range(dim):
                    for d2 in range(d1 + 1, dim):
                        for delta1 in [-step, step]:
                            for delta2 in [-step, step]:
                                x_cand = self.best_x.copy()
                                x_cand[d1] += delta1
                                x_cand[d2] += delta2
                                x_cand = np.clip(x_cand, -5, 5)
                                candidates.append(x_cand)
            
            # Evaluate candidates, sorted by distance to best (closest first)
            candidates.sort(key=lambda x: np.linalg.norm(x - self.best_x))
            for x_cand in candidates:
                if self.calls >= budget - 1:
                    break
                self._evaluate(x_cand)
                
                # If we found a better point, generate more candidates from it
                if self.best_x is not None and self.calls < budget - 10:
                    # Generate a few more candidates from the new best
                    for d in range(dim):
                        for delta in [-step, step]:
                            x_new = self.best_x.copy()
                            x_new[d] += delta
                            x_new = np.clip(x_new, -5, 5)
                            if self.calls < budget - 1:
                                self._evaluate(x_new)

        # Ensure we've made at least one call
        if self.calls == 0:
            x0 = np.zeros(dim)
            self._evaluate(x0)

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
