"""Multi-objective optimizer for ZDT1 using grid-free local searches along the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Grid-free continuous multi-objective optimization for ZDT1.
    
    Strategy:
    1. Select multiple target points along the Pareto front (x1 values).
    2. For each target, initialize x with x1=target and x[1:]=0.0.
    3. Use a local optimizer (Powell or Nelder-Mead) on x[1:] to minimize f2, keeping x[0] fixed.
    4. Collect all evaluated points into an archive for hypervolume computation.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._eval_count = 0
        self._archive = []

    def _reset(self):
        self._eval_count = 0
        self._archive = []

    def _eval_and_record(self, problem, x):
        """Evaluate x, record the result, and return the objective vector."""
        x = np.asarray(x, dtype=np.float64)
        f = np.asarray(problem(x), dtype=np.float64)
        self._eval_count += 1
        self._archive.append((x.copy(), f.copy()))
        return f

    def _optimize_tail_powell(self, problem, x_fixed, bounds, budget_for_this):
        """
        Optimizes x[1:] given x[0] fixed using Powell's method.
        Returns best x, best f, evals used.
        """
        dim_tail = self.dim - 1
        if dim_tail == 0:
            x = np.zeros(self.dim)
            x[0] = x_fixed
            f = self._eval_and_record(problem, x)
            return x, f, 1

        lo_tail = np.array([b[0] for b in bounds[1:]])
        hi_tail = np.array([b[1] for b in bounds[1:]])
        mid_tail = (lo_tail + hi_tail) / 2.0

        # Initial point: all zeros (which is on the Pareto front for ZDT1)
        y0 = np.zeros(dim_tail)
        y0 = np.clip(y0, lo_tail, hi_tail)

        def objective(y_vec):
            x = np.zeros(self.dim)
            x[0] = x_fixed
            x[1:] = y_vec
            f = self._eval_and_record(problem, x)
            return f[1]  # minimize f2

        # Check if we have budget for at least one evaluation
        if self._eval_count >= self.budget:
            f0 = objective(y0)
            x_best = np.zeros(self.dim)
            x_best[0] = x_fixed
            x_best[1:] = y0
            return x_best, np.array([x_fixed, f0]), 1

        # Define bounds for Powell
        bounds_tail = list(zip(lo_tail, hi_tail))

        # Use Powell's method with maxiter limited by remaining budget
        # Powell uses roughly dim * maxiter evaluations, so limit maxiter
        maxiter = max(1, (budget_for_this - 1) // max(1, dim_tail))

        try:
            result = minimize(
                objective, y0, method='Powell',
                bounds=bounds_tail,
                options={'maxiter': maxiter, 'xtol': 1e-8, 'ftol': 1e-10}
            )
        except Exception:
            result = None

        if result is not None and result.x is not None:
            y_best = np.clip(result.x, lo_tail, hi_tail)
        else:
            y_best = y0

        # Evaluate the best point to get the full f vector
        x_best = np.zeros(self.dim)
        x_best[0] = x_fixed
        x_best[1:] = y_best
        f_best = self._eval_and_record(problem, x_best)

        return x_best, f_best, 0  # evals already counted in _eval_and_record

    def _optimize_tail_nm(self, problem, x_fixed, bounds, budget_for_this):
        """
        Optimizes x[1:] given x[0] fixed using Nelder-Mead.
        Returns best x, best f, evals used.
        """
        dim_tail = self.dim - 1
        if dim_tail == 0:
            x = np.zeros(self.dim)
            x[0] = x_fixed
            f = self._eval_and_record(problem, x)
            return x, f, 1

        lo_tail = np.array([b[0] for b in bounds[1:]])
        hi_tail = np.array([b[1] for b in bounds[1:]])

        # Initial point
        y0 = np.zeros(dim_tail)
        y0 = np.clip(y0, lo_tail, hi_tail)

        def objective(y_vec):
            x = np.zeros(self.dim)
            x[0] = x_fixed
            x[1:] = y_vec
            f = self._eval_and_record(problem, x)
            return f[1]

        if self._eval_count >= self.budget:
            f0 = objective(y0)
            x_best = np.zeros(self.dim)
            x_best[0] = x_fixed
            x_best[1:] = y0
            return x_best, np.array([x_fixed, f0]), 1

        # Nelder-Mead with maxiter limited by remaining budget
        maxiter = max(1, (budget_for_this - 1) // 2)

        try:
            result = minimize(
                objective, y0, method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10,
                         'adaptive': True}
            )
        except Exception:
            result = None

        if result is not None and result.x is not None:
            y_best = np.clip(result.x, lo_tail, hi_tail)
        else:
            y_best = y0

        x_best = np.zeros(self.dim)
        x_best[0] = x_fixed
        x_best[1:] = y_best
        f_best = self._eval_and_record(problem, x_best)

        return x_best, f_best, 0

    def __call__(self, problem, bounds):
        self._reset()
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Number of targets along the Pareto front
        # Each target needs at least ~dim evaluations for initialization
        # With 200 budget and dim=30, we can afford about 4-5 targets
        n_targets = 5
        
        # Check if budget is sufficient
        min_cost_per_target = self.dim + 2
        if self.budget < n_targets * min_cost_per_target:
            n_targets = max(1, self.budget // min_cost_per_target)
        
        # Targets for x1 (which equals f1 on the Pareto front)
        # Spread evenly from near 0 to near 1
        x1_targets = np.linspace(0.02, 0.98, n_targets)
        
        # Distribute budget among targets
        budget_per_target = self.budget // n_targets
        
        for i, t in enumerate(x1_targets):
            if self._eval_count >= self.budget:
                break
            
            remaining = self.budget - self._eval_count
            # Give each target roughly equal budget, with the last one getting the rest
            if i == n_targets - 1:
                budget_for_this = remaining
            else:
                budget_for_this = min(budget_per_target, remaining)
            
            if budget_for_this < 2:
                # Not enough budget for meaningful optimization
                x = np.zeros(self.dim)
                x[0] = t
                self._eval_and_record(problem, x)
                continue
            
            # Alternate between Powell and Nelder-Mead for diversity
            if i % 2 == 0:
                self._optimize_tail_powell(problem, t, bounds, budget_for_this)
            else:
                self._optimize_tail_nm(problem, t, bounds, budget_for_this)
        
        # If we have remaining budget, do some random perturbations around best points
        if self._eval_count < self.budget and self._archive:
            # Find the best point in the archive (lowest f2)
            best_f2 = min(f[1] for _, f in self._archive)
            best_x = None
            for x, f in self._archive:
                if f[1] == best_f2:
                    best_x = x
                    break
            
            if best_x is not None:
                # Do a few random perturbations
                n_extra = min(self.budget - self._eval_count, 10)
                for _ in range(n_extra):
                    if self._eval_count >= self.budget:
                        break
                    x_new = best_x.copy()
                    # Perturb a random subset of tail variables
                    n_perturb = max(1, self.dim // 5)
                    idx = np.random.choice(range(1, self.dim), size=n_perturb, replace=False)
                    for j in idx:
                        x_new[j] += np.random.uniform(-0.05, 0.05)
                        x_new[j] = np.clip(x_new[j], lo[j], hi[j])
                    self._eval_and_record(problem, x_new)
        
        return np.array([a[0] for a in self._archive])
# EVOLVE-BLOCK-END
