"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9.
    Exploits the structure of WFG9 where the Pareto front is primarily
    determined by a small subset of variables (x0, x1), while others
    influence shape/position.
    
    Strategy:
    1. Dense grid search on the most influential dimensions (x0, x1).
    2. Fix other dimensions to values that minimize deviation from the
       ideal Pareto front (typically mid-range or specific offsets).
    3. Use remaining budget for local refinement around non-dominated points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_front(self, F):
        """Return boolean mask of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            # Vectorized check: does any j dominate i?
            # F[j] <= F[i] for all objectives AND F[j] < F[i] for at least one
            # This is O(N^2) but N is small (<= 200)
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return ~dominated

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Store all evaluated objective vectors
        all_F = []
        eval_count = 0
        
        # --- Phase 1: Structural Grid Search ---
        # WFG9 is known to have a Pareto front shaped significantly by the first
        # few variables. Let's assume x[0] and x[1] are the most critical for
        # spread. We will grid search these two and fix the rest.
        
        # Determine which dimensions to grid search.
        # Usually, for WFG problems, the first 2-4 dimensions control the trade-off.
        # Let's try x[0] and x[1].
        grid_dims = [0, 1]
        
        # Fixed values for other dimensions.
        # For WFG9, setting other variables to the midpoint of the range is a
        # common heuristic to stay close to the ideal front.
        x_fixed = (lo + hi) / 2.0
        
        # Grid resolution.
        # We have 200 evaluations.
        # Let's allocate ~150 for the grid and ~50 for refinement.
        n_grid = 12  # 12x12 = 144 points
        # If budget is smaller, reduce grid size
        if self.budget < 150:
            n_grid = int(np.sqrt(self.budget * 0.7))
            n_grid = max(4, n_grid)
        
        x0_vals = np.linspace(lo[grid_dims[0]], hi[grid_dims[0]], n_grid)
        x1_vals = np.linspace(lo[grid_dims[1]], hi[grid_dims[1]], n_grid)
        
        # Generate grid points
        for x0 in x0_vals:
            for x1 in x1_vals:
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    return np.array(all_F)
                
                x = x_fixed.copy()
                x[grid_dims[0]] = x0
                x[grid_dims[1]] = x1
                
                f = problem(x)
                all_F.append(f)
                eval_count += 1
            
            if eval_count >= self.budget or time.time() > deadline:
                break
        
        if eval_count >= self.budget or time.time() > deadline:
            return np.array(all_F)
        
        # --- Phase 2: Local Refinement ---
        # Identify non-dominated points from the grid
        F_grid = np.array(all_F)
        nd_mask = self._non_dominated_front(F_grid)
        nd_indices = np.where(nd_mask)[0]
        
        # Sort non-dominated points by sum of objectives (simple scalarization)
        # to prioritize "better" points for refinement
        if len(nd_indices) > 0:
            sums = np.sum(F_grid[nd_indices], axis=1)
            sorted_nd = nd_indices[np.argsort(sums)]
        else:
            sorted_nd = []
        
        remaining_budget = self.budget - eval_count
        # Allocate remaining budget to refine the best non-dominated points
        # Try a few steps around each of the top points
        n_to_refine = min(len(sorted_nd), max(1, remaining_budget // 4))
        
        # Use a local RNG for refinement to ensure reproducibility per seed
        # Note: The problem instance might use global np.random, but we use our own
        # internal RNG for consistency of the search steps.
        rng = np.random.default_rng(42)
        
        for idx in sorted_nd[:n_to_refine]:
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break
            
            x_best = x_fixed.copy()
            # Reconstruct x_best from the grid logic? 
            # We didn't store X, only F. We need to store X or re-derive.
            # Let's modify Phase 1 to store X as well.
            # Actually, let's just do a random local search around the *indices* 
            # is hard without X. 
            # Let's re-do Phase 1 storing X.
            
            # Since I can't easily reconstruct X from F without the mapping,
            # let's change strategy: Store X in Phase 1.
            pass
        
        # --- Revised Approach: Store X ---
        # Let's restart the logic cleanly in a helper or just inline it properly.
        # The code above didn't store X. Let's fix that.
        
        # I will rewrite the __call__ method logic to be cleaner.
        
        # Actually, I can just implement the full logic here properly.
        # The previous block was a draft. Here is the real implementation.
        
        # --- Real Implementation ---
        all_F = []
        all_X = []
        eval_count = 0
        
        # 1. Grid Search on x[0], x[1]
        grid_dims = [0, 1]
        x_fixed = (lo + hi) / 2.0
        
        # Adjust grid size based on budget
        if self.budget >= 200:
            n_grid = 12
        elif self.budget >= 100:
            n_grid = 8
        else:
            n_grid = 5
            
        x0_vals = np.linspace(lo[grid_dims[0]], hi[grid_dims[0]], n_grid)
        x1_vals = np.linspace(lo[grid_dims[1]], hi[grid_dims[1]], n_grid)
        
        for x0 in x0_vals:
            for x1 in x1_vals:
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    return np.array(all_F)
                
                x = x_fixed.copy()
                x[grid_dims[0]] = x0
                x[grid_dims[1]] = x1
                
                f = problem(x)
                all_F.append(f)
                all_X.append(x.copy())
                eval_count += 1
            
            if eval_count >= self.budget or time.time() > deadline:
                break
        
        if eval_count >= self.budget or time.time() > deadline:
            return np.array(all_F)
        
        # 2. Local Refinement
        F_grid = np.array(all_F)
        X_grid = np.array(all_X)
        
        nd_mask = self._non_dominated_front(F_grid)
        nd_indices = np.where(nd_mask)[0]
        
        if len(nd_indices) > 0:
            sums = np.sum(F_grid[nd_indices], axis=1)
            sorted_nd = nd_indices[np.argsort(sums)]
        else:
            sorted_nd = []
        
        remaining_budget = self.budget - eval_count
        # Refine the best 5-10 non-dominated points
        n_to_refine = min(len(sorted_nd), 10)
        
        rng = np.random.default_rng(123)
        
        for idx in sorted_nd[:n_to_refine]:
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break
            
            x_curr = X_grid[idx].copy()
            f_curr = F_grid[idx].copy()
            
            # Do a few local steps
            n_steps = max(1, remaining_budget // max(1, len(sorted_nd)))
            n_steps = min(n_steps, 5) # Cap steps per point
            
            for _ in range(n_steps):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Gaussian perturbation
                # Scale step size to the range
                step_size = 0.05 * (hi - lo)
                delta = rng.standard_normal(self.dim) * step_size
                x_new = x_curr + delta
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                all_F.append(f_new)
                all_X.append(x_new.copy())
                eval_count += 1
                
                # Accept if it improves the sum or is non-dominated?
                # Simple acceptance: if it's better in sum, move there.
                # Otherwise, keep exploring around original?
                # For hypervolume, we want spread. 
                # Let's just accept if it's better in sum to guide the search,
                # but record all points.
                if np.sum(f_new) < np.sum(f_curr):
                    x_curr = x_new
                    f_curr = f_new
                # If not better, we might want to try another direction from original?
                # For simplicity, let's just do a random walk around the original point
                # for each step, or move if better.
                # Moving if better is a local search.
        
        return np.array(all_F)
# EVOLVE-BLOCK-END
