import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Grid-based decomposition: divide each dimension into 2 bins
        # Total cells = 2^12 = 4096, but we only have budget for ~200 evals
        # So we use a coarser approach: initialize with LHS-like spread, then local search
        
        # Initialize with Latin Hypercube Sampling for maximal spatial coverage
        N_init = 50
        X_init = self._lhs_sample(N_init, dim, lo, hi)
        
        # Evaluate initial points
        F_init = np.array([problem(x) for x in X_init])
        eval_count = N_init
        
        # Local search on each initial point to improve it
        # We'll do a simple coordinate descent / Nelder-Mead hybrid
        X_best = X_init.copy()
        F_best = F_init.copy()
        
        # Phase 1: Local search on all points
        # Budget remaining: 200 - 50 = 150
        # We can do ~3 local search steps per point (150/50 = 3)
        local_budget_per_point = 3
        
        for i in range(N_init):
            if time.time() > deadline - 10:
                break
            if eval_count >= self.budget:
                break
            
            x = X_best[i].copy()
            f = F_best[i].copy()
            
            # Simple local search: try small perturbations in random directions
            for step in range(local_budget_per_point):
                if time.time() > deadline - 5:
                    break
                if eval_count >= self.budget:
                    break
                
                # Generate a random direction
                direction = np.random.randn(dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                
                # Try a small step
                step_size = 0.1 * (hi - lo)
                x_trial = x + direction * step_size
                x_trial = np.clip(x_trial, lo, hi)
                x_trial = np.nan_to_num(x_trial, nan=(lo + hi) / 2.0, posinf=hi, neginf=lo)
                
                f_trial = problem(x_trial)
                eval_count += 1
                
                # Accept if it's better in at least one objective and not worse in others
                # (simple dominance check)
                if self._is_better_or_equal(f_trial, f):
                    x = x_trial
                    f = f_trial
                    X_best[i] = x
                    F_best[i] = f
        
        # Phase 2: Use remaining budget for global exploration
        # Add new random points and local search them
        remaining_budget = self.budget - eval_count
        
        if remaining_budget > 10 and time.time() < deadline - 20:
            # Generate new random points
            N_new = min(remaining_budget // 4, 20)  # Leave budget for local search
            if N_new > 0:
                X_new = lo + (hi - lo) * np.random.random((N_new, dim))
                X_new = np.nan_to_num(X_new, nan=(lo + hi) / 2.0, posinf=hi, neginf=lo)
                F_new = np.array([problem(x) for x in X_new])
                eval_count += N_new
                
                # Local search on new points
                local_budget_per_new = max(1, (self.budget - eval_count) // N_new)
                for i in range(N_new):
                    if time.time() > deadline - 5:
                        break
                    if eval_count >= self.budget:
                        break
                    
                    x = X_new[i].copy()
                    f = F_new[i].copy()
                    
                    for step in range(local_budget_per_new):
                        if time.time() > deadline - 5:
                            break
                        if eval_count >= self.budget:
                            break
                        
                        direction = np.random.randn(dim)
                        direction /= (np.linalg.norm(direction) + 1e-10)
                        step_size = 0.05 * (hi - lo)
                        x_trial = x + direction * step_size
                        x_trial = np.clip(x_trial, lo, hi)
                        x_trial = np.nan_to_num(x_trial, nan=(lo + hi) / 2.0, posinf=hi, neginf=lo)
                        
                        f_trial = problem(x_trial)
                        eval_count += 1
                        
                        if self._is_better_or_equal(f_trial, f):
                            x = x_trial
                            f = f_trial
                    
                    X_new[i] = x
                    F_new[i] = f
                
                # Combine all points
                X_all = np.vstack([X_best, X_new])
                F_all = np.vstack([F_best, F_new])
            else:
                X_all = X_best
                F_all = F_best
        else:
            X_all = X_best
            F_all = F_best
        
        # Final: Keep only non-dominated points for maximum hypervolume
        # But since the evaluator records ALL returned objective vectors,
        # we need to return points that are all useful
        # Actually, the contract says "Every returned objective vector is recorded"
        # and the return value is ignored. So we just need to ensure we've made
        # the evaluations. The recording happens on the evaluator side.
        # We just need to not exceed budget and not crash.
        
        return X_all

    def _lhs_sample(self, n, dim, lo, hi):
        """Latin Hypercube Sampling"""
        X = np.zeros((n, dim))
        for i in range(dim):
            # Generate n random permutations
            perm = np.random.permutation(n)
            # Map to [0, 1]
            X[:, i] = (perm + np.random.random(n)) / n
            # Map to [lo, hi]
            X[:, i] = lo[i] + X[:, i] * (hi[i] - lo[i])
        
        # Ensure finite
        X = np.nan_to_num(X, nan=(lo + hi) / 2.0, posinf=hi, neginf=lo)
        return X

    def _is_better_or_equal(self, f1, f2):
        """Check if f1 is better than or equal to f2 (minimization)"""
        # f1 is better if all(f1 <= f2) and any(f1 < f2)
        # For local search, we accept if f1 is not worse in any objective
        # and better in at least one
        return np.all(f1 <= f2) and np.any(f1 < f2)
