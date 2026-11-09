"""Multi-objective optimizer using deterministic multi-start local optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 30s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Stratified sampling for initial population
        # We want to use about 50-70% of budget for global exploration, rest for local
        # Let's pick 40 initial points (20% of budget) to start
        n_init = 40
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Generate n_init points
        points = sampler.random(n_init)
        X_init = lo + (hi - lo) * points
        
        # Evaluate initial points
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Store all evaluated points and their objectives
        all_X = list(X_init)
        all_F = list(F_init)
        
        # Define scalarized objectives for local optimization
        # We will run local minimization on each of the n_init points, 
        # alternating between objective 0 and objective 1 to cover the Pareto front.
        
        # To ensure we don't exceed budget, we calculate max steps per local run.
        # Each local run consumes evaluations until convergence or budget.
        
        # Strategy: For each initial point, run a local minimization for obj 0, 
        # then for obj 1. This creates a "ladder" of points along the front.
        
        # We need to be careful with the budget.
        # Let's allocate remaining budget to local steps.
        remaining_budget = self.budget - used
        
        # Number of local optimizations to run
        # We have n_init points. We can run 2 local optimizations per point (one per objective).
        # Total potential local runs = 2 * n_init = 80.
        # Average budget per local run = remaining_budget / 80 = 160 / 80 = 2.
        # This is too low for L-BFGS-B to converge well.
        
        # Better strategy: Pick the best few points from the initial set and run deeper local optimizations.
        # Or, run local optimizations on all points but with a tight step limit, 
        # effectively doing a "steepest descent" step.
        
        # Let's try: For each initial point, perform a few steps of gradient descent 
        # (approximated by finite differences or just using scipy minimize with maxiter=1 or 2)
        # Actually, scipy minimize with maxiter=1 is just one step.
        
        # Alternative: Use the initial points as seeds for local searches, but limit the number of 
        # full local searches to fit the budget.
        # Let's do 10 full local searches for obj 0 and 10 for obj 1.
        # 20 local searches * 10 evals each = 200 evals. Total 40 + 200 = 240 > 200.
        
        # Revised Plan:
        # 1. Generate 20 initial points (Sobol). Eval 20.
        # 2. For each of the 20 points, run 1 local step for obj 0 (eval 20).
        # 3. For each of the 20 points, run 1 local step for obj 1 (eval 20).
        # Total 60. Still lots of budget left.
        
        # Let's be more aggressive.
        # 1. Generate 10 initial points. Eval 10.
        # 2. Run local minimization on each of the 10 points for obj 0. (Allow up to 15 evals each -> 150)
        # 3. Run local minimization on each of the 10 points for obj 1. (Allow up to 15 evals each -> 150)
        # Total 10 + 150 + 150 = 310. Too many.
        
        # Let's stick to a robust multi-start with limited local steps.
        # 1. Generate 20 initial points. Eval 20.
        # 2. For each point, run minimize on obj 0 with maxiter=5. (5 evals/point * 20 = 100)
        # 3. For each point, run minimize on obj 1 with maxiter=5. (5 evals/point * 20 = 100)
        # Total 220. Slightly over.
        
        # Adjust:
        # 1. Generate 15 initial points. Eval 15.
        # 2. For each point, run minimize on obj 0 with maxiter=6. (6 * 15 = 90)
        # 3. For each point, run minimize on obj 1 with maxiter=6. (6 * 15 = 90)
        # Total 195. Safe.
        
        n_starts = 15
        points = sampler.random(n_starts)
        X_init = lo + (hi - lo) * points
        F_init = np.array([problem(x) for x in X_init])
        used = n_starts
        
        all_X = list(X_init)
        all_F = list(F_init)
        
        max_iter_local = 6
        
        for i in range(n_starts):
            if used >= self.budget or time.time() > deadline:
                break
            
            # Local minimization for Objective 0
            def obj0_func(x):
                nonlocal used
                if used >= self.budget or time.time() > deadline:
                    return 0.0
                f = problem(x)
                used += 1
                all_X.append(x)
                all_F.append(f)
                return f[0]
            
            res0 = minimize(obj0_func, X_init[i], method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter_local})
            
            if used >= self.budget or time.time() > deadline:
                break
            
            # Local minimization for Objective 1
            def obj1_func(x):
                nonlocal used
                if used >= self.budget or time.time() > deadline:
                    return 0.0
                f = problem(x)
                used += 1
                all_X.append(x)
                all_F.append(f)
                return f[1]
            
            res1 = minimize(obj1_func, X_init[i], method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter_local})
            
        # The evaluator records ALL returned objective vectors.
        # The problem statement says: "Every returned objective vector is recorded".
        # My `problem` wrapper inside the local minimization calls `problem(x)` and appends to `all_X/all_F`.
        # However, the evaluator's `problem` function is the one passed to `__call__`.
        # When I call `problem(x)` inside `obj0_func`, the evaluator's wrapper records it.
        # I do NOT need to return `all_F` because the evaluator ignores the return value of `__call__`
        # but records every call to the `problem` function provided by the evaluator.
        
        # Wait, the contract says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This implies the evaluator wraps the `problem` function such that every call is logged.
        # So I just need to make sure I call `problem` exactly 200 times (or less) and don't crash.
        
        return None
# EVOLVE-BLOCK-END
