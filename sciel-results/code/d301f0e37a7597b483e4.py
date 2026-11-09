"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using non-uniform x1 sampling and efficient search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the global numpy random state (seeded by evaluator)
        # but we'll use a local rng for reproducibility within the call
        rng = np.random.default_rng()

        # ZDT4: x1 in [-5, 5], x2..x10 in [0, 1]
        # f1 = 1 + 8*(x1^2 - 6cos(12*pi*x1))  -- convex-ish but with oscillations
        # f2 = g(x) * h(f1, g)
        # The Pareto front is f2 = h(f1, g) where g depends on x2..x10
        # Key insight: x1 controls f1, and we want to cover the range of f1 well
        # f1 ranges from about 1 to 1+8*(25+6) = 249, but the Pareto front has f1 in [1, 1+8*25+48] 
        # Actually f1 = 1 + 8*(x1^2 - 6cos(12*pi*x1))
        # At x1=0: f1 = 1 + 8*(0-6) = 1-48 = -47? No wait, cos(0)=1 so 0-6*1=-6, f1=1-48=-47
        # Hmm, that doesn't seem right for a minimization problem. Let me reconsider.
        # Actually ZDT4 is defined as:
        # f1(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 - 10cos(20*pi*(xi-0.5))) for i=2..n
        # Wait, I need to be careful. The standard ZDT4:
        # f1 = 1 + 8*(x1^2 - 6*cos(12*pi*x1))  ... no that's not standard either.
        
        # Let me just focus on the strategy: non-uniform sampling of x1 biased toward lower values
        # Since the Pareto front of ZDT4 is convex and the interesting region is where f1 is small,
        # we should sample x1 more densely near 0 (or wherever f1 is minimal).
        
        # For ZDT4, f1 = 1 + 8*(x1^2 - 6*cos(12*pi*x1))
        # The minimum of x1^2 - 6*cos(12*pi*x1) occurs near x1=0 where it's -6, giving f1 = 1-48 = -47
        # But wait, that gives negative f1 which is odd. Let me just use the empirical finding:
        # "Non-uniform x1 sampling biased towards lower values better covers the convex Pareto front"
        
        # Strategy: 
        # 1. Generate diverse x1 values using a transformation that biases toward lower x1
        # 2. For each x1, optimize x2..x10 to minimize g(x) (which affects f2)
        # 3. Use a simple local search or direct evaluation strategy
        
        # Let's use a hybrid approach:
        # - Allocate budget to explore different x1 values
        # - For each x1, do a few evaluations with different x2..x10 to find good g values
        
        # First, let's understand the budget: 200 evaluations total
        # We need to be efficient.
        
        # Approach: 
        # 1. Sample x1 non-uniformly (biased toward values that give low f1)
        # 2. For each x1, sample x2..x10 to try to minimize g
        # 3. Keep track of all evaluated points and return them
        
        # The key for ZDT4 is that g(x) = sum((xi-0.5)^2 - 10cos(20*pi*(xi-0.5))) for i=2..n
        # The minimum of g is achieved when each xi = 0.5, giving g = sum(0 - 10cos(0)) = -10*(n-1)
        # But we want to MINIMIZE both objectives, and f2 = g * h(f1, g)
        # h(u,v) = 1 - sqrt(u/v) if v > 0, else 1
        # So we want g to be as small as possible (but positive for h to work well)
        
        # Actually, for the Pareto front, we want to find the trade-off between f1 and f2.
        # The ideal is to have g as small as possible while varying f1.
        
        # Let's use a direct approach:
        # - Generate a set of candidate x1 values with non-uniform distribution
        # - For each x1, set x2..x10 to values that minimize g (i.e., near 0.5)
        # - Add some perturbation to explore
        
        # Non-uniform x1 sampling: use a power transformation to bias toward lower x1
        # x1 in [-5, 5], but we want to bias toward x1 near 0 (where f1 is minimal)
        # Actually, the finding says "biased towards lower values" - this might mean lower f1 values
        # which corresponds to x1 near 0.
        
        # Let's sample x1 using: x1 = (u^(1/3) - 0.5) * 10 where u ~ Uniform(0,1)
        # This biases toward x1 near 0.
        
        # Actually, let me just use a simple strategy:
        # - Use 50% of budget for diverse x1 exploration
        # - Use 50% for local refinement
        
        n_evals = 0
        X_list = []
        F_list = []
        
        def evaluate(x):
            nonlocal n_evals
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            n_evals += 1
            return x_clipped, f
        
        # Phase 1: Diverse x1 exploration with good x2..x10
        # Sample x1 non-uniformly
        n_phase1 = int(budget * 0.6)
        n_phase2 = budget - n_phase1
        
        # Generate x1 values biased toward center (where f1 is low)
        # Use inverse CDF of a distribution centered at 0
        u = rng.random(n_phase1)
        # Transform to bias toward 0: x1 = 5 * (2*u - 1) would be uniform
        # Instead, use x1 = 5 * (u^(1/2) - 0.5) * 2 for bias toward 0? 
        # Let's use: x1 = 5 * sin(pi * u) which maps [0,1] to [0,5] with density peaked at center
        # Then center it: x1 = 5 * sin(pi * u) - 2.5? No, let's keep it simple.
        # The finding says "biased towards lower values" - let's interpret as lower x1 magnitude
        x1_vals = 5.0 * (2.0 * u - 1.0)  # uniform in [-5, 5]
        # Apply a transformation to bias toward 0:
        x1_vals = np.sign(x1_vals) * np.abs(x1_vals) ** 0.5 * 5.0 / np.sqrt(5.0)
        # This compresses large values and expands small ones
        x1_vals = np.clip(x1_vals, -5.0, 5.0)
        
        # For each x1, set x2..x10 near 0.5 (minimizes g) with some variation
        for i in range(n_phase1):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1_vals[i]
            # Set x2..x10 to 0.5 with small random perturbation
            x[1:] = 0.5 + 0.05 * rng.standard_normal(dim - 1)
            x[1:] = np.clip(x[1:], lo[1:], hi[1:])
            x, f = evaluate(x)
            X_list.append(x)
            F_list.append(f)
        
        # Phase 2: Local refinement around best points
        if n_evals < budget and len(F_list) > 0:
            F_arr = np.array(F_list)
            X_arr = np.array(X_list)
            
            # Find non-dominated points
            def is_dominated(f, F):
                # Check if f is dominated by any point in F
                for fj in F:
                    if np.all(fj <= f) and np.any(fj < f):
                        return True
                return False
            
            # Keep track of all points for non-dominated check
            all_X = list(X_list)
            all_F = list(F_list)
            
            # Sort by first objective to find good candidates
            order = np.argsort(F_arr[0])
            # Take top candidates for refinement
            n_refine = min(n_phase2, 30)
            for idx in order[:n_refine]:
                if time.time() > deadline or n_evals >= budget:
                    break
                x_base = X_arr[idx].copy()
                # Local search: perturb x1 and x2..x10
                for _ in range(3):
                    if n_evals >= budget or time.time() > deadline:
                        break
                    x_trial = x_base.copy()
                    # Perturb x1 more aggressively
                    x_trial[0] += 0.1 * rng.standard_normal()
                    # Perturb x2..x10 slightly
                    x_trial[1:] += 0.02 * rng.standard_normal(dim - 1)
                    x_trial = np.clip(x_trial, lo, hi)
                    x_trial, f_trial = evaluate(x_trial)
                    X_list.append(x_trial)
                    F_list.append(f_trial)
                    all_X.append(x_trial)
                    all_F.append(f_trial)
                    # If better, update base
                    if np.any(f_trial < F_arr[idx]):
                        x_base = x_trial.copy()
                        F_arr[idx] = f_trial
                        X_arr[idx] = x_trial
        
        # Return all evaluated points
        if len(X_list) > 0:
            return np.array(X_list)
        return np.array([])
# EVOLVE-BLOCK-END
