"""
Hybrid Adaptive Coordinate Descent and Nelder-Mead Optimizer
for BBOB-style Schwefel f20 on [-5, 5]^5.

Combines the robust global exploration of multi-start sampling with the
fast local convergence of adaptive coordinate descent, as suggested by
high-scoring inspirations.
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = None
        best_f = np.inf

        def count_call():
            nonlocal calls
            calls += 1

        def safe_eval(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            try:
                x = np.asarray(x, dtype=float)
                if x.shape != (dim,):
                    x = x.flatten()[:dim]
                # Clip to box [-5, 5]
                x = np.clip(x, -5.0, 5.0)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                calls += 1
                return np.inf

        # Phase 1: Global Exploration
        # We use a mix of the origin and random points to find a good starting basin.
        n_global_starts = 8
        starts = []
        
        # Always try the origin first (common BBOB heuristic)
        x0 = np.zeros(dim)
        f0 = safe_eval(x0)
        starts.append((f0, x0.copy()))
        
        # Add random points
        for _ in range(n_global_starts - 1):
            if calls >= budget or time.time() > deadline:
                break
            x_rand = rng.uniform(-4.5, 4.5, size=dim)
            f_rand = safe_eval(x_rand)
            starts.append((f_rand, x_rand.copy()))

        # Sort starts by fitness
        starts.sort(key=lambda x: x[0])
        
        if not starts:
            return np.zeros(dim), np.inf

        # Phase 2: Local Refinement with Adaptive Coordinate Descent
        # We refine the best few points. The adaptive CD is very efficient locally.
        n_refine = 3
        # Allocate remaining budget for refinement
        # Reserve a small amount for final polish
        reserve_for_polish = 20
        available_budget = budget - calls - reserve_for_polish
        if available_budget < 50:
            # If budget is too tight, just use what's left for one refinement
            n_refine = 1
            budget_per_refine = max(10, available_budget)
        else:
            budget_per_refine = available_budget // n_refine

        for i in range(min(n_refine, len(starts))):
            if calls >= budget or time.time() > deadline:
                break
            
            x = starts[i][1].copy()
            fx = starts[i][0]
            
            # If this start is already worse than the current best by a lot, maybe skip?
            # But let's just try to improve it.
            
            # Adaptive Coordinate Descent
            steps = np.ones(dim) * 1.5
            no_improve = 0
            max_no_improve = 10
            max_iter = budget_per_refine
            
            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                improved_this_pass = False
                # Cycle through dimensions
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Try positive direction
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] + steps[d], -5.0, 5.0)
                    f_trial = safe_eval(x_trial)
                    
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        steps[d] *= 1.3  # Increase step
                        improved_this_pass = True
                        no_improve = 0
                    else:
                        steps[d] *= 0.7  # Decrease step
                        if steps[d] < 0.01:
                            steps[d] = 0.01
                    
                    # Try negative direction (optional, but helps if we overshot)
                    # To save budget, we only do this if the positive step was small or failed
                    # Actually, standard adaptive CD often just adjusts the step and tries again in next iter.
                    # But trying both directions immediately can be more robust.
                    # Let's try negative only if we haven't improved in this dimension recently.
                    # For simplicity and speed, let's stick to the inspiration's logic:
                    # The inspiration tried both signs in a loop.
                    # Let's try the negative sign as well if budget allows.
                    
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    x_trial_neg = x.copy()
                    # Note: x has been updated if positive worked. 
                    # If positive worked, we moved. If not, x is same.
                    # We should probably check from the *current* best x.
                    # Let's refine the logic: 
                    # 1. Try +step. If better, update x.
                    # 2. Try -step from NEW x? Or from OLD x?
                    # Usually, you try +, if fail, try -.
                    # If + worked, we are at new x. Trying -step from new x might go back.
                    # Let's just do a simple hill climb:
                    # Try +step. If better, update.
                    # Else, try -step. If better, update.
                    # Else, shrink step.
                    
                    # My previous logic updated x if + worked. 
                    # If + didn't work, x is unchanged.
                    # Let's try -step from the current x (which is the best so far in this dim).
                    
                    x_trial_neg = x.copy()
                    x_trial_neg[d] = np.clip(x_trial_neg[d] - steps[d], -5.0, 5.0)
                    f_trial_neg = safe_eval(x_trial_neg)
                    
                    if f_trial_neg < fx:
                        x = x_trial_neg
                        fx = f_trial_neg
                        steps[d] *= 1.3
                        improved_this_pass = True
                        no_improve = 0
                    else:
                        # Both failed, step is likely too big or we are at a local min
                        # Step was already halved if + failed. Halve again?
                        # The inspiration halved on fail. I halved on + fail.
                        # If - also fails, we should probably shrink more.
                        steps[d] *= 0.7
                        if steps[d] < 0.01:
                            steps[d] = 0.01

                if improved_this_pass:
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        # Random perturbation to escape local minima
                        x = np.clip(x + rng.normal(0, 0.5, dim), -5.0, 5.0)
                        fx = safe_eval(x)
                        no_improve = 0
                        steps = np.maximum(steps, 0.5)
                        # If this perturbation was bad, we might want to revert?
                        # But safe_eval updates best_x if it's better.
                        # We continue from this new x.

        # Phase 3: Final Polish with Nelder-Mead
        # Use the best point found so far
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                try:
                    res = minimize(
                        safe_eval,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'maxfev': remaining,
                            'xatol': 1e-10,
                            'fatol': 1e-10
                        }
                    )
                    # safe_eval already updates best_x and best_f
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_eval(best_x) # Ensure at least one call if none made (though Phase 1 ensures this)

        return best_x, best_f
