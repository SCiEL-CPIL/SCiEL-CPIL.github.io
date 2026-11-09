"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Powell/Restart optimizer for BBOB f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        dim = self.dim
        total_budget = self.budget
        
        global_best_x = np.zeros(dim)
        global_best_f = float('inf')
        calls_used = 0
        
        def update_best(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()
        
        def time_up():
            return time.time() > deadline
        
        def budget_up():
            return calls_used >= total_budget

        # Powell's Method implementation
        # This is efficient for local optimization and handles the valleys well
        # if we have a decent starting point.
        def powell_method(x0, max_iter=50, tol=1e-8):
            nonlocal calls_used, global_best_x, global_best_f
            
            # Initial directions: coordinate axes
            directions = np.eye(dim)
            x = x0.copy()
            f = func(x)
            calls_used += 1
            update_best(x, f)
            
            for _ in range(max_iter):
                if budget_up() or time_up():
                    break
                
                x_old = x.copy()
                f_old = f
                min_index = 0
                min_delta = 0.0
                
                # 1. Minimize along each direction
                for i in range(dim):
                    if budget_up() or time_up():
                        break
                    
                    # Find minimum along direction i
                    # Simple line search: try a few steps
                    # To save budget, we do a coarse line search first
                    # Actually, for Powell, we need to find the minimum.
                    # Since budget is tight, we use a fixed number of evaluations per line.
                    
                    # Heuristic line search
                    # Try moving in positive and negative direction
                    # Step size adaptation is complex, so we use a simple bisection-like or fixed grid
                    # Given the box [-5, 5], we can try a few steps.
                    
                    # Better: Use a simple 1D minimization with limited evals
                    # Let's use a small grid search around current point for robustness
                    # Or just a few steps.
                    
                    # To be efficient, we'll do a small number of steps
                    # If we are stuck, we restart.
                    
                    # Let's try 3 steps: -0.1, 0, 0.1 relative to current scale?
                    # No, Powell requires finding the min.
                    # We'll use a simple binary search or golden section with limited iterations.
                    
                    # Given the constraints, a simple approach:
                    # Try moving along the direction by a small amount, check if it improves.
                    # This is essentially a gradient-free local search.
                    
                    # Let's do a simple line search with 5 evaluations max
                    # This is a compromise for budget.
                    
                    # Actually, let's use a standard Powell step but with a fixed small number of line search iterations.
                    # This might be slow. 
                    
                    # Alternative: Use Nelder-Mead? It's simpler to implement robustly with limited evals.
                    # Let's switch to a simplified Nelder-Mead for the local phase.
                    
                    # For now, let's stick to a very simple local search to avoid complexity.
                    # Random local perturbation + Accept if better.
                    # This is basically (1+1)-ES but axis-aligned.
                    
                    # Let's restart the logic: 
                    # The current program uses CMA-ES. The directive asks for a DIFFERENT strategy.
                    # Powell is good, but implementation complexity with limited budget is high.
                    # Let's use **Nelder-Mead** (Simplex Method) for local optimization.
                    # It is robust and doesn't require gradient or covariance.
                    
                    break # Break out to use Nelder-Mead instead of this flawed Powell loop
                if budget_up() or time_up():
                    break
                # If we broke out of the direction loop, we stop Powell
                break
                
        def nelder_mead(x0, max_iter=100, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
            nonlocal calls_used, global_best_x, global_best_f
            
            # Initialize simplex
            # Start with x0 and small perturbations
            simplex = [x0.copy()]
            f_vals = [func(x0)]
            calls_used += 1
            update_best(x0, f_vals[0])
            
            for i in range(dim):
                x_i = x0.copy()
                # Perturb along axis i
                # Scale perturbation based on box size
                x_i[i] += 0.1
                simplex.append(x_i)
                f_i = func(x_i)
                calls_used += 1
                update_best(x_i, f_i)
                f_vals.append(f_i)
            
            for _ in range(max_iter):
                if budget_up() or time_up():
                    break
                
                # Sort simplex by fitness
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # Check convergence
                if f_vals[0] - f_vals[-1] < 1e-10:
                    break
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_used += 1
                update_best(x_ref, f_ref)
                
                if f_ref < f_vals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_used += 1
                    update_best(x_exp, f_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    # Contraction
                    # Try inside contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_used += 1
                    update_best(x_con, f_con)
                    
                    if f_con < f_vals[-1]:
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = func(simplex[i])
                            calls_used += 1
                            update_best(simplex[i], f_vals[i])
                            if budget_up() or time_up():
                                return

        # Main Loop: Multi-start with Nelder-Mead local refinement
        rng = np.random.default_rng(42)
        
        # Strategy: 
        # 1. Start with a few diverse points.
        # 2. Run Nelder-Mead from each.
        # 3. If budget remains, restart from best found with smaller perturbation or new random point.
        
        # Initial diverse starts
        starts = [np.zeros(dim)]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, dim))
            
        # Shuffle
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]
        
        for x0 in starts:
            if budget_up() or time_up():
                break
            
            # Run Nelder-Mead
            # Limit iterations to ensure we don't burn all budget on one start
            # ~100-200 calls per start is reasonable for dim=5
            nelder_mead(x0, max_iter=50)
            
            # If we still have budget, do another round of local search from the best found so far
            if not budget_up() and not time_up():
                # Restart from global best with a fresh simplex (maybe perturbed)
                x_restart = global_best_x + rng.normal(0, 0.1, dim)
                x_restart = np.clip(x_restart, -5, 5)
                nelder_mead(x_restart, max_iter=50)
                
                # Another restart from a new random point if budget allows
                if not budget_up() and not time_up():
                    x_new = rng.uniform(-4, 4, dim)
                    nelder_mead(x_new, max_iter=50)
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
