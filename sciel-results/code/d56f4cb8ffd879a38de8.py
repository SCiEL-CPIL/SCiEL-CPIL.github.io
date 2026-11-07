import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 270.0
        
        best_x = np.zeros(n)
        best_f = float('inf')
        evals_left = budget

        def time_exceeded():
            return time.time() > deadline

        # Phase 1: Strategic Probing
        # We use a small number of evaluations to find a few good starting points.
        # For Rastrigin in [-5, 5]^5, the global minimum is at (0,0,0,0,0) shifted/rotated.
        # However, due to hidden shift/rotation, we don't know where 0 is in the box.
        # A coarse grid is risky if it's too fine. A set of random points + center is a good balance.
        
        # Let's reserve 20% of the budget for probing, or at least 50-100 evals if budget allows.
        # With budget=500, let's use 50 evals for probing.
        num_probes = min(50, max(10, budget // 10))
        
        probe_points = []
        # Always include center
        probe_points.append(np.zeros(n))
        
        # Add random points
        # Use a fixed seed for reproducibility? No, the task says hidden seeded shift.
        # But our optimizer should be robust. Randomness in optimizer is fine.
        # Actually, using np.random is fine.
        
        for _ in range(num_probes - 1):
            # Sample from uniform distribution in [-5, 5]^n
            p = np.random.uniform(-5.0, 5.0, size=n)
            probe_points.append(p)
            
        # Evaluate probes
        for p in probe_points:
            if evals_left <= 0 or time_exceeded():
                break
            f = func(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()
            evals_left -= 1

        # Phase 2: Multi-Start Nelder-Mead
        # Select the top K probe points as starting points for local search.
        # Let's keep track of the best few probe points.
        # We didn't store all probe results in a list for sorting, let's just re-use best_x
        # and maybe a few others if we stored them. 
        # To be efficient, let's just restart from the global best found in Phase 1, 
        # and potentially a second best if we tracked it.
        
        # Actually, a better strategy for multi-start:
        # We can't easily "re-evaluate" probes. 
        # Let's just run Nelder-Mead from the best point found so far.
        # To add "multi-start" flavor without extra evals, we can run NM with 
        # a slightly perturbed initial simplex if the first run stalls.
        
        # Let's implement a robust Nelder-Mead.
        # Parameters:
        # alpha = 1.0 (reflection)
        # gamma = 2.0 (expansion)
        # rho = 0.5 (contraction)
        # sigma = 0.5 (shrink)
        
        def nelder_mead(func, x0, budget, n, deadline, best_f, best_x):
            # Initialize simplex
            # x0 is the best point. 
            # Create simplex by perturbing x0.
            # Step size for simplex: 0.05 or 0.1?
            # Since we are in a local basin, 0.05 is good.
            
            step = 0.05
            simplex = []
            f_simplex = []
            
            # First vertex is x0
            simplex.append(x0.copy())
            f0 = func(x0)
            f_simplex.append(f0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            budget -= 1
            
            # Other vertices
            for i in range(n):
                xi = x0.copy()
                # Perturb dimension i
                if xi[i] + step <= 5.0:
                    xi[i] += step
                else:
                    xi[i] -= step
                simplex.append(xi)
                fi = func(xi)
                f_simplex.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                budget -= 1
                if budget <= 0 or time.time() > deadline:
                    return best_x, best_f, budget

            # Main loop
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while budget > 0 and time.time() < deadline:
                # Sort simplex by function value
                idx = np.argsort(f_simplex)
                simplex = [simplex[i] for i in idx]
                f_simplex = [f_simplex[i] for i in idx]
                
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                
                # Centroid of all but worst
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = x_centroid + alpha * (x_centroid - x_worst)
                f_r = func(x_r)
                budget -= 1
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = x_centroid + gamma * (x_centroid - x_worst)
                    f_e = func(x_e)
                    budget -= 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                        if f_e < best_f:
                            best_f = f_e
                            best_x = x_e.copy()
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                        if f_r < best_f:
                            best_f = f_r
                            best_x = x_r.copy()
                elif f_r < f_simplex[-2]:
                    # Simple Reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                    if f_r < best_f:
                        best_f = f_r
                        best_x = x_r.copy()
                else:
                    # Contraction
                    # Outer contraction
                    x_c = x_centroid + rho * (x_worst - x_centroid)
                    f_c = func(x_c)
                    budget -= 1
                    
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            budget -= 1
                            if f_simplex[i] < best_f:
                                best_f = f_simplex[i]
                                best_x = simplex[i].copy()
                            if budget <= 0 or time.time() > deadline:
                                return best_x, best_f, budget
                            
                if budget <= 0 or time.time() > deadline:
                    return best_x, best_f, budget

            return best_x, best_f, budget

        # Run Nelder-Mead from the best probe point
        best_x, best_f, evals_left = nelder_mead(
            func, best_x, evals_left, n, deadline, best_f, best_x
        )

        # If we still have budget and time, try a second restart from a slightly different point
        # to escape potential local minima.
        # We can pick the 2nd best probe point if we had stored it. 
        # Since we didn't store all probes, let's just do a random restart if we have >50% budget left.
        if evals_left > 50 and time.time() < deadline:
            # Generate a new random point near the current best? 
            # Or just a new random point in the box?
            # Rastrigin has many local minima. A random restart is safer.
            x_restart = np.random.uniform(-5.0, 5.0, size=n)
            f_restart = func(x_restart)
            evals_left -= 1
            if f_restart < best_f:
                best_f = f_restart
                best_x = x_restart.copy()
            
            # Run NM again
            best_x, best_f, evals_left = nelder_mead(
                func, best_x, evals_left, n, deadline, best_f, best_x
            )

        return best_x, best_f

