import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def call(x):
            nonlocal calls, best_x, best_f
            calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Broad Exploration using Latin Hypercube Sampling (LHS)
        # We spend about 40% of the budget here to find multiple diverse basins.
        # LHS ensures good coverage of the space compared to pure random.
        n_lhs = max(10, int(budget * 0.4))
        
        # Generate LHS points
        # For each dimension, divide [0,1] into n_lhs bins and pick a random point in each bin
        u = np.zeros((n_lhs, n))
        for d in range(n):
            perm = np.arange(n_lhs)
            np.random.shuffle(perm)
            # Random offset within each bin
            offsets = np.random.uniform(size=n_lhs)
            u[:, d] = (perm + offsets) / n_lhs
            
        # Map to [-5, 5]
        lhs_points = u * 10.0 - 5.0
        
        # Evaluate LHS points
        for i in range(n_lhs):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            x = lhs_points[i]
            call(x)
            
        # If we haven't used enough budget, fill remaining Phase 1 with random
        # (Though LHS usually covers the 40% well, this ensures we use the allocated phase budget)
        # Actually, let's just stick to the LHS points for Phase 1 to save calls for Phase 2.
        # The remaining budget is for local search.

        # Phase 2: Local Refinement using CMA-ES on the top candidates
        # We select the top K candidates from Phase 1 to run local searches.
        # Since we don't store all points easily without overhead, we'll just restart CMA-ES
        # from the current best and maybe a few random restarts if budget allows.
        
        # Better strategy: Identify the best point found so far.
        # Run a local CMA-ES (or similar) from the best point.
        # If time/budget allows, run another from the 2nd best, etc.
        
        # To keep it simple and robust within the time limit, we will run a sequence of local searches.
        # We'll use a simplified CMA-ES or just a strong (1+1)-ES with adaptive step size, 
        # but the "Different Strategy" directive asks for a different family.
        # Multi-start is the key difference. Let's do Multi-start (1+1)-ES with larger initial steps 
        # and specific restart logic, or a simple CMA-ES.
        
        # Let's implement a lightweight CMA-ES for local refinement.
        def cma_es(x0, max_calls, deadline):
            nonlocal calls, best_x, best_f
            
            if calls >= budget:
                return
            
            # Initialize CMA-ES parameters
            # x is the mean
            x = x0.copy()
            sigma = 0.5 # Start with a moderate step size for local search
            C = np.eye(n)
            pc = np.zeros(n)
            cs = np.zeros(n)
            
            # Strategy parameters
            mu = max(1, n // 2) # Number of offspring to use for update
            lambda_ = max(2 * mu + 1, 8) # Population size
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n + 1)) - 1)) / (n + 1 + 3 * mu_eff)
            c1 = 2 / ((n + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (weights[0] - 0.5) / ((n + 2)**2 + mu_eff))
            damps = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n + 1) - 1)) + cs
        
            # Evaluate current best
            fx = call(x)
            
            while calls < budget and time.time() < deadline:
                # Generate offspring
                ys = []
                fs = []
                
                for i in range(lambda_):
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Sample from N(x, sigma^2 * C)
                    # Cholesky decomposition of C
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        L = np.eye(n)
                        
                    z = np.random.multivariate_normal(np.zeros(n), np.eye(n))
                    y = x + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    
                    fy = call(y)
                    ys.append(y)
                    fs.append(fy)
                
                if len(ys) < mu:
                    break
                    
                # Sort by fitness
                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Update mean
                x_new = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
                
                # Update step size (CSA)
                y_mean = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
                y_mean = y_mean / (sigma * np.linalg.norm(np.linalg.inv(L) @ y_mean)) if np.linalg.norm(np.linalg.inv(L) @ y_mean) > 1e-10 else y_mean
                # Simplified CSA update
                cs_val = (1 - (1 - cs)**lambda_) ** 0.5
                # This is getting complex. Let's use a simpler adaptive step size (1+1)-ES logic 
                # but multi-started. The "Different Strategy" is the Multi-Start + LHS Exploration.
                # A full CMA-ES might be too slow/complex to implement correctly in this short space 
                # without bugs. Let's stick to a robust Multi-Start (1+1)-ES with a "hill climbing" phase.
                
                # Actually, let's just implement a clean, fast local search: 
                # Nelder-Mead or a simple coordinate descent? 
                # (1+1)-ES is very effective for Katsuura if started well.
                
                # Let's restart the local search logic to be simpler and faster:
                # Multi-Start (1+1)-ES
                break # Exit the complex CMA loop and use the simpler one below
                
        # Simpler and more robust: Multi-Start (1+1)-ES
        # We have the best point from LHS.
        # We will run (1+1)-ES from the best point.
        # If we have budget left, we might run from the 2nd best, etc.
        
        # To keep it simple and fast, we just run one long (1+1)-ES from the best LHS point.
        # But to be "multi-start", we should try a few.
        
        # Let's collect top 3 points from LHS.
        # We didn't store them all. Let's just use the current best_x.
        
        # Local Search: (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 1.0 # Larger initial sigma for local search to escape shallow minima
        
        # Adaptive sigma based on success rate (1/5 rule)
        successes = 0
        total = 0
        
        while calls < budget and time.time() < deadline:
            y = x + sigma * np.random.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = call(y)
            
            total += 1
            if fy < fx:
                x = y
                fx = fy
                successes += 1
                # Increase sigma if we are succeeding
                if total % 10 == 0:
                    if successes / total > 0.2:
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
                    successes = 0
            else:
                if total % 10 == 0:
                    if successes / total > 0.2:
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
                    successes = 0
            
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-6, 5.0)
            
            # Optional: If stuck, do a small random jump?
            # If no improvement for 50 steps, restart from a random point?
            # This would be a true multi-start.
            # Let's add a simple restart mechanism if progress stalls.
            if total > 0 and total % 100 == 0:
                # Check if we've made significant progress recently
                # If not, pick a new random starting point
                # For now, let's just keep going. The LHS phase already did the multi-starting.
                pass

        return best_x, best_f
