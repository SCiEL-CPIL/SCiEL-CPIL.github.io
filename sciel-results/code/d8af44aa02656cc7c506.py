# EVOLVE-BLOCK-START
import numpy as np
import time
import cma


class Optimizer:
    """
    Optimizer for BBOB f7 (Step ellipsoidal) using CMA-ES with IPOP and restarts.
    CMA-ES is robust to non-smoothness and adapts to the ellipsoidal structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, aim to finish well before.
        # CMA-ES can be slow in high dims, but dim=5 is small.
        # However, many restarts can add up.
        deadline = start_time + 280 
        
        dim = self.dim
        budget = self.budget
        
        # Wrap func to count calls and track best
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def wrapped(x):
            self._call_count += 1
            # Ensure x is a numpy array for safety, though cma passes arrays
            x_arr = np.asarray(x, dtype=float)
            fx = float(func(x_arr))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x_arr.copy()
            return fx

        # CMA-ES Parameters
        # For dim=5, population size is typically 4 + 3*log(5) ~ 8
        # We will use a smaller population to conserve budget for more restarts
        # or rely on IPOP to increase it if stuck.
        
        # Strategy: 
        # 1. Start with a reasonable initial sigma.
        # 2. Use IPOP-CMA (increases population size when stagnation is detected).
        # 3. Use derandomized restarts (cma restarts automatically if stagnation).
        # 4. We manually manage a few restarts if the first one doesn't converge 
        #    quickly, using the best point found so far as a new center or 
        #    a random point if the first failed.

        # Initial center: 0 is a safe bet, but the shift is in [-4,4].
        # Random start is also viable. Let's try 0 first, then random if it fails fast.
        
        # We will run a sequence of CMA-ES instances.
        # Each instance gets a sub-budget.
        # If it converges (tolerance reached) or uses its sub-budget, we move to the next.
        
        # To maximize AOCC, we want fast initial progress.
        # CMA-ES is good at this if the initial sigma is right.
        # Sigma = 2.0 covers a good portion of the box.
        
        # We will use the 'cma' library's restart mechanism.
        # cma.fmin2 or cma.CMAEvolutionStrategy with restarts.
        # However, we need to strictly control the number of function calls.
        # cma.CMAEvolutionStrategy allows us to inspect the budget.
        
        # Let's use a manual loop with cma.CMAEvolutionStrategy to have full control.
        
        # Define bounds for clipping, though CMA-ES doesn't natively handle box constraints well 
        # without penalty or reflection. 
        # The problem states func clips inputs to [-5,5]. 
        # CMA-ES will propose points outside, but func will clip them. 
        # This can be inefficient if many points are clipped. 
        # It's better to keep the search distribution inside the box.
        # We can use 'cma' with bounds? No, standard cma doesn't.
        # We can use a penalty function, but that wastes calls.
        # Alternatively, we can use a transformation or just rely on the fact that 
        # the optimum is in [-4,4] and start in 0 with sigma 2.
        # Points outside [-5,5] will be clipped, effectively acting as a boundary.
        # This is usually okay for small dims.
        
        # Let's try a few restarts.
        # Restart 1: Center 0, Sigma 2.0
        # Restart 2: Best found so far, Sigma 1.0 (if not converged)
        # Restart 3: Random point, Sigma 2.0
        
        # We'll define a helper to run one CMA-ES instance with a specific budget cap.
        
        def run_cma_es(x0, sigma0, max_calls, maxtime):
            """
            Runs one CMA-ES instance.
            Returns the best x and f found in this instance.
            """
            # cma requires a vector for x0
            x0 = np.asarray(x0, dtype=float)
            
            # Population size
            # Default is 4 + 3*log(n). For n=5, ~8.
            # We can reduce this to 4 or 5 to save calls if we want many restarts,
            # but IPOP might increase it.
            # Let's stick to default or slightly smaller.
            popsize = 4 + int(3 * np.log(dim)) # ~8
            
            # If we have very little budget left, reduce popsize
            if max_calls < 50:
                popsize = 4
            
            # CMA options
            options = {
                'popsize': popsize,
                'maxfevals': max_calls, # Hard cap on function evaluations for this instance
                'verb_disp': 0,
                'verb_log': 0,
                'tolfun': 1e-12, # Stop if function value change is tiny
                'tolx': 1e-8,    # Stop if x change is tiny
                'toltime': maxtime, # Stop if time limit reached
                'bounds': [(-5.0, 5.0)] * dim # cma supports bounds in recent versions? 
                # Note: cma 2.x+ supports 'bounds'. If not, we rely on clipping.
                # Let's assume cma supports bounds or we remove it if it errors.
                # Actually, standard cma does NOT support bounds directly in the main loop 
                # without specific flags or versions. 
                # To be safe and compatible, I will REMOVE bounds from options 
                # and rely on the fact that the function clips. 
                # However, this can be inefficient. 
                # A better approach for cma with bounds is to use a penalty or 
                # transform. But for simplicity and robustness against cma version issues,
                # we will omit 'bounds' and trust the clipping in 'func'.
                # Wait, cma.CMAEvolutionStrategy accepts 'bounds' in newer versions?
                # Let's check. cma 2.8+ supports 'bounds'. 
                # If the environment has an older cma, this might fail.
                # To be safe, I will NOT use 'bounds' in options.
                # Instead, I will ensure the initial sigma and center keep the mass 
                # mostly inside, and accept some clipped evaluations.
            }
            
            # Remove bounds to ensure compatibility
            options.pop('bounds', None)
            
            try:
                es = cma.CMAEvolutionStrategy(x0, sigma0, options)
                
                # Manual loop to respect time and call limits strictly
                while not es.stop() and self._call_count < self.budget:
                    if time.time() > deadline - 5: # 5s buffer
                        break
                    
                    # Ask for next points
                    ys = es.ask()
                    
                    # Evaluate them
                    fs = []
                    for y in ys:
                        if self._call_count >= self.budget:
                            break
                        if time.time() > deadline - 5:
                            break
                        fy = wrapped(y)
                        fs.append(fy)
                        
                    # If we broke out early, we might have fewer fs than ys
                    # cma.tell expects len(fs) == len(ys) usually, or handles it?
                    # It's safer to only tell if we evaluated all, or pad?
                    # Actually, if we break, we should stop the loop.
                    if len(fs) < len(ys):
                        # We didn't evaluate all. We can't tell cma.
                        # We just break.
                        break
                    
                    es.tell(ys, fs)
                    
            except Exception:
                pass
            
            return es.result.xbest, es.result.fbest

        # --- Execution Plan ---
        
        # Phase 1: Initial CMA-ES run
        # Start at 0, sigma 2.0
        # Give it a generous chunk of the budget, but leave room for restarts.
        # Let's give it 300 calls initially.
        init_budget = 300
        if self.budget < 350:
             init_budget = self.budget - 50
             
        x0 = np.zeros(dim)
        sigma0 = 2.0
        
        # Run first instance
        # We pass max_calls = init_budget, but wrapped() enforces global budget.
        # run_cma_es will stop if global budget is hit.
        
        # To allow IPOP to work, we might need to let it run longer if it's making progress.
        # But for AOCC, early progress is key.
        
        # Let's run a few short CMA-ES runs.
        # Run 1: x0=0, sigma=2.0, budget=200
        # Run 2: x0=best_so_far, sigma=1.0, budget=100 (if not converged)
        # Run 3: x0=random, sigma=2.0, budget=remaining
        
        # Actually, cma's internal restart logic is powerful.
        # We can just instantiate cma.fmin2 with a large budget and let it restart internally?
        # fmin2 doesn't expose the call count easily to stop it mid-restart if time is up.
        # The manual loop above is safer.
        
        # Let's refine the manual loop strategy:
        # We will run CMA-ES instances sequentially.
        
        # Instance 1
        max_calls_1 = 250
        x_best_1, f_best_1 = run_cma_es(np.zeros(dim), 2.0, max_calls_1, deadline - time.time())
        
        # Instance 2: If we have budget left and haven't converged to a very low value
        # Use the best point from instance 1 as center, smaller sigma
        if self._call_count < self.budget - 20 and time.time() < deadline - 10:
            center2 = self._best_x if self._best_x is not None else np.zeros(dim)
            # Reduce sigma to refine
            sigma2 = 1.0 
            max_calls_2 = min(150, self.budget - self._call_count - 10)
            if max_calls_2 > 20:
                 run_cma_es(center2, sigma2, max_calls_2, deadline - time.time() - 5)
        
        # Instance 3: Global exploration if still budget left
        # Sometimes the first run gets stuck in a local optimum.
        # Try a random start.
        if self._call_count < self.budget - 20 and time.time() < deadline - 10:
            rng = np.random.default_rng(123)
            center3 = rng.uniform(-4, 4, dim)
            sigma3 = 2.0
            max_calls_3 = min(100, self.budget - self._call_count - 10)
            if max_calls_3 > 20:
                 run_cma_es(center3, sigma3, max_calls_3, deadline - time.time() - 5)
        
        # Final check: if we have very little budget left, do a few random samples?
        # CMA-ES is better.
        
        # Ensure we have a result
        if self._best_x is None:
            self._best_x = np.zeros(dim)
            self._best_f = func(self._best_x)
            
        return self._best_x, self._best_f


# EVOLVE-BLOCK-END
