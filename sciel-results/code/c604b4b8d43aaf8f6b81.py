import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Time limit safety margin: Evaluator kills at 330s. 
        # We stop at 320s to be safe.
        time_limit = 320.0
        
        # State
        self.best_x = np.zeros(n)
        self.best_f = np.inf
        self.calls_made = 0
        
        def call_func(x):
            if self.calls_made >= budget:
                return np.inf
            if time.time() - self.start_time > time_limit:
                return np.inf
            fx = func(x)
            self.calls_made += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # --- Phase 1: Global Exploration ---
        # Evaluate 50 random points to identify promising basins
        n_initial = 50
        candidates = []
        
        # Include center and corners? No, just random uniform in [-4, 4]
        # The problem states minimizer is in [-4, 4]
        for _ in range(n_initial):
            if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                break
            x = rng.uniform(-4.0, 4.0, size=n)
            fx = call_func(x)
            candidates.append((fx, x))
            
        # Sort by fitness
        candidates.sort(key=lambda item: item[0])
        
        # Select top 5
        top_k = min(5, len(candidates))
        starts = [x for f, x in candidates[:top_k]]
        
        # If we didn't get 5 (e.g., budget ran out), pad with randoms or just use what we have
        while len(starts) < 5 and self.calls_made < budget:
             x = rng.uniform(-4.0, 4.0, size=n)
             fx = call_func(x)
             if fx < self.best_f:
                 self.best_f = fx
                 self.best_x = x.copy()
             starts.append(x)
        
        if not starts:
            return self.best_x, self.best_f

        # --- Phase 2: Interleaved (1+1)-ES from Top 5 ---
        # Allocate remaining budget evenly? Or just interleave 1 step each?
        # Interleaving 1 step each is robust.
        # (1+1)-ES parameters:
        # sigma starts at 1.0? Or 0.5?
        # For f13 (sharp ridge), we need to be careful not to jump over the ridge.
        # But also need to find it.
        # Standard (1+1)-ES: 
        #   z ~ N(0, I)
        #   y = x + sigma * z
        #   if f(y) < f(x): x = y, sigma *= 1.5 (or similar)
        #   else: sigma *= 0.87 (or similar)
        
        # Let's use a slightly more aggressive success rate control or fixed steps.
        # Given the "sharp ridge", a smaller sigma might be better once close, 
        # but larger to find it.
        
        # Let's maintain state for each start point.
        # state[i] = [x, sigma, f_x]
        
        states = []
        for x0 in starts:
            # We already have f(x0) from Phase 1? 
            # We stored (fx, x). Let's re-evaluate or use stored.
            # To save calls, we can use the stored f. But we need to be careful 
            # if the global best changed. 
            # Actually, call_func updates global best. 
            # Let's just re-evaluate the start point to get a fresh f for the local search state?
            # No, that wastes 5 calls. 
            # We can just use the f from the candidates list.
            # But the candidates list is sorted. 
            # Let's just re-evaluate to be safe and simple, as it's only 5 calls.
            # Or better, just use the current global best logic.
            # Actually, the (1+1)-ES needs a local f. 
            # Let's just evaluate x0 again.
            fx0 = call_func(x0)
            states.append({
                'x': x0.copy(),
                'f': fx0,
                'sigma': 1.0 # Initial step size
            })
            
        # If we ran out of budget during setup, return
        if self.calls_made >= budget:
            return self.best_x, self.best_f

        # Main Loop
        # Interleave steps from each state until budget is exhausted.
        # We can also adapt sigma based on success rate of the last few steps?
        # Simple fixed adaptation: sigma *= 1.5 on success, 0.8 on failure.
        
        while self.calls_made < budget and time.time() - self.start_time < time_limit:
            for i in range(len(states)):
                if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                    break
                
                s = states[i]
                x = s['x']
                sigma = s['sigma']
                
                # Generate perturbation
                z = rng.standard_normal(n)
                y = x + sigma * z
                # Clip to box [-5, 5]
                y = np.clip(y, -5.0, 5.0)
                
                fy = call_func(y)
                
                if fy < s['f']:
                    # Success
                    s['x'] = y
                    s['f'] = fy
                    s['sigma'] *= 1.5
                    # If this is the new global best, update
                    if fy < self.best_f:
                        self.best_f = fy
                        self.best_x = y.copy()
                else:
                    # Failure
                    s['sigma'] *= 0.8
                    # If sigma gets too small, maybe restart this lineage?
                    # Or just let it die.
                    if s['sigma'] < 1e-6:
                        # Reset this lineage to a random point?
                        # Or just skip it in future iterations?
                        # Let's reset it to a new random point to keep diversity.
                        new_x = rng.uniform(-4.0, 4.0, size=n)
                        new_f = call_func(new_x)
                        s['x'] = new_x
                        s['f'] = new_f
                        s['sigma'] = 1.0
                        if new_f < self.best_f:
                            self.best_f = new_f
                            self.best_x = new_x.copy()

        return self.best_x, self.best_f
