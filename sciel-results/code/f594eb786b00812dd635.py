"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    1+1 Evolution Strategy with Adaptive Step Size.
    
    Strategy:
    1. Initialize with a random point.
    2. Iteratively propose a new point by adding Gaussian noise scaled by the current step size.
    3. If the new point is better, move to it and slightly increase the step size (or keep it).
    4. If the new point is worse, keep the current point and slightly decrease the step size.
    5. This adaptive mechanism allows the search to zoom into local minima (shrinking steps)
       while still having the potential to escape shallow local minima if the step size hasn't
       shrunk too much, although pure 1+1 ES is prone to getting stuck in local minima.
       
    To handle the multimodal nature of f22 (21 peaks), we rely on the fact that the 
    "hidden" rotation/shift is fixed per instance, and a well-tuned ES can navigate 
    the smooth valleys between peaks if started in a reasonable location. 
    However, since we don't know the location, a simple random start might miss the global 
    minimum. 
    
    Improvement: Instead of a single trajectory, we use a "multi-start" 1+1 ES where we 
    split the budget among a few independent ES runs starting from different random points, 
    and keep the best result. This is structurally different from the current program which 
    does a global scan then one local refinement. Here, the exploration is stochastic and 
    interleaved with exploitation in each run.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_1plus1_es(self, func, x0, budget, deadline):
        """
        Runs a 1+1 Evolution Strategy starting from x0 with a specific budget.
        Returns (best_x, best_fx, calls_made).
        """
        dim = self.dim
        x = x0.copy()
        fx = func(x)
        calls = 1
        
        # Initial step size: roughly 1/10th of the domain width
        # Domain is [-5, 5], width 10. 
        # Starting with a moderate step size to allow movement.
        sigma = 0.5 
        
        # 1/5th rule parameters
        # If success rate is > 0.2, increase sigma; if < 0.2, decrease.
        # We use a simple windowed approach or just immediate adaptation for speed.
        # For low budget, immediate adaptation is often better to avoid lag.
        
        best_x = x.copy()
        best_fx = fx
        
        # To avoid getting stuck in a tiny local minimum immediately, 
        # we can use a slightly larger initial sigma.
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate mutation
            # Standard deviation of the noise
            noise = np.random.normal(0, sigma, size=dim)
            x_trial = x + noise
            
            # Clip to bounds
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            calls += 1
            
            if f_trial < fx:
                # Success
                x = x_trial
                fx = f_trial
                # Increase step size slightly to continue exploring
                # Factor 1.05 is a gentle increase
                sigma *= 1.05
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Failure
                # Decrease step size to refine local search
                # Factor 0.95 is a gentle decrease
                sigma *= 0.95
                
            # Prevent sigma from becoming too small or too large
            if sigma < 1e-5:
                sigma = 1e-5
            if sigma > 5.0:
                sigma = 5.0

        return best_x, best_fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin
        dim = self.dim
        budget = self.budget
        
        # We will split the budget into a few independent runs.
        # Since f22 has 21 peaks, a single 1+1 ES starting from one point is likely to
        # get stuck in a local minimum. 
        # Let's try 3 independent runs with different starting points.
        # Budget per run: budget // 3
        
        # To ensure we make some calls even if one run fails or is slow, 
        # we allocate time and budget carefully.
        
        n_runs = 3
        budget_per_run = budget // n_runs
        
        # Use a fixed seed for reproducibility of the starting points?
        # The task says "hidden seeded shift", but the Optimizer can use its own RNG.
        # Using a fixed seed for the Optimizer's internal RNG ensures deterministic behavior 
        # for the same instance, which is good for debugging but might be suboptimal if 
        # the "best" start points are instance-dependent. 
        # However, random starts are generally robust.
        # Let's use a random seed but ensure we don't run out of time.
        
        rng = np.random.default_rng(42) # Fixed seed for internal logic stability
        
        global_best_x = None
        global_best_fx = np.inf
        total_calls = 0
        
        for i in range(n_runs):
            if time.time() > deadline or total_calls >= budget:
                break
                
            # Generate a starting point
            # Uniformly random in [-5, 5]
            x_start = rng.uniform(-5, 5, size=dim)
            
            # Run the ES
            # We pass the remaining budget for this specific run
            # But we must ensure we don't exceed the total budget.
            # The _run_1plus1_es function takes a budget parameter which is the max calls 
            # it can make *in addition* to the initial one? 
            # Let's refine: _run_1plus1_es makes 1 call for x0. 
            # So if budget_per_run is 166, it will make 166 calls total.
            
            x_opt, f_opt, calls_made = self._run_1plus1_es(
                func, x_start, budget_per_run, deadline
            )
            
            total_calls += calls_made
            
            if f_opt < global_best_fx:
                global_best_fx = f_opt
                global_best_x = x_opt.copy()
                
            # If we have budget left, we could run another, but the loop handles that.
            
        # If we didn't make any calls (unlikely), make one
        if total_calls == 0:
            x0 = np.zeros(dim)
            fx0 = func(x0)
            global_best_x = x0
            global_best_fx = fx0
            total_calls = 1

        return global_best_x, global_best_fx

# EVOLVE-BLOCK-END
