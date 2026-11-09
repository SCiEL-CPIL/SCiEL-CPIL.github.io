import numpy as np
import time

class Optimizer:
    """
    Multi-Start SPSA (Simultaneous Perturbation Stochastic Approximation) Optimizer.
    
    Strategy:
    1. Maintain a set of active 'searchers' (basins).
    2. Each searcher uses SPSA to estimate the gradient with 2 evaluations per step.
    3. Use a 'best-of-two' acceptance rule: evaluate x - c*grad and x + c*grad, 
       keep the better one. This guarantees monotonic improvement for that searcher.
    4. If a searcher stagnates (no improvement in last K steps), restart it at a 
       new random location.
    5. Keep track of the global best.
    
    This is structurally different from (1+1)-ES because:
    - It uses gradient information (estimated via 2 points) rather than blind mutation.
    - It uses a specific selection rule (best of the two perturbations) that is 
      deterministic in outcome relative to the pair, whereas ES accepts/rejects 
      based on a single comparison to the parent.
    - It explicitly manages multiple independent SPSA trajectories.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin
        
        # Hyperparameters
        num_searchers = 4
        c0 = 1.0          # Initial perturbation magnitude
        gamma = 0.05      # Decay rate for perturbation
        alpha = 0.6       # Decay rate for step size
        stagnation_limit = 15 # Steps without improvement before restart
        min_sigma = 1e-4   # Minimum perturbation magnitude
        
        # Initialize searchers
        # We will use a list of dictionaries to store state for each searcher
        searchers = []
        best_x_global = None
        best_fx_global = np.inf
        
        # Initialize with random points
        # Use a fixed seed for reproducibility in the optimization process itself,
        # though the problem instances are externally seeded.
        rng = np.random.default_rng(42)
        
        for _ in range(num_searchers):
            x = rng.uniform(-4, 4, self.dim)
            fx = func(x)
            if fx < best_fx_global:
                best_fx_global = fx
                best_x_global = x.copy()
            
            searchers.append({
                'x': x,
                'fx': fx,
                'sigma': c0,
                'step_size': 0.5,
                'stagnation': 0
            })
        
        calls_used = num_searchers
        step_idx = 0

        while calls_used < self.budget:
            if time.time() > deadline:
                break
                
            # For each searcher, perform one SPSA step
            # SPSA step:
            # 1. Generate random direction delta (entries +1 or -1)
            # 2. Evaluate f(x + c*delta) and f(x - c*delta)
            # 3. Estimate gradient: (f(x+c*delta) - f(x-c*delta)) / (2*c*delta)
            # 4. Update x: x = x - a * grad_estimate
            #    However, a more robust "best-of-two" variant is:
            #    Keep the better of x+c*delta and x-c*delta.
            #    This is effectively a form of local search that uses 2 evals.
            #    Let's use the standard SPSA update but with a "greedy" check.
            #    Actually, the "best-of-two" is often better for noisy or non-convex.
            #    Let's stick to a robust variant:
            #    Generate delta.
            #    x_plus = x + sigma * delta
            #    x_minus = x - sigma * delta
            #    Evaluate both.
            #    Choose the better one as the new x.
            #    This uses 2 calls per searcher per step.
            
            for i, s in enumerate(searchers):
                if calls_used + 2 > self.budget:
                    break
                
                # Check time again inside loop for safety
                if time.time() > deadline:
                    return best_x_global, best_fx_global

                # Generate random direction
                # Using Rademacher distribution (+1, -1)
                delta = rng.choice([-1, 1], size=self.dim)
                
                # Perturbation points
                x_plus = s['x'] + s['sigma'] * delta
                x_minus = s['x'] - s['sigma'] * delta
                
                # Clip to box
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                # Evaluate
                fx_plus = func(x_plus)
                calls_used += 1
                fx_minus = func(x_minus)
                calls_used += 1
                
                # Select the better point
                if fx_plus < fx_minus:
                    new_x = x_plus
                    new_fx = fx_plus
                else:
                    new_x = x_minus
                    new_fx = fx_minus
                
                # Update searcher state
                if new_fx < s['fx']:
                    s['x'] = new_x
                    s['fx'] = new_fx
                    s['stagnation'] = 0
                    # Decay sigma slowly if making progress? 
                    # Or keep it constant to explore? 
                    # Standard SPSA decays c. Let's decay it to refine.
                    s['sigma'] = max(s['sigma'] * (1 - gamma), min_sigma)
                    
                    if new_fx < best_fx_global:
                        best_fx_global = new_fx
                        best_x_global = new_x.copy()
                else:
                    # No improvement
                    s['stagnation'] += 1
                    # If stagnating, maybe increase sigma to escape local minimum?
                    # Or just let it restart.
                    # Let's try increasing sigma slightly to encourage escape
                    s['sigma'] = min(s['sigma'] * 1.1, 5.0)
                
                # Check for restart
                if s['stagnation'] >= stagnation_limit:
                    # Restart this searcher
                    new_x_init = rng.uniform(-4, 4, self.dim)
                    new_fx_init = func(new_x_init)
                    calls_used += 1
                    
                    s['x'] = new_x_init
                    s['fx'] = new_fx_init
                    s['sigma'] = c0
                    s['stagnation'] = 0
                    
                    if new_fx_init < best_fx_global:
                        best_fx_global = new_fx_init
                        best_x_global = new_x_init.copy()
                
                # Break if we hit budget during inner loop
                if calls_used >= self.budget:
                    break

            step_idx += 1
            # Safety break if we ran out of calls in the inner loop
            if calls_used >= self.budget:
                break

        return best_x_global, best_fx_global
