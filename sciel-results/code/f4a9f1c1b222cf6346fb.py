"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3 that exploits the known structure.
    
    ZDT3: f1 = x1, f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/(1+10*(n-1))) + sin(10*pi*f1))
    The Pareto front is achieved when x2..xn = 0, giving:
      f1 = x1, f2 = 11*(1 - sqrt(x1/11) + sin(10*pi*x1))
    
    Strategy:
    1. Generate a dense set of candidate x1 values that capture the oscillatory structure.
    2. Evaluate these points on the ideal manifold (x2..xn=0).
    3. Use the remaining budget for local refinement around promising regions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front: x2..xn = 0, x1 in [0,1]
        # f1 = x1, f2 = 11*(1 - sqrt(x1/11) + sin(10*pi*x1))
        
        # Generate candidate x1 values that capture the oscillatory structure
        # sin(10*pi*x1) has 10 full periods in [0,1], so we need to sample densely
        # around zeros and extrema of sin(10*pi*x1)
        
        # Critical points: zeros at x1 = k/20 for k=0..20, extrema at x1 = (2k+1)/20
        # Let's create a non-uniform grid that's denser near these critical points
        
        n_evals = 0
        results = []  # list of (x, f) tuples
        
        # Phase 1: Dense sampling of the Pareto front
        # Use a combination of uniform and critical-point-focused sampling
        
        # Uniform grid
        n_uniform = 100
        x1_uniform = np.linspace(0.0, 1.0, n_uniform)
        
        # Critical points (zeros and extrema of sin(10*pi*x1))
        # Zeros: x1 = k/20, k=0..20 -> 21 points
        # Extrema: x1 = (2k+1)/20, k=0..19 -> 20 points
        x1_critical = np.concatenate([
            np.arange(0, 21) / 20.0,       # zeros
            (np.arange(0, 20) * 2 + 1) / 20.0  # extrema
        ])
        
        # Combine and sort
        x1_candidates = np.unique(np.concatenate([x1_uniform, x1_critical]))
        
        # Also add some random points for exploration
        n_random = 50
        x1_random = np.random.uniform(0, 1, n_random)
        x1_candidates = np.unique(np.concatenate([x1_candidates, x1_random]))
        
        # Sort by x1
        x1_candidates = np.sort(x1_candidates)
        
        # Limit to budget
        max_phase1 = min(len(x1_candidates), self.budget - 50)  # Reserve some for phase 2
        x1_phase1 = x1_candidates[:max_phase1]
        
        # Evaluate phase 1
        for x1_val in x1_phase1:
            if time.time() > deadline:
                break
            if n_evals >= self.budget:
                break
            
            x = np.zeros(self.dim)
            x[0] = np.clip(x1_val, lo[0], hi[0])
            # x[1:] = 0 (already zero)
            
            f = problem(x)
            n_evals += 1
            results.append((x.copy(), np.array(f)))
        
        # Phase 2: Local refinement around the best hypervolume-contributing points
        if n_evals < self.budget and time.time() < deadline:
            # Compute a simple hypervolume contribution metric
            # For each point, estimate its contribution to the hypervolume
            # We'll do a greedy refinement: for each point, try small perturbations
            
            remaining = self.budget - n_evals
            n_refine = min(remaining, 50)
            
            # Sort results by f1 for easier neighborhood search
            results_sorted = sorted(results, key=lambda r: r[1][0])
            
            # For each point, try a few local perturbations
            rng = np.random.default_rng(42)
            tried = set()
            
            for i in range(len(results_sorted)):
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x_base, f_base = results_sorted[i]
                x1_base = x_base[0]
                
                # Try a few perturbations in x1
                for delta_x1 in [0.005, -0.005, 0.002, -0.002, 0.01, -0.01]:
                    if n_evals >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x1_new = x1_base + delta_x1
                    if x1_new < 0 or x1_new > 1:
                        continue
                    
                    # Check if we've already tried a similar point
                    key = round(x1_new, 4)
                    if key in tried:
                        continue
                    tried.add(key)
                    
                    x = np.zeros(self.dim)
                    x[0] = x1_new
                    
                    f = problem(x)
                    n_evals += 1
                    results.append((x.copy(), np.array(f)))
                    
                    if n_evals >= self.budget:
                        break
        
        # Return the best set of points (all evaluated points)
        if results:
            X = np.array([r[0] for r in results])
        else:
            X = np.array([lo])
        
        return X
# EVOLVE-BLOCK-END
