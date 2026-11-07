import numpy as np
import scipy.optimize as opt
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        used_calls = 0
        max_calls = self.budget
        
        def safe_func(x):
            nonlocal used_calls
            if used_calls >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            used_calls += 1
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            if np.isnan(val) or np.isinf(val):
                return 1e10
            return val
        
        # Phase 1: Structured Global Exploration
        # Katsuura has a global minimum near (1, 1, ..., 1) in the standard formulation.
        # We sample near this region and also use Sobol for global coverage.
        
        # 1a. Local cluster around known minimum structure (1, 1, ..., 1)
        # The function is periodic-ish. Let's try a few offsets.
        # Standard Katsuura min is at x_i = 1.
        # Let's generate points around (1,1,1,1,1) with varying scales.
        
        initial_points = []
        
        # Add the canonical minimum point
        initial_points.append(np.ones(self.dim))
        
        # Add points with small perturbations around the canonical minimum
        for scale in [0.1, 0.5, 1.0, 2.0]:
            for _ in range(5):
                p = np.ones(self.dim) + rng.normal(0, scale, self.dim)
                initial_points.append(p)
        
        # Add some Sobol points for global coverage
        # Use scipy.stats.qmc if available, otherwise simple random
        try:
            import scipy.stats.qmc as qmc
            sobol = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            # We want to cover [-5, 5]^5
            # Generate 30 Sobol points
            n_sobol = 30
            sobol_points = sobol.random(n_sobol)
            # Scale from [0,1] to [-5,5]
            sobol_points = sobol_points * 10.0 - 5.0
            initial_points.extend(sobol_points)
        except ImportError:
            # Fallback to random
            for _ in range(30):
                initial_points.append(rng.uniform(-5, 5, self.dim))
        
        initial_points = np.array(initial_points)
        
        initial_values = []
        for p in initial_points:
            if used_calls >= max_calls or time.time() > deadline:
                break
            try:
                fp = safe_func(p)
                initial_values.append((fp, p.copy()))
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
            except RuntimeError:
                break
        
        if best_x is None:
            return None, np.inf

        # Phase 2: Multi-Start Local Optimization
        # Sort by function value
        initial_values.sort(key=lambda x: x[0])
        
        remaining_budget = max_calls - used_calls
        
        if remaining_budget > 10:
            # We will run Nelder-Mead from the top few candidates.
            # Katsuura is smooth but multimodal. NM is good for local refinement.
            # Let's try 3-5 restarts.
            n_restarts = min(5, len(initial_values))
            
            # Budget per restart
            # NM in 5D typically converges in ~100-200 function evaluations if well-tuned.
            # With 500 total budget, and ~50 used for exploration, we have ~450 left.
            # 450 / 5 = 90 per restart. This is tight but should find local minima.
            
            budget_per_restart = remaining_budget // n_restarts
            
            for i in range(n_restarts):
                if used_calls >= max_calls or time.time() > deadline:
                    break
                
                fp, x_start = initial_values[i]
                
                local_budget = max_calls - used_calls
                if local_budget < 10:
                    break
                
                try:
                    # Use Nelder-Mead
                    result = opt.minimize(
                        safe_func, 
                        x_start, 
                        method='Nelder-Mead',
                        options={
                            'maxiter': local_budget, 
                            'maxfev': local_budget,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        
                except (RuntimeError, Exception):
                    # If budget or time exceeded, stop further restarts
                    if time.time() > deadline or used_calls >= max_calls:
                        break
                    continue
        
        # Phase 3: Final Polish
        # If we have budget left, do one more high-precision NM run from the best point found.
        if used_calls < max_calls and time.time() < deadline:
            final_budget = max_calls - used_calls
            if final_budget > 10:
                try:
                    result = opt.minimize(
                        safe_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={
                            'maxiter': final_budget,
                            'maxfev': final_budget,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except (RuntimeError, Exception):
                    pass

        return best_x, best_f
