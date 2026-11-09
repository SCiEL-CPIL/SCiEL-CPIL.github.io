import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Global state
        best_x = np.zeros(n)
        best_f = np.inf
        eval_count = 0
        
        class _BudgetExceeded(Exception):
            pass
        
        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise _BudgetExceeded
            if eval_count >= budget:
                raise _BudgetExceeded
            try:
                f_val = func(np.array(x, dtype=np.float64))
                if not np.isfinite(f_val):
                    f_val = 1e10
            except Exception:
                f_val = 1e10
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=np.float64)
            return f_val

        rng = np.random.default_rng(42)
        
        # Phase 1: Initial sampling to find a good starting point
        n_samples = min(20, budget)
        
        # Try origin and structured points
        safe_func(np.zeros(n))
        
        for offset in [0.5, -0.5, 1.0, -1.0]:
            if eval_count < budget:
                safe_func(np.full(n, offset))
        
        # Random samples for coverage
        for i in range(1, n_samples):
            if eval_count >= budget:
                break
            x = rng.uniform(-5, 5, n)
            safe_func(x)

        if best_f == np.inf:
            return np.zeros(n), 0.0

        # Phase 2: Coordinate Descent with adaptive step size
        # Lunacek bi-Rastrigin has axis-aligned structure, so coordinate descent can be effective
        
        x_current = best_x.copy()
        step_sizes = np.ones(n) * 0.5  # Initial step sizes per coordinate
        
        # Number of coordinate descent iterations
        max_cd_iterations = 300
        cd_iter = 0
        
        while eval_count < budget and time.time() < deadline and cd_iter < max_cd_iterations:
            cd_iter += 1
            
            improved = False
            
            for i in range(n):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Try positive and negative steps along coordinate i
                x_plus = x_current.copy()
                x_plus[i] += step_sizes[i]
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = safe_func(x_plus)
                
                if eval_count >= budget or time.time() > deadline:
                    break
                
                x_minus = x_current.copy()
                x_minus[i] -= step_sizes[i]
                x_minus = np.clip(x_minus, -5, 5)
                f_minus = safe_func(x_minus)
                
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Determine best move
                f_current = safe_func(x_current)  # Re-evaluate to get current value
                if eval_count >= budget or time.time() > deadline:
                    break
                
                best_f_local = min(f_plus, f_minus, f_current)
                
                if f_plus < f_current - 1e-10:
                    x_current = x_plus
                    improved = True
                    step_sizes[i] *= 1.1  # Increase step size if improving
                elif f_minus < f_current - 1e-10:
                    x_current = x_minus
                    improved = True
                    step_sizes[i] *= 1.1
                else:
                    # No improvement, decrease step size
                    step_sizes[i] *= 0.7
                
                # Keep step sizes in reasonable bounds
                step_sizes[i] = np.clip(step_sizes[i], 0.001, 5.0)
            
            # If no improvement in this full cycle, add random perturbation to escape local minima
            if not improved:
                # Random perturbation with decaying magnitude
                perturbation = rng.normal(0, 0.1, n)
                x_perturbed = np.clip(x_current + perturbation, -5, 5)
                f_perturbed = safe_func(x_perturbed)
                
                if f_perturbed < best_f - 1e-10:
                    x_current = x_perturbed
                    best_x = x_current.copy()
                    best_f = f_perturbed
                else:
                    # If still no improvement, reduce all step sizes
                    step_sizes *= 0.8
                    step_sizes = np.clip(step_sizes, 0.001, 5.0)
            
            # Update global best
            if safe_func(x_current) < best_f - 1e-10:
                best_x = x_current.copy()
                best_f = safe_func(x_current)
        
        # Phase 3: Final polish - fine-grained local search
        remaining = budget - eval_count
        if remaining > 5 and time.time() < deadline - 5:
            polish_sigma = 0.05
            for _ in range(min(remaining // 3, 30)):
                if time.time() > deadline - 5 or eval_count >= budget:
                    break
                x = np.clip(best_x + rng.normal(0, polish_sigma, n), -5, 5)
                f_val = safe_func(x)
                polish_sigma *= 0.9
        
        return best_x, best_f
