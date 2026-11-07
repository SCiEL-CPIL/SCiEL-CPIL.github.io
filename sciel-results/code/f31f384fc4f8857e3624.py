import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 295.0
        d = self.dim
        budget = self.budget
        
        def _check_time():
            return time.time() - start_time < timeout
        
        # Track best
        best_x = None
        best_f = np.inf
        n_evals = 0
        
        def _update_best(x, f):
            nonlocal best_x, best_f, n_evals
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 1: Random restarts with local search to find good basins
        # Since f7 is step-ellipsoidal, the landscape is discontinuous.
        # We need to find the "step" boundaries.
        
        # Strategy:
        # 1. Sample a few random points to get a rough idea of the scale.
        # 2. Use a (1+1)-ES with restarts or a simple coordinate descent 
        #    that can handle flat regions by taking larger steps if no improvement.
        
        # Let's try a robust approach:
        # - Start with a few random points.
        # - Then use a (1+1)-ES that adapts step size based on success rate.
        # - If stuck on a plateau (no improvement for K generations), increase step size significantly.
        
        # Initial random sampling
        for _ in range(10):
            if n_evals >= budget or not _check_time():
                break
            x = np.random.uniform(-5, 5, d)
            f = func(x)
            n_evals += 1
            _update_best(x, f)
        
        if best_x is None:
            return np.zeros(d), np.inf

        x = best_x.copy()
        f = best_f
        sigma = 1.0
        chi = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # Success rate tracking for sigma adaptation
        # In (1+1)-ES, if we succeed 1/5th of the time, sigma is optimal.
        # Here we use a simple adaptive rule.
        
        # We will use a restart mechanism.
        # If no improvement for 30 generations, restart from best with larger sigma or new random point.
        
        max_no_improve = 30
        no_improve_count = 0
        
        while n_evals < budget and _check_time():
            # Generate candidate
            z = np.random.randn(d)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            f_y = func(y)
            n_evals += 1
            
            if f_y < f:
                # Success
                x = y
                f = f_y
                _update_best(x, f)
                no_improve_count = 0
                # Adapt sigma: decrease slightly if we want to be cautious, 
                # but standard (1+1) ES uses success rate.
                # Simple rule: if success, keep sigma or decrease slightly.
                # But for step functions, we might want to keep sigma large to jump over steps.
                # Let's use a success rate based adaptation.
                # We'll track successes in a window.
                # For simplicity, let's just keep sigma constant unless we stagnate.
                # Actually, let's implement a proper (1+1)-ES with 1/5 success rate target.
            else:
                # Failure
                no_improve_count += 1
            
            # Adapt sigma based on simple heuristic
            # If we have been successful recently, decrease sigma.
            # If we have been failing, increase sigma.
            # Let's use a counter.
            # This is a simplified version. A better way is to track success rate over last N steps.
            
            # Let's restart the sigma adaptation logic to be more robust for step functions.
            # If no improvement for max_no_improve, increase sigma significantly.
            if no_improve_count >= max_no_improve:
                # Stagnation detected
                # Option 1: Increase sigma
                sigma *= 2.0
                sigma = min(sigma, 10.0)
                no_improve_count = 0
                
                # Option 2: If sigma is already large, try a random restart
                if sigma > 5.0:
                    # Random restart
                    x = np.random.uniform(-5, 5, d)
                    f = func(x)
                    n_evals += 1
                    _update_best(x, f)
                    sigma = 1.0
                    no_improve_count = 0
            
            # Optional: Fine-tuning if very close to convergence?
            # Hard to tell for step functions.
            
            # If sigma becomes too small, we might be stuck in a local minimum (or on a flat region).
            # If sigma < 1e-4, increase it to explore again.
            if sigma < 1e-4:
                sigma = 0.1

        # Phase 2: Coordinate-wise refinement
        # This is useful for polishing the solution, especially if the step boundaries are aligned with axes.
        # However, for rotated ellipsoidal, this might not be as effective.
        # But it's cheap and can help.
        
        if _check_time() and n_evals < budget:
            # Try small steps in each coordinate
            step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]
            improved = True
            while improved and _check_time() and n_evals < budget:
                improved = False
                for ss in step_sizes:
                    if not _check_time() or n_evals >= budget:
                        break
                    for i in range(d):
                        for direction in [-1, 1]:
                            if not _check_time() or n_evals >= budget:
                                break
                            x_try = x.copy()
                            x_try[i] += direction * ss
                            x_try = np.clip(x_try, -5, 5)
                            f_try = func(x_try)
                            n_evals += 1
                            if f_try < f:
                                f = f_try
                                x = x_try.copy()
                                _update_best(x, f)
                                improved = True

        return best_x, best_f
