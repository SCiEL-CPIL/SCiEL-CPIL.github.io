"""Model-and-optimize with coordinate-wise line search polish for Sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Model-based optimizer for BBOB-style Sphere functions with coordinate descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)
        
        N = self.dim

        def wrapped_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            x = np.clip(x, -5.0, 5.0)
            
            val = func(x)
            self.calls += 1
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # PHASE 1: Estimate the center x* of the Sphere using N+1 points
        D = 4.0
        points = [np.zeros(N)]
        for i in range(N):
            p = np.zeros(N)
            p[i] = D
            points.append(p)
        
        values = []
        for p in points:
            try:
                v = wrapped_func(p)
                values.append(v)
            except (ValueError, TimeoutError, Exception):
                break
        
        if len(values) < N + 1:
            # Fallback: Random search + coordinate descent
            rng = np.random.default_rng(42)
            for _ in range(20):
                if self.calls >= self.budget: break
                x = rng.uniform(-5, 5, N)
                try:
                    wrapped_func(x)
                except:
                    break
            self._coordinate_descent_polish(wrapped_func, self.best_x, deadline)
            return (self.best_x, self.best_f)

        # Solve for x* using the linear system
        A = np.zeros((N, N))
        b = np.zeros(N)
        
        x0 = points[0]
        f0 = values[0]
        
        for i in range(1, N + 1):
            xi = points[i]
            fi = values[i]
            diff = xi - x0
            norm_sq_diff = np.dot(xi, xi) - np.dot(x0, x0)
            val_diff = fi - f0
            A[i-1] = 2.0 * diff
            b[i-1] = norm_sq_diff - val_diff
        
        try:
            x_est = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x_est = np.linalg.lstsq(A, b, rcond=None)[0]
        
        x_start = np.clip(x_est, -5.0, 5.0)
        
        # Evaluate start point
        try:
            v = wrapped_func(x_start)
        except (ValueError, TimeoutError, Exception):
            pass

        # PHASE 2: Coordinate-wise line search polish
        self._coordinate_descent_polish(wrapped_func, x_start, deadline)
        
        return (self.best_x, self.best_f)

    def _coordinate_descent_polish(self, wrapped_func, x0, deadline):
        """Cyclic coordinate descent with line search for Sphere functions."""
        N = self.dim
        x = x0.copy()
        
        # Initial step sizes: use model-based estimate of curvature
        # For Sphere f(x) = ||x - x*||^2 + c, the Hessian is 2I
        # So optimal step in coordinate i is proportional to (x_i - x*_i)
        # We don't know x*_i exactly, so use adaptive steps
        
        # Start with moderate steps
        steps = np.ones(N) * 0.5
        
        max_cycles = 50
        min_step = 1e-12
        max_step = 10.0
        
        for cycle in range(max_cycles):
            if self.calls >= self.budget or time.time() > deadline:
                break
            
            # Check if converged
            if self.best_f < 1e-10:
                break
            
            improved = False
            for i in range(N):
                if self.calls >= self.budget or time.time() > deadline:
                    break
                
                # Try positive and negative directions along coordinate i
                best_local_f = self.best_f
                best_local_x = x.copy()
                
                # Binary search for optimal step in direction i
                # Try a range of step sizes
                step_sizes = [steps[i] * s for s in [1.0, 0.5, 0.25, 0.1, 2.0, 4.0]]
                step_sizes = [s for s in step_sizes if min_step < abs(s) < max_step]
                
                for sign in [1.0, -1.0]:
                    for s in step_sizes:
                        if self.calls >= self.budget or time.time() > deadline:
                            break
                        
                        dx = np.zeros(N)
                        dx[i] = sign * s
                        x_trial = x + dx
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        
                        try:
                            f_trial = wrapped_func(x_trial)
                            if f_trial < best_local_f:
                                best_local_f = f_trial
                                best_local_x = x_trial.copy()
                                improved = True
                        except (ValueError, TimeoutError, Exception):
                            break
                
                if improved and best_local_f < self.best_f:
                    x = best_local_x
                    # Adapt step size: if we improved, keep or slightly increase
                    # If we didn't improve much, decrease
                    if best_local_f < self.best_f - 1e-8:
                        steps[i] = min(steps[i] * 1.5, max_step)
                    else:
                        steps[i] = max(steps[i] * 0.7, min_step)
                else:
                    # No improvement: shrink step
                    steps[i] = max(steps[i] * 0.5, min_step)
            
            # If no improvement in entire cycle and steps are tiny, stop
            if not improved and np.max(steps) < 1e-6:
                break
            
            # Early stopping if we've converged well
            if self.best_f < 1e-8:
                break
        
        return
# EVOLVE-BLOCK-END
