# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using Cyclic Coordinate Descent with Parabolic Line Search.
    Exploits the quadratic nature of the BBOB Sphere function (f1) for rapid convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls_made = 0
        self._start_time = 0.0
        self._best_x = None
        self._best_f = np.inf

    def _evaluate(self, func, x):
        """Evaluate function with call counting and deadline guard."""
        if self._calls_made >= self.budget:
            return None
        if time.time() - self._self._start_time > 325: # Safety margin
            return None
        self._calls_made += 1
        try:
            val = func(x)
            val = float(val)
            if np.isnan(val) or np.isinf(val):
                return 1e10
            return val
        except Exception:
            return 1e10

    def _line_search_1d(self, func, x, dim, step_size):
        """
        Perform a 1D line search along dimension 'dim' of x.
        Uses parabolic interpolation to find the minimum.
        Returns the new value and the updated x.
        """
        current_val = self._evaluate(func, x)
        if current_val is None:
            return current_val, x

        # Try to bracket the minimum
        # Start with a step, if it improves, expand or refine.
        # For Sphere, a simple golden section or parabolic fit works well.
        
        x1 = x.copy()
        x1[dim] -= step_size
        f1 = self._evaluate(func, x1)
        if f1 is None:
            return current_val, x

        x2 = x.copy()
        x2[dim] += step_size
        f2 = self._evaluate(func, x2)
        if f2 is None:
            return current_val, x

        # We have points (x0-d, f0-d), (x0, f0), (x0+d, f0+d)
        # If f1 or f2 is better, we can fit a parabola.
        # If both are worse, the minimum is likely very close, or we need to shrink step.
        
        # Determine the best of the 3 points
        if f1 < current_val and f1 <= f2:
            # Minimum is likely to the left
            # Fit parabola through (x0-d, f1), (x0, f0), (x0+d, f2)
            # Or just do a finer search in [x0-2d, x0]
            # Let's use parabolic interpolation
            h = step_size
            # Coefficients for parabola y = a*t^2 + b*t + c where t is offset from x0
            # Points: (-h, f1), (0, f0), (h, f2)
            # f(t) = a*t^2 + b*t + c
            # c = f0
            # f1 = a*h^2 - b*h + f0  => a*h^2 - b*h = f1 - f0
            # f2 = a*h^2 + b*h + f0  => a*h^2 + b*h = f2 - f0
            # Sum: 2*a*h^2 = f1 + f2 - 2*f0 => a = (f1 + f2 - 2*f0) / (2*h^2)
            # Diff: 2*b*h = f2 - f1 => b = (f2 - f1) / (2*h)
            denom = 2 * h * h
            if abs(denom) < 1e-15:
                return current_val, x
            a = (f1 + f2 - 2 * current_val) / denom
            b = (f2 - f1) / (2 * h)
            
            if a > 1e-15: # Convex
                t_min = -b / (2 * a)
                # Clamp t_min to reasonable range, e.g., [-2h, 2h]
                t_min = np.clip(t_min, -2 * h, 2 * h)
                x_new = x.copy()
                x_new[dim] += t_min
                f_new = self._evaluate(func, x_new)
                if f_new is not None and f_new < current_val:
                    return f_new, x_new
            else:
                # If not convex or flat, just take the best of f1, f2
                if f1 < f2:
                    return f1, x1
                else:
                    return f2, x2

        elif f2 < current_val and f2 <= f1:
            # Minimum is likely to the right
            # Same parabola fit
            denom = 2 * h * h
            if abs(denom) < 1e-15:
                return current_val, x
            a = (f1 + f2 - 2 * current_val) / denom
            b = (f2 - f1) / (2 * h)
            
            if a > 1e-15:
                t_min = -b / (2 * a)
                t_min = np.clip(t_min, -2 * h, 2 * h)
                x_new = x.copy()
                x_new[dim] += t_min
                f_new = self._evaluate(func, x_new)
                if f_new is not None and f_new < current_val:
                    return f_new, x_new
            else:
                if f1 < f2:
                    return f1, x1
                else:
                    return f2, x2
        else:
            # Current point is best among the three.
            # Shrink step size for next iteration or try smaller steps
            # For now, return current best. The outer loop will reduce step size.
            return current_val, x

    def __call__(self, func):
        self._start_time = time.time()
        D = self.dim
        budget = self.budget
        
        # Phase 1: Initialization
        # Sample a few points to find a good starting location
        rng = np.random.default_rng(42)
        
        # Center
        x0 = np.zeros(D)
        f0 = self._evaluate(func, x0)
        
        if f0 is None:
            return (np.zeros(D), 0.0)
        
        self._best_x = x0.copy()
        self._best_f = f0

        # Try some axis-aligned points to potentially jump closer
        for i in range(D):
            for sign in [-1, 1]:
                if self._calls_made >= budget:
                    break
                x_test = np.zeros(D)
                x_test[i] = sign * 3.0
                f_test = self._evaluate(func, x_test)
                if f_test is not None and f_test < self._best_f:
                    self._best_f = f_test
                    self._best_x = x_test.copy()
        
        # Random samples
        n_init = min(20, budget // 10)
        for _ in range(n_init):
            if self._calls_made >= budget:
                break
            x_test = rng.uniform(-5.0, 5.0, size=D)
            f_test = self._evaluate(func, x_test)
            if f_test is not None and f_test < self._best_f:
                self._best_f = f_test
                self._best_x = x_test.copy()

        # Phase 2: Cyclic Coordinate Descent
        # Start with a moderate step size
        step_size = 1.0
        x_current = self._best_x.copy()
        f_current = self._best_f
        
        # We will iterate until budget or time runs out
        # Adaptive step size: if no improvement, shrink step. If improvement is large, maybe keep or grow?
        # For Sphere, shrinking is safer as we approach the minimum.
        
        stagnant_counter = 0
        
        while self._calls_made < budget and time.time() - self._start_time < 325:
            improved = False
            old_f = f_current
            
            for dim in range(D):
                if self._calls_made >= budget:
                    break
                if time.time() - self._start_time > 325:
                    break
                
                # Line search along dimension 'dim'
                f_new, x_new = self._line_search_1d(func, x_current, dim, step_size)
                
                if f_new is not None:
                    if f_new < f_current - 1e-12:
                        f_current = f_new
                        x_current = x_new
                        improved = True
                        
                        if f_current < self._best_f:
                            self._best_f = f_current
                            self._best_x = x_current.copy()
                    else:
                        # No improvement from line search, maybe step is too small or we are at minimum
                        pass
            
            if improved:
                stagnant_counter = 0
                # Slightly reduce step size to refine
                # But not too fast, otherwise we might miss the minimum if step was too big initially
                step_size *= 0.7 
            else:
                stagnant_counter += 1
                if stagnant_counter > 2:
                    # Stagnant. Reduce step size significantly.
                    step_size *= 0.5
                    if step_size < 1e-8:
                        # Converged or stuck. Try a small random restart?
                        # Or just break if we are very precise.
                        # For Sphere, if step is tiny and no improvement, we are likely at min.
                        # But let's try a few random perturbations to ensure we didn't get stuck in a local min (unlikely for Sphere).
                        # Actually, Sphere has no local minima. So if we are stuck, it's numerical precision.
                        break
                else:
                    step_size *= 0.8

        return (self._best_x if self._best_x is not None else np.zeros(D),
                self._best_f if self._best_f < np.inf else 0.0)
# EVOLVE-BLOCK-END
