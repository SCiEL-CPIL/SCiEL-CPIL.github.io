"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer for linear slope functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: finish well before 300s
        deadline = start_time + 290.0

        rng = np.random.default_rng(0)
        
        # Phase 1: Estimate Gradient
        # Use center of the box
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        best_f = f0
        best_x = x0.copy()
        
        calls = 1
        
        # Use a relatively large epsilon to avoid numerical noise
        # but small enough to stay within bounds and assume linearity
        eps = 1.0
        
        grad = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
                
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            fp = func(x_plus)
            fm = func(x_minus)
            calls += 2
            
            if fp < best_f:
                best_f = fp
                best_x = x_plus.copy()
            if fm < best_f:
                best_f = fm
                best_x = x_minus.copy()
                
            grad[i] = (fp - fm) / (2.0 * eps)
            
        # Phase 2: Jump to predicted optimal vertex
        # For a linear function f(x) = c + g^T x, the minimum on [-5, 5]^dim
        # is at the vertex where x_i = -5 if g_i > 0, and x_i = 5 if g_i < 0.
        x_vertex = np.zeros(self.dim)
        for i in range(self.dim):
            if grad[i] > 0:
                x_vertex[i] = -5.0
            else:
                x_vertex[i] = 5.0
                
        f_vertex = func(x_vertex)
        calls += 1
        
        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()
            
        # Phase 3: Polish with (1+1)-ES
        # In case the function is not perfectly linear or gradient estimation was noisy.
        # We use the remaining budget to refine the solution.
        sigma = 0.5
        x_curr = best_x.copy()
        f_curr = best_f
        
        # Use remaining calls for local search
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate candidate
            y = x_curr + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5.0, 5.0)
            
            fy = func(y)
            calls += 1
            
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                x_curr = y.copy()
                f_curr = fy
                sigma *= 1.1  # Increase step size on success
            else:
                sigma *= 0.9  # Decrease step size on failure
                if sigma < 1e-7:
                    # Convergence reached
                    break
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
