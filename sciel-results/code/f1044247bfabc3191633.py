import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        budget_used = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        # Phase 1: Gradient Estimation using central differences at the origin
        # For better accuracy, use central differences where possible
        x0 = np.zeros(dim)
        f0 = func(x0)
        budget_used += 1
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        # Step size for gradient estimation - use a moderate step
        delta = 0.5
        grad = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline or budget_used >= budget:
                break
            e_i = np.zeros(dim)
            e_i[i] = 1.0
            
            # Central difference: f(x+delta*e_i) - f(x-delta*e_i) / (2*delta)
            x_plus = np.clip(x0 + delta * e_i, -5, 5)
            x_minus = np.clip(x0 - delta * e_i, -5, 5)
            
            f_plus = func(x_plus)
            budget_used += 1
            if f_plus < best_f:
                best_f = f_plus
                best_x = x_plus.copy()
            
            f_minus = func(x_minus)
            budget_used += 1
            if f_minus < best_f:
                best_f = f_minus
                best_x = x_minus.copy()
                
            grad[i] = (f_plus - f_minus) / (2.0 * delta)

        # Phase 2: Informed Jump to vertex
        # For a linear function f(x) = c + g^T x, the minimum on [-5,5]^dim is at the vertex
        # where x_i = -5 if g_i > 0, x_i = 5 if g_i < 0, x_i = 0 if g_i == 0.
        # This gives a huge jump in quality, crucial for AOCC.
        x_jump = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 1e-8:
                x_jump[i] = -5.0
            elif grad[i] < -1e-8:
                x_jump[i] = 5.0
            else:
                x_jump[i] = 0.0

        if budget_used < budget:
            f_jump = func(x_jump)
            budget_used += 1
            if f_jump < best_f:
                best_f = f_jump
                best_x = x_jump.copy()

        # Phase 3: Local Refinement - verify the vertex and refine if needed
        # For a linear function, once at the correct vertex, we're done.
        # But if gradient estimation was slightly off, we may need to refine.
        # Use a targeted search: evaluate neighboring vertices and nearby points.
        
        x = best_x.copy()
        fx = best_f
        
        # First, try all 2^5 = 32 vertices if we have budget
        # This is expensive but guarantees finding the optimal vertex for a linear function
        if budget_used + 32 <= budget and time.time() < deadline:
            # Generate all vertices
            vertices = np.array([[(-1)**(1 - (i >> j) & 1) * 5.0 for j in range(dim)] for i in range(2**dim)])
            for v in vertices:
                if time.time() > deadline or budget_used >= budget:
                    break
                fv = func(v)
                budget_used += 1
                if fv < best_f:
                    best_f = fv
                    best_x = v.copy()
                    x = v.copy()
                    fx = fv
        
        # Then do a local refinement around the best point found
        sigma = 0.1
        fail_count = 0
        max_fails = 10
        
        while budget_used < budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget_used += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.05, 1.0)
                fail_count = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.95
                fail_count += 1
                
            if fail_count > max_fails:
                sigma = max(sigma, 0.01)
                fail_count = 0
                
            if sigma < 1e-10:
                break

        return best_x, best_f
