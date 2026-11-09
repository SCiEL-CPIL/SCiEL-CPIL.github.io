"""
Optimizer for BBOB f5 (Linear slope).
Uses gradient estimation to directly locate the optimal vertex.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _estimate_gradient(self, func, x_center, h, dim):
        """
        Estimate gradient at x_center using forward differences with step h.
        Returns the gradient vector.
        """
        f_center = func(x_center)
        grad = np.zeros(dim)
        for i in range(dim):
            x_test = x_center.copy()
            x_test[i] += h
            f_test = func(x_test)
            grad[i] = (f_test - f_center) / h
        return grad

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        calls_made = 0
        best_x = None
        best_fx = np.inf
        
        # Strategy:
        # 1. Estimate gradient at the center (0,0,...,0).
        # 2. The minimum of a linear function on a box is at a vertex.
        # 3. The optimal vertex is given by x_i = -5 * sign(grad_i).
        # 4. Evaluate this vertex.
        # 5. If budget allows, verify or refine (though for linear, it's exact).
        
        # Step 1: Estimate gradient
        # Use a step size that is safe within bounds if starting at 0.
        # Box is [-5, 5]. If we start at 0, h=4 is safe (0+4=4 < 5).
        # However, for linear functions, any h works. Larger h is better for SNR.
        # Let's use h=4.0 to stay safely within bounds for the forward difference.
        h = 4.0 
        x_center = np.zeros(dim)
        
        # We need to call func(x_center) first, then func(x_center + h*e_i) for each i.
        # Total calls for gradient: 1 + dim.
        
        if budget < 1 + dim:
            # If we don't have enough budget for gradient estimation, 
            # just do a random search or return 0. 
            # But budget is 500, dim is 5, so 6 calls is fine.
            pass
            
        grad = self._estimate_gradient(func, x_center, h, dim)
        calls_made += 1 + dim # 1 for center, dim for derivatives
        
        # Step 2: Construct optimal vertex
        # For linear f(x) = a^T x + b, min is at x_i = -5 if a_i > 0, else 5.
        # sign(a_i) is 1 if a_i > 0, -1 if a_i < 0.
        # So x_i = -5 * sign(a_i).
        # If a_i is 0, any x_i is fine, pick 0 or 5. sign(0) is 0 in numpy, 
        # but we want a vertex. If grad is 0, the function is flat in that dim.
        # Let's handle 0 by picking 5.0 arbitrarily.
        signs = np.sign(grad)
        # Where sign is 0, set to 1 (arbitrary choice)
        signs[signs == 0] = 1.0
        x_opt = -5.0 * signs
        
        # Step 3: Evaluate the optimal vertex
        if calls_made < budget:
            f_opt = func(x_opt)
            calls_made += 1
            best_x = x_opt.copy()
            best_fx = f_opt
            
        # Step 4: Verification / Fallback
        # For a linear function, x_opt IS the minimum. 
        # However, if the gradient estimation was slightly off due to noise 
        # (though unlikely for linear) or if the function isn't perfectly linear,
        # we might want to check neighbors.
        # But the task specifies "Linear slope".
        # To be safe against any hidden non-linearity or rotation issues 
        # that might shift the minimum slightly off the vertex (which shouldn't happen 
        # for a pure linear function on a box, the min is always at a vertex),
        # we can check the "next best" vertices if we have budget.
        
        # The next best vertices are those differing in the coordinate with the 
        # smallest absolute gradient component (least impact on changing the sign).
        # But since we are sure it's linear, the current x_opt is the global min.
        
        # If we have extra budget, we can just use it to confirm or do nothing.
        # AOCC benefits from early good values. We got it at call 1+dim+1.
        # For dim=5, that's call 7.
        
        # If budget remains, we could potentially refine if we suspected non-linearity,
        # but for f5, it's linear.
        
        return best_x, best_fx

# EVOLVE-BLOCK-END
