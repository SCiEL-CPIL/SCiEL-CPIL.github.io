"""Rewritten optimizer for BBOB f5 (Linear slope) using exact affine model fitting."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Exact linear-model optimizer for f(x) = w^T x + b on [-5,5]^d.
    
    Strategy:
      1. Evaluate at d+1 points forming an affine basis (origin + axis points).
      2. Solve for gradient w and offset b exactly.
      3. The minimum of a linear function over a box is at the vertex opposing the gradient.
      4. Verify with one final call at that vertex.
      5. Use remaining budget to refine (though for a true linear function, no refinement needed).
      
    Since f5 is a linear slope, this should find the exact optimum in d+2 calls (d+1 for model, 1 for verification).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # Step 1: Build affine basis
        # Points: x0 = 0, x1 = e1, x2 = e2, ..., xd = ed
        # where ei is the i-th unit vector
        # f(x0) = b
        # f(x_i) = w_i + b  => w_i = f(x_i) - f(x0)
        
        # We need d+1 evaluations for the model.
        # But we must be careful: the function clips to [-5,5].
        # The origin 0 is in the box. The unit vectors are also in the box.
        
        # Let's use a slightly more robust basis if possible, but origin+unit vectors is fine.
        
        # Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        
        # Evaluate at unit vectors
        w = np.zeros(d)
        for i in range(d):
            xi = np.zeros(d)
            xi[i] = 1.0
            fi = func(xi)
            w[i] = fi - f0
        
        # Now we have the exact gradient w and offset b = f0.
        # The linear function is f(x) = w^T x + b.
        # To minimize over [-5,5]^d, we choose x_i = -5 if w_i > 0, else +5 if w_i < 0.
        # If w_i == 0, either value works.
        
        x_opt = np.zeros(d)
        for i in range(d):
            if w[i] > 0:
                x_opt[i] = -5.0
            elif w[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0  # arbitrary, doesn't matter
        
        # Verify the optimum
        f_opt_est = func(x_opt)
        
        # We've used d+2 calls so far (1 for origin, d for unit vectors, 1 for verification).
        # Total calls so far: d + 2 = 5 + 2 = 7 for d=5.
        # Budget is 500. We have plenty of calls left.
        
        # Since the function is exactly linear, no further refinement is needed.
        # However, to be safe and to potentially handle any numerical issues or if the function
        # isn't perfectly linear (though the task says it is), we could do a small local search.
        # But the task specifies f5 is Linear slope, so the exact solution should be perfect.
        
        # For the AOCC score, early progress counts. We've already found the optimum in 7 calls.
        # The best value b_i for i >= 7 will be f_opt_est (assuming no better value exists).
        # The AOCC will be very high because we reach the optimum quickly.
        
        # We don't need to use the remaining budget, but we must not exceed it.
        # The evaluator ignores the return value of __call__, so we can just return.
        
        return x_opt, f_opt_est
# EVOLVE-BLOCK-END
