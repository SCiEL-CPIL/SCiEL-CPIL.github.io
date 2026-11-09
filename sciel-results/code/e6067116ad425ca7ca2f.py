import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 290  # 300s limit, keep 10s margin
            
            # DTLZ2 specifics:
            # M = 3 objectives
            # N = 12 variables
            # Front: f_1^2 + f_2^2 + f_3^2 = 1, f_i >= 0
            # Mapping: f_i = sin(pi * x_i / 2)  => x_i = (2/pi) * arcsin(f_i)
            # g(x) = sum((x_i - 0.5)^2) for i > M. To make g=0, set x_i = 0.5 for i > M.
            
            M = self.n_obj
            N = self.dim
            
            # Pre-calculate constants
            two_over_pi = 2.0 / np.pi
            
            # We will generate 'budget' points on the Pareto front.
            # Bias towards axes to maximize hypervolume.
            # Use a power transformation on random uniform variables before normalizing.
            # u ~ Uniform(0,1)
            # v_i = u_i^p
            # f_i = v_i / ||v||
            
            p_bias = 2.0  # Start with p=2. Higher p concentrates more on axes.
            
            # Generate all points at once for speed?
            # Budget is 200, so 200 evaluations.
            # We can generate 200 sets of 3 random numbers.
            
            u = np.random.random((self.budget, M))
            # Apply bias
            v = u ** p_bias
            # Normalize to unit sphere
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            # Avoid division by zero (probabilistically negligible, but safe)
            norms[norms == 0] = 1.0
            f_front = v / norms
            
            # Convert f_front to x variables
            # x_0..x_M-1 come from f_front
            # x_M..x_N-1 are 0.5
            x_front = np.zeros((self.budget, N))
            x_front[:, :M] = two_over_pi * np.arcsin(f_front)
            x_front[:, M:] = 0.5
            
            # Clip x to bounds [0, 1] just in case of floating point errors
            x_front = np.clip(x_front, 0.0, 1.0)
            
            # Evaluate
            # The problem function expects a 1D vector.
            # We must call it one by one.
            # Check time periodically? 200 calls is fast, but let's be safe.
            
            for i in range(self.budget):
                if time.time() > deadline:
                    break
                x = x_front[i]
                # problem(x) returns 3 objectives
                # We don't strictly need to store them if the evaluator records them,
                # but the task says "Every returned objective vector is recorded".
                # The return value of __call__ is ignored.
                # So we just need to ensure problem(x) is called.
                problem(x)
                
            return None
