# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    Strategy:
    1. Start at a random vertex in {-5, 5}^d.
    2. Use Greedy Coordinate Descent on the vertices:
       For each dimension i, flip the coordinate to the opposite bound.
       If the function value improves, keep the flip.
       This guarantees finding the global minimum for a linear function
       in exactly d+1 calls (1 initial + d flips).
    3. Stall with the best point for the remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        
        d = self.dim
        calls_made = 0
        
        # Step 1: Random vertex initialization
        # Using random choice to avoid systematic bias
        x_best = np.random.choice([-5.0, 5.0], size=d)
        f_best = func(x_best)
        calls_made += 1
        
        # Step 2: Greedy Coordinate Descent
        # Randomize dimension order to break ties or avoid specific patterns
        indices = np.random.permutation(d)
        
        for i in indices:
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            # Create a test point by flipping the i-th coordinate
            x_test = x_best.copy()
            x_test[i] = -x_test[i]
            
            f_test = func(x_test)
            calls_made += 1
            
            # Update best if improvement found
            if f_test < f_best:
                f_best = f_test
                x_best = x_test
        
        # Step 3: Stall with the best point found
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(x_best)
            calls_made += 1
        
        return x_best, f_best
# EVOLVE-BLOCK-END
