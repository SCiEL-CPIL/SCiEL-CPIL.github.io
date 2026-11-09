import numpy as np
import time

class Optimizer:
       def __init__(self, budget, dim):
           self.budget = budget
           self.dim = dim

       def __call__(self, func):
           start_time = time.time()
           deadline = start_time + 290.0
           
           # Initial point: Center
           x = np.zeros(self.dim)
           fx = func(x)
           best_x = x.copy()
           best_f = fx
           
           sigma = 5.0 # Large initial step to cover [-5, 5]
           calls = 1
           rng = np.random.default_rng(42)
           
           while calls < self.budget:
               if time.time() > deadline:
                   break
               
               # Generate candidate
               step = sigma * rng.standard_normal(self.dim)
               y = x + step
               y = np.clip(y, -5, 5)
               
               fy = func(y)
               calls += 1
               
               if fy < fx:
                   x = y
                   fx = fy
                   if fx < best_f:
                       best_f = fx
                       best_x = x.copy()
                   sigma *= 1.1 # Expand
               else:
                   sigma *= 0.8 # Shrink
               
               # Safety: prevent sigma from becoming too small or large
               if sigma < 1e-6:
                   sigma = 1e-6
               if sigma > 10.0:
                   sigma = 10.0
                   
           return best_x, best_f
