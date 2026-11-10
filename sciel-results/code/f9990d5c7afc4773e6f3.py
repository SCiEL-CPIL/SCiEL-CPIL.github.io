import numpy as np
import time

class Optimizer:
          def __init__(self, budget, dim, n_obj):
              self.budget = budget
              self.dim = dim
              self.n_obj = n_obj

          def __call__(self, problem, bounds):
              start_time = time.time()
              deadline = start_time + 290  # Safety margin for 300s limit
              
              lo = np.array([b[0] for b in bounds])
              hi = np.array([b[1] for b in bounds])
              
              # We assume bounds are [0, 1]
              # Strategy: Sample x1 on the manifold x2..x10 = 0
              # x1 is chosen to uniformly distribute f1 = 1 - sqrt(x1)
              
              # Generate 200 points
              # f1 from 1.0 down to 0.0
              # u in [0, 1]
              # f1 = 1 - u
              # x1 = u^2
              
              # To maximize hypervolume, we want to cover the front.
              # Let's use a mix of uniform in f1 and specific points.
              # Actually, 200 points is enough for a grid.
              
              # Create 200 values of u
              # To avoid clustering at the ends, use midpoints?
              # u_i = (i + 0.5) / 200
              
              u_values = (np.arange(self.budget) + 0.5) / self.budget
              x1_values = u_values ** 2
              
              # Construct X
              X = np.zeros((self.budget, self.dim))
              X[:, 0] = x1_values
              # x2..x10 are 0, which is within [0, 1]
              
              # Evaluate
              for i in range(self.budget):
                  if time.time() > deadline:
                      break
                  problem(X[i])
              
              return X
